"""
zkLLM ViT 32 blocks 完整验证脚本

对 jina-v4 ViT 每个 block 按完整 Transformer 顺序运行 8 步证明，
并对所有有权重承诺的步骤做 IPA fold + 承诺绑定验证。

ViT 结构（Qwen2_5_VLVisionBlock）：
  norm1 (RMSNorm)、fused QKV (Linear(1280,3840))、o_proj、norm2、MLP(SwiGLU)
  - num_heads=16, head_dim=80, MHA (kv_dim=1280, num_kv_heads=16)
  - Window Attention 块（除 7,15,23,31 外）：Step 3 拆分为 16×seq=64（64²=2^12 满足 NTT 约束）
  - Full Attention 块（7,15,23,31）：seq=1024（1024²=2^20）
  - 权重 W 通过 IPA 证明；bias 是公开参数，单独保存供公开验证

证明步骤（每块）：
  Step 1: input_layernorm (norm1)
  Step 2: self-attn linear (q/k/v，从 fused QKV 拆分后的 W)
  Step 3: self-attn attn  (MHA zkSoftmax)
  Step 4: self-attn o_proj
  Step 5: skip connection 1
  Step 6: post_attention_layernorm (norm2)
  Step 7: FFN (SwiGLU，复用 swiglu-table.bin)
  Step 8: skip connection 2

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/verify_vit.py [--blocks 0 7] [--workdir zkllm-workdir/jina-v4]
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent.resolve()
BIN_DIR   = ROOT / "src" / "zkllm"
ZKLLM_CWD = ROOT / "src" / "zkllm"

VIT_HIDDEN    = 1280
VIT_INTER     = 3420
VIT_KV_HEADS  = 16      # MHA: num_kv_heads = num_heads = 16
SEQ_LEN       = 1024
RMS_EPS       = 1e-6
SCALE         = 1 << 16

# Window vs Full Attention blocks (fullatt_block_indexes from jina-v4 ViT config)
# Blocks 7,15,23,31 use full 1024-token attention; all others use 16 windows of 64 tokens.
# 64² = 4096 = 2^12 satisfies the zkAttn NTT constraint.
FULL_ATT_BLOCKS = {7, 15, 23, 31}
WIN_SEQ  = 64               # tokens per window
NUM_WINS = SEQ_LEN // WIN_SEQ  # = 16

_P_FP    = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP    = pow(2, 384, _P_FP)
_R_FP_INV = pow(_R_FP, -1, _P_FP)
_P_FR    = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


# ── Window Attention 拆分工具 ────────────────────────────────────────────────

def _split_qkv_for_windows(vit_wd: Path, prefix: str,
                            n_real: int = SEQ_LEN, n_wins: int = NUM_WINS) -> bool:
    """把 linear 步骤写出的全序列 temp_Q/K/V 拆成 n_wins 个 WIN_SEQ×dim 窗口文件。
    最后一个窗口如不足 WIN_SEQ 则零填充，保证每窗口文件大小一致（64²=2^12 满足 NTT）。
    """
    q_src = vit_wd / f"{prefix}-temp_Q.bin"
    if not q_src.exists():
        return False
    Q = np.fromfile(str(q_src),               dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    K = np.fromfile(str(vit_wd/f"{prefix}-temp_K.bin"), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    V = np.fromfile(str(vit_wd/f"{prefix}-temp_V.bin"), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    for w in range(n_wins):
        sl = slice(w * WIN_SEQ, min((w + 1) * WIN_SEQ, n_real))
        chunk_q, chunk_k, chunk_v = Q[sl], K[sl], V[sl]
        if chunk_q.shape[0] < WIN_SEQ:
            pad = WIN_SEQ - chunk_q.shape[0]
            chunk_q = np.vstack([chunk_q, np.zeros((pad, VIT_HIDDEN), np.int32)])
            chunk_k = np.vstack([chunk_k, np.zeros((pad, VIT_HIDDEN), np.int32)])
            chunk_v = np.vstack([chunk_v, np.zeros((pad, VIT_HIDDEN), np.int32)])
        wp = f"{prefix}-win{w}"
        chunk_q.tofile(str(vit_wd / f"{wp}-temp_Q.bin"))
        chunk_k.tofile(str(vit_wd / f"{wp}-temp_K.bin"))
        chunk_v.tofile(str(vit_wd / f"{wp}-temp_V.bin"))
    return True


def _get_seq_params(h_in_path: Path, block_idx: int):
    """h_in 文件大小 → (n_real, seq_use, n_wins)。

    window 块：seq_use = n_real，n_wins = ceil(n_real / WIN_SEQ)
    full-attn 块：seq_use = 下一个 2^k ≥ n_real（供 zkAttn NTT），n_wins = 1
    文件不存在时（随机数模式）回退到 (SEQ_LEN, SEQ_LEN, NUM_WINS)。
    """
    if not h_in_path.exists() or h_in_path.stat().st_size == 0:
        return SEQ_LEN, SEQ_LEN, NUM_WINS
    n_real = h_in_path.stat().st_size // 4 // VIT_HIDDEN
    if block_idx in FULL_ATT_BLOCKS:
        pad = 1
        while pad < n_real:
            pad <<= 1
        return n_real, pad, 1
    n_wins = (n_real + WIN_SEQ - 1) // WIN_SEQ
    return n_real, n_real, n_wins


# ── IPA 验证工具 ──────────────────────────────────────────────────────────────

def _read_fp_mont(data, off):
    return (int.from_bytes(data[off:off+48], "little") * _R_FP_INV) % _P_FP

def _read_g1_jacobian(data, off):
    from py_ecc.optimized_bls12_381 import FQ, Z1
    X = _read_fp_mont(data, off)
    Y = _read_fp_mont(data, off+48)
    Z = _read_fp_mont(data, off+96)
    if Z == 0:
        return Z1
    zi = pow(Z, _P_FP-2, _P_FP)
    return (FQ((X*zi*zi)%_P_FP), FQ((Y*zi*zi*zi)%_P_FP), FQ(1))

def _read_fr(data, off):
    return int.from_bytes(data[off:off+32], "little")

def _load_commitment(path):
    data = open(path, "rb").read()
    return [_read_g1_jacobian(data, i*144) for i in range(len(data)//144)]

def _eval_g1_ml(pts, u_vec):
    from py_ecc.optimized_bls12_381 import add, multiply, Z1
    pts = list(pts)
    for u in u_vec:
        omu = (1-u) % _P_FR
        new = []
        for i in range(0, len(pts), 2):
            a, b = pts[i], (pts[i+1] if i+1<len(pts) else Z1)
            new.append(add(multiply(a, omu), multiply(b, u)))
        pts = new
    return pts[0]

def verify_ipa_cpp(proof_path: str, com_path: str, gpu_id: int = 0) -> dict:
    """调用 C++ GPU verify-ipa binary 验证 IPA proof。"""
    bin_path = BIN_DIR / "verify-ipa"
    if not bin_path.exists():
        return verify_ipa_python(proof_path, com_path)
    r = subprocess.run(
        [str(bin_path), proof_path, com_path],
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    if r.returncode not in (0, 1):
        return {"fold_ok": False, "binding_ok": False,
                "error": r.stderr.decode(errors="replace")}
    try:
        return json.loads(r.stdout.decode().strip())
    except json.JSONDecodeError:
        return {"fold_ok": False, "binding_ok": False,
                "error": r.stdout.decode(errors="replace")}

def verify_ipa_python(proof_path: str, com_path: str) -> dict:
    """Python py_ecc fallback（仅在 C++ binary 不可用时使用）。"""
    from py_ecc.optimized_bls12_381 import add, multiply, eq
    data = open(proof_path, "rb").read()
    assert data[:4] in (b"IPA\x00", b"\x00API"), f"Bad magic: {data[:4]!r}"
    k       = struct.unpack_from("<I", data, 4)[0]
    com_log = struct.unpack_from("<I", data, 8)[0]
    off     = 12
    C_init  = _read_g1_jacobian(data, off); off += 144
    u_out   = [_read_fr(data, off+i*32) for i in range(com_log)]; off += com_log*32
    u_in    = [_read_fr(data, off+i*32) for i in range(k)];       off += k*32
    rounds  = []
    for _ in range(k):
        L0 = _read_g1_jacobian(data, off); off += 144
        L1 = _read_g1_jacobian(data, off); off += 144
        rounds.append((L0, L1))
    g_final = _read_g1_jacobian(data, off); off += 144
    w_final = _read_fr(data, off)
    C = C_init
    for u, (L0, L1) in zip(u_in, rounds):
        omu = (1-u) % _P_FR
        C = add(add(multiply(L0, (omu*omu)%_P_FR),
                    multiply(C,  (u*omu) %_P_FR)),
                multiply(L1, (u*u)  %_P_FR))
    fold_ok = eq(C, multiply(g_final, w_final))
    binding_ok = None
    if os.path.exists(com_path):
        try:
            binding_ok = eq(C_init, _eval_g1_ml(_load_commitment(com_path), u_out))
        except Exception:
            binding_ok = False
    return {"fold_ok": fold_ok, "binding_ok": binding_ok}


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_input(path, rows, cols):
    if not path.exists():
        rng = np.random.default_rng(abs(hash(str(path))) % (2**31))
        (rng.standard_normal((rows, cols)) * SCALE).astype(np.int32).tofile(str(path))

def _rms_inv(input_path, out_path, seq_len, embed_dim):
    X = np.fromfile(str(input_path), dtype=np.int32).reshape(seq_len, embed_dim) / SCALE
    rms_inv = 1.0 / np.sqrt((X**2).mean(axis=1) + RMS_EPS)
    (rms_inv * SCALE).round().astype(np.int32).tofile(str(out_path))

def _run(cmd, cwd, gpu_id=0, env=None):
    r = subprocess.run(cmd, capture_output=True, cwd=cwd,
                       env=env or {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    return r.returncode, r.stderr.decode(errors="replace")



def _verify_proofs(proof_specs, workdir: Path, gpu_id: int = 0):
    # Resolve commitment paths and separate out missing files up front.
    valid   = []   # [(name, proof_path, com_path)]
    results = {}
    for name, path in proof_specs:
        if not Path(path).exists():
            results[name] = None
            continue
        com_path = path.replace("-ipa-proof.bin", ".weight-commitment.bin")
        main_com = workdir / Path(com_path).name
        if main_com.exists():
            com_path = str(main_com)
        if not os.path.exists(com_path):
            results[name] = {"fold_ok": None, "binding_ok": None,
                             "error": "commitment not found"}
            continue
        valid.append((name, path, com_path))

    if not valid:
        return results

    bin_path = BIN_DIR / "verify-ipa"
    if not bin_path.exists():
        # Fallback: individual Python verification
        for name, path, com_path in valid:
            try:
                results[name] = verify_ipa_python(path, com_path)
            except Exception as e:
                results[name] = {"fold_ok": False, "binding_ok": False, "error": str(e)}
        return results

    # Batch call: verify-ipa proof1 com1 proof2 com2 ...
    # One subprocess → one CUDA context init for all N proofs.
    cmd = [str(bin_path)]
    for _, path, com_path in valid:
        cmd += [path, com_path]
    try:
        r = subprocess.run(cmd, capture_output=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
        out = r.stdout.decode().strip()
        if len(valid) == 1:
            parsed = json.loads(out)           # single object
            for name, _, _ in valid:
                results[name] = parsed
        else:
            parsed = json.loads(out)           # JSON array
            for (name, _, _), item in zip(valid, parsed):
                results[name] = item
    except Exception as e:
        err = str(e)
        for name, _, _ in valid:
            results[name] = {"fold_ok": False, "binding_ok": False, "error": err}
    return results


def _setup_vit_workdir(workdir: Path, block_idx: int) -> Path:
    """
    为每个 block 创建独立的 workdir/vit-b{N}/ 子目录（避免并行时 symlink 冲突）。
    包含指向 ViT pp、权重、承诺文件的 symlinks，与 LLM pp 文件隔离。
    返回 vit workdir 路径。
    """
    vit_wd = workdir / f"vit-b{block_idx}"
    vit_wd.mkdir(exist_ok=True)

    # PP 文件：binary 期望 self_attn.q_proj.weight-pp.bin，但 ViT 的是 vit_self_attn.q_proj.weight-pp.bin
    pp_links = {
        "self_attn.q_proj.weight-pp.bin":          "vit_self_attn.q_proj.weight-pp.bin",
        "self_attn.k_proj.weight-pp.bin":          "vit_self_attn.k_proj.weight-pp.bin",
        "self_attn.v_proj.weight-pp.bin":          "vit_self_attn.v_proj.weight-pp.bin",
        "self_attn.o_proj.weight-pp.bin":          "vit_self_attn.o_proj.weight-pp.bin",
        "mlp.gate_proj.weight-pp.bin":             "vit_mlp.gate_proj.weight-pp.bin",
        "mlp.up_proj.weight-pp.bin":               "vit_mlp.up_proj.weight-pp.bin",
        "mlp.down_proj.weight-pp.bin":             "vit_mlp.down_proj.weight-pp.bin",
        "input_layernorm.weight-pp.bin":           "vit_input_layernorm.weight-pp.bin",
        "post_attention_layernorm.weight-pp.bin":  "vit_post_attention_layernorm.weight-pp.bin",
    }
    for link_name, target_name in pp_links.items():
        link   = vit_wd / link_name
        target = workdir / target_name
        if not link.exists() and target.exists():
            link.symlink_to(target.resolve())

    # 权重 + 承诺文件：vit-block-N-*.bin
    prefix = f"vit-block-{block_idx}"
    for src in workdir.glob(f"{prefix}-*.bin"):
        link = vit_wd / src.name
        if not link.exists():
            link.symlink_to(src.resolve())

    return vit_wd


# ── 单 block 完整 8 步证明 ────────────────────────────────────────────────────

def verify_vit_block(block_idx: int, workdir: Path, gpu_id: int = 0) -> dict:
    prefix = f"vit-block-{block_idx}"
    cwd    = str(ZKLLM_CWD)
    step_ms = {}
    step_rc = {}

    # 每个 block 独立子目录，并行时互不冲突
    vit_wd = _setup_vit_workdir(workdir, block_idx)
    wd = str(vit_wd)

    # 激活文件放在 vit_wd 中（有 prefix，不与 LLM 冲突）
    h_in_raw = vit_wd / f"{prefix}-h_in.bin"
    h_mid    = vit_wd / f"{prefix}-h_mid.bin"
    h_out    = vit_wd / f"{prefix}-h_out.bin"
    tmp_a    = vit_wd / f"{prefix}-tmp_a.bin"
    tmp_b    = vit_wd / f"{prefix}-tmp_b.bin"

    _make_input(h_in_raw, SEQ_LEN, VIT_HIDDEN)  # 仅在文件缺失时写入（随机数模式）
    n_real, seq_use, n_wins = _get_seq_params(h_in_raw, block_idx)

    # full-attn 块：将 h_in 零填充至 seq_use（下一个 2^k），满足 zkAttn NTT 约束
    if block_idx in FULL_ATT_BLOCKS and seq_use > n_real:
        _X = np.fromfile(str(h_in_raw), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
        h_in = vit_wd / f"{prefix}-h_in_padded.bin"
        np.vstack([_X, np.zeros((seq_use - n_real, VIT_HIDDEN), np.int32)]).tofile(str(h_in))
    else:
        h_in = h_in_raw

    # Step 1: input_layernorm (norm1)
    rms_inv_pre = vit_wd / f"{prefix}-rms_inv_pre.bin"
    _rms_inv(h_in, rms_inv_pre, seq_use, VIT_HIDDEN)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "input",
                    str(h_in), str(seq_use), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_a), str(rms_inv_pre)], cwd, gpu_id)
    step_ms["rmsnorm_pre"] = round((time.perf_counter()-t0)*1000)
    step_rc["rmsnorm_pre"] = rc
    if rc != 0:
        print(f"    [Step1 FAIL] {err[-300:]}", file=sys.stderr)
    attn_in = tmp_a

    # Step 2: self-attn linear (MHA: kv_dim=1280, num_kv_heads=16)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "linear",
                    str(attn_in), str(seq_use), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_b), str(VIT_HIDDEN)], cwd, gpu_id)
    step_ms["attn_linear"] = round((time.perf_counter()-t0)*1000)
    step_rc["attn_linear"] = rc
    if rc != 0:
        print(f"    [Step2 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 3: zkAttn
    # Window 块：cross-window batch — split Q/K/V into n_wins window files, then call
    # self-attn attn ONCE with n_wins arg; binary loops internally, eliminating
    # per-window subprocess fork / CUDA context re-init overhead.
    # Full-attn 块：single call, seq = seq_use (padded to next power-of-2).
    t0 = time.perf_counter()
    if block_idx not in FULL_ATT_BLOCKS:
        _split_qkv_for_windows(vit_wd, prefix, n_real, n_wins)
        rc, err = _run([str(BIN_DIR / "self-attn"), "attn",
                        str(attn_in), str(WIN_SEQ), str(VIT_HIDDEN),
                        wd, prefix, str(tmp_b),
                        str(VIT_HIDDEN), str(VIT_KV_HEADS),
                        "0", str(VIT_KV_HEADS),   # g_start, g_end (all heads)
                        str(n_wins)],              # cross-window batch
                       cwd, gpu_id)
        if rc != 0:
            print(f"    [Step3 win-batch FAIL] {err[-300:]}", file=sys.stderr)
    else:
        rc, err = _run([str(BIN_DIR / "self-attn"), "attn",
                        str(attn_in), str(seq_use), str(VIT_HIDDEN),
                        wd, prefix, str(tmp_b),
                        str(VIT_HIDDEN), str(VIT_KV_HEADS)], cwd, gpu_id)
        if rc != 0:
            print(f"    [Step3 FAIL] {err[-300:]}", file=sys.stderr)
    step_ms["zkAttn"] = round((time.perf_counter()-t0)*1000)
    step_rc["zkAttn"] = rc

    # Step 4: o_proj
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "o_proj",
                    str(attn_in), str(seq_use), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_b)], cwd, gpu_id)
    step_ms["o_proj"] = round((time.perf_counter()-t0)*1000)
    step_rc["o_proj"] = rc
    if rc != 0:
        print(f"    [Step4 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 5: skip connection 1
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "skip-connection"),
                    str(h_in), str(tmp_b), str(h_mid)], cwd, gpu_id)
    step_ms["skip1"] = round((time.perf_counter()-t0)*1000)
    step_rc["skip1"] = rc

    # Step 6: post_attention_layernorm (norm2)
    rms_inv_post = vit_wd / f"{prefix}-rms_inv_post.bin"
    _rms_inv(h_mid, rms_inv_post, seq_use, VIT_HIDDEN)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "post_attention",
                    str(h_mid), str(seq_use), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_a), str(rms_inv_post)], cwd, gpu_id)
    step_ms["rmsnorm_post"] = round((time.perf_counter()-t0)*1000)
    step_rc["rmsnorm_post"] = rc
    if rc != 0:
        print(f"    [Step6 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 7: FFN (SwiGLU，复用 swiglu-table.bin from CWD)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "ffn"),
                    str(tmp_a), str(seq_use), str(VIT_HIDDEN), str(VIT_INTER),
                    wd, prefix, str(tmp_b)], cwd, gpu_id)
    step_ms["ffn"] = round((time.perf_counter()-t0)*1000)
    step_rc["ffn"] = rc
    if rc != 0:
        print(f"    [Step7 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 8: skip connection 2
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "skip-connection"),
                    str(h_mid), str(tmp_b), str(h_out)], cwd, gpu_id)
    step_ms["skip2"] = round((time.perf_counter()-t0)*1000)
    step_rc["skip2"] = rc

    prover_ok = all(v == 0 for v in step_rc.values())

    # IPA 验证（C++ GPU binary；proof 文件在 vit_wd，commitment 文件在 workdir）
    ipa = {}
    if prover_ok:
        ipa = _verify_proofs([
            ("input_layernorm",          str(vit_wd/f"{prefix}-input_layernorm-ipa-proof.bin")),
            ("self_attn.k_proj",         str(vit_wd/f"{prefix}-self_attn.k_proj-ipa-proof.bin")),
            ("self_attn.q_proj",         str(vit_wd/f"{prefix}-self_attn.q_proj-ipa-proof.bin")),
            ("self_attn.v_proj",         str(vit_wd/f"{prefix}-self_attn.v_proj-ipa-proof.bin")),
            ("self_attn.o_proj",         str(vit_wd/f"{prefix}-self_attn.o_proj-ipa-proof.bin")),
            ("post_attention_layernorm", str(vit_wd/f"{prefix}-post_attention_layernorm-ipa-proof.bin")),
            ("mlp.gate_proj",            str(vit_wd/f"{prefix}-mlp.gate_proj-ipa-proof.bin")),
            ("mlp.up_proj",              str(vit_wd/f"{prefix}-mlp.up_proj-ipa-proof.bin")),
            ("mlp.down_proj",            str(vit_wd/f"{prefix}-mlp.down_proj-ipa-proof.bin")),
        ], workdir=workdir, gpu_id=gpu_id)

    # 清理激活和临时文件（不删 symlinks/proof files）
    for p in [h_in_raw, h_mid, h_out, tmp_a, tmp_b, rms_inv_pre, rms_inv_post]:
        p.unlink(missing_ok=True)
    if h_in != h_in_raw:          # full-attn 时创建的 h_in_padded
        h_in.unlink(missing_ok=True)
    for suf in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (vit_wd / f"{prefix}-{suf}").unlink(missing_ok=True)
    # 清理 window 临时 Q/K/V 文件（n_wins 个，动态，不再硬编码 16）
    if block_idx not in FULL_ATT_BLOCKS:
        for w in range(n_wins):
            for suf in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                (vit_wd / f"{prefix}-win{w}-{suf}").unlink(missing_ok=True)

    fold_pass = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("fold_ok"))
    bind_pass = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("binding_ok"))
    bind_tot  = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("binding_ok") is not None)
    ipa_total = len([v for v in ipa.values() if v is not None])
    all_pass  = prover_ok and fold_pass == ipa_total and (bind_tot == 0 or bind_pass == bind_tot)

    status = "✓ PASS" if all_pass else "✗ FAIL"
    failed = [k for k, v in step_rc.items() if v != 0]
    total_ms = sum(step_ms.values())
    print(f"  Block {block_idx}: {status}  "
          f"({total_ms}ms  fold={fold_pass}/{ipa_total}  binding={bind_pass}/{bind_tot})"
          + (f"  failed={failed}" if failed else ""))

    return {
        "block": block_idx,
        "all_pass": all_pass,
        "prover_ok": prover_ok,
        "step_rc": step_rc,
        "step_ms": step_ms,
        "ipa": ipa,
    }


# ── 主入口 ────────────────────────────────────────────────────────────────────

def _run_blocks_on_gpu(block_list, workdir, gpu_id):
    """在指定 GPU 上串行跑一批 blocks（供 ProcessPoolExecutor 调用）。"""
    return [verify_vit_block(bi, workdir, gpu_id=gpu_id) for bi in block_list]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks",   nargs="+", type=int, default=[0, 7])
    parser.add_argument("--workdir",  default="zkllm-workdir/jina-v4")
    parser.add_argument("--out",      default="notes/experiment_results/verify_vit.json")
    parser.add_argument("--parallel", action="store_true",
                        help="两卡并行：偶数块→GPU0，奇数块→GPU1")
    args = parser.parse_args()

    workdir = (ROOT / args.workdir).resolve()
    if not workdir.exists():
        print(f"ERROR: workdir not found: {workdir}", file=sys.stderr)
        sys.exit(1)

    # C++ binary preferred; py_ecc only needed as fallback
    if not (BIN_DIR / "verify-ipa").exists():
        try:
            from py_ecc.optimized_bls12_381 import add  # noqa: F401
        except ImportError:
            print("ERROR: verify-ipa binary not found and py_ecc not installed.", file=sys.stderr)
            sys.exit(1)

    mode = "2-GPU parallel" if args.parallel else "single GPU"
    print(f"zkLLM ViT 验证 — blocks {args.blocks}  [{mode}]")
    print(f"  seq={SEQ_LEN}, hidden={VIT_HIDDEN}, inter={VIT_INTER}")
    print(f"  MHA: num_kv_heads={VIT_KV_HEADS}, head_dim={VIT_HIDDEN//VIT_KV_HEADS}")
    print()

    t0 = time.perf_counter()
    results = []

    if args.parallel and len(args.blocks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        blocks_0 = args.blocks[::2]   # 偶数位置 → GPU 0
        blocks_1 = args.blocks[1::2]  # 奇数位置 → GPU 1
        print(f"  GPU0: blocks {blocks_0}")
        print(f"  GPU1: blocks {blocks_1}")
        print()
        with ProcessPoolExecutor(max_workers=2) as ex:
            f0 = ex.submit(_run_blocks_on_gpu, blocks_0, workdir, 0)
            f1 = ex.submit(_run_blocks_on_gpu, blocks_1, workdir, 1)
            res_0 = f0.result()
            res_1 = f1.result()
        # 按原始 blocks 顺序重组结果
        idx_0 = {r["block"]: r for r in res_0}
        idx_1 = {r["block"]: r for r in res_1}
        for bi in args.blocks:
            r = idx_0.get(bi) or idx_1.get(bi)
            print(f"[Block {bi}] ", end="")
            if r["all_pass"]:
                print(f"✓ PASS  ({sum(r['step_ms'].values())}ms)")
            else:
                print(f"✗ FAIL  failed={[k for k,v in r['step_rc'].items() if v!=0]}")
            results.append(r)
    else:
        for bi in args.blocks:
            print(f"[Block {bi}] running 8-step proof ...")
            results.append(verify_vit_block(bi, workdir, gpu_id=0))

    elapsed = round((time.perf_counter()-t0)*1000)
    n_pass  = sum(1 for r in results if r["all_pass"])

    print()
    print("━"*56)
    print(f"Blocks passed:  {n_pass}/{len(results)}")
    print(f"Total time:     {elapsed}ms")
    print("━"*56)

    summary = {"blocks": args.blocks, "blocks_pass": n_pass,
               "elapsed_ms": elapsed, "results": results}
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results: {out}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()

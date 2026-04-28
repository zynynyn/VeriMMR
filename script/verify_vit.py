"""
zkLLM ViT 32 blocks 完整验证脚本

对 jina-v4 ViT 每个 block 按完整 Transformer 顺序运行 8 步证明，
并对所有有权重承诺的步骤做 IPA fold + 承诺绑定验证。

ViT 结构（Qwen2_5_VLVisionBlock）：
  norm1 (RMSNorm)、fused QKV (Linear(1280,3840))、o_proj、norm2、MLP(SwiGLU)
  - num_heads=16, head_dim=80, MHA (kv_dim=1280, num_kv_heads=16)
  - 所有块使用 seq_len=1024（统一序列长度）
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

_P_FP    = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP    = pow(2, 384, _P_FP)
_R_FP_INV = pow(_R_FP, -1, _P_FP)
_P_FR    = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


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
        try:
            results[name] = verify_ipa_cpp(path, com_path, gpu_id=gpu_id)
        except Exception as e:
            results[name] = {"fold_ok": False, "binding_ok": False, "error": str(e)}
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
    h_in  = vit_wd / f"{prefix}-h_in.bin"
    h_mid = vit_wd / f"{prefix}-h_mid.bin"
    h_out = vit_wd / f"{prefix}-h_out.bin"
    tmp_a = vit_wd / f"{prefix}-tmp_a.bin"
    tmp_b = vit_wd / f"{prefix}-tmp_b.bin"

    _make_input(h_in, SEQ_LEN, VIT_HIDDEN)

    # Step 1: input_layernorm (norm1)
    rms_inv_pre = vit_wd / f"{prefix}-rms_inv_pre.bin"
    _rms_inv(h_in, rms_inv_pre, SEQ_LEN, VIT_HIDDEN)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "input",
                    str(h_in), str(SEQ_LEN), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_a), str(rms_inv_pre)], cwd, gpu_id)
    step_ms["rmsnorm_pre"] = round((time.perf_counter()-t0)*1000)
    step_rc["rmsnorm_pre"] = rc
    if rc != 0:
        print(f"    [Step1 FAIL] {err[-300:]}", file=sys.stderr)
    attn_in = tmp_a

    # Step 2: self-attn linear (MHA: kv_dim=1280, num_kv_heads=16)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "linear",
                    str(attn_in), str(SEQ_LEN), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_b), str(VIT_HIDDEN)], cwd, gpu_id)
    step_ms["attn_linear"] = round((time.perf_counter()-t0)*1000)
    step_rc["attn_linear"] = rc
    if rc != 0:
        print(f"    [Step2 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 3: zkAttn (MHA: num_kv_heads=16, head_dim=80)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "attn",
                    str(attn_in), str(SEQ_LEN), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_b),
                    str(VIT_HIDDEN), str(VIT_KV_HEADS)], cwd, gpu_id)
    step_ms["zkAttn"] = round((time.perf_counter()-t0)*1000)
    step_rc["zkAttn"] = rc
    if rc != 0:
        print(f"    [Step3 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 4: o_proj
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "o_proj",
                    str(attn_in), str(SEQ_LEN), str(VIT_HIDDEN),
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
    _rms_inv(h_mid, rms_inv_post, SEQ_LEN, VIT_HIDDEN)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "post_attention",
                    str(h_mid), str(SEQ_LEN), str(VIT_HIDDEN),
                    wd, prefix, str(tmp_a), str(rms_inv_post)], cwd, gpu_id)
    step_ms["rmsnorm_post"] = round((time.perf_counter()-t0)*1000)
    step_rc["rmsnorm_post"] = rc
    if rc != 0:
        print(f"    [Step6 FAIL] {err[-300:]}", file=sys.stderr)

    # Step 7: FFN (SwiGLU，复用 swiglu-table.bin from CWD)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "ffn"),
                    str(tmp_a), str(SEQ_LEN), str(VIT_HIDDEN), str(VIT_INTER),
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
    for p in [h_in, h_mid, h_out, tmp_a, tmp_b, rms_inv_pre, rms_inv_post]:
        p.unlink(missing_ok=True)
    for suf in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (vit_wd / f"{prefix}-{suf}").unlink(missing_ok=True)

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

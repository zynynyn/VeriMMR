"""
zkLLM 完整层验证脚本 — 每层 8 步全流程

对 jina-v4 最后 K 层，按完整 Transformer Block 顺序运行所有 zkLLM prover
并对有权重承诺的步骤做 IPA fold + 承诺绑定验证：

  Step 1: input_layernorm      (RMSNorm + 权重 IPA)
  Step 2: self-attn linear     (q/k/v 投影 IPA)
  Step 3: self-attn attn       (GQA zkSoftmax sumcheck)
  Step 4: self-attn o_proj     (输出投影 IPA)
  Step 5: skip connection 1    (h + attn_out)
  Step 6: post_attention_layernorm (RMSNorm + 权重 IPA)
  Step 7: FFN                  (gate/up/SwiGLU/down IPA)
  Step 8: skip connection 2    (h' + ffn_out)

IPA 验证：
  - fold 一致性：C_final == g_final × w_final
  - 承诺绑定：C_init == com(u_out)（有 commitment 文件时执行）

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/verify_layers.py [--layers 30 31 32] [--workdir zkllm-workdir/jina-v4]
  # Fiat-Shamir 模式（verifier 端重现层选择）：
  python script/verify_layers.py --challenge-seed <hex64> [--k 6]
"""

import argparse
import hashlib
import json
import os
import random as _random
import struct
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent.resolve()
BIN_DIR   = ROOT / "src" / "zkllm"
ZKLLM_CWD = ROOT / "src" / "zkllm"

EMBED_DIM    = 2048
HIDDEN_DIM   = 11008
KV_DIM       = 256
NUM_KV_HEADS = 2
SEQ_LEN      = 1024

_P_FP = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP = pow(2, 384, _P_FP)
_R_FP_INV = pow(_R_FP, -1, _P_FP)
_P_FR = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
RMS_EPS = 1e-6


def _fiat_shamir_layers(challenge_hex: str, K: int = 6, total: int = 36) -> list:
    """从 challenge_hex（SHA256 hex）派生 K 个层索引，与 interactive_demo.py 逻辑一致。"""
    seed = bytes.fromhex(challenge_hex)
    rng  = _random.Random(seed)
    return sorted(rng.sample(range(total), K))


# ── IPA proof 解码 ──────────────────────────────────────────────────────────

def _read_fp_mont(data: bytes, offset: int) -> int:
    raw = int.from_bytes(data[offset:offset + 48], "little")
    return (raw * _R_FP_INV) % _P_FP


def _read_g1_jacobian(data: bytes, offset: int):
    """blstrs Jacobian (Y²=X³+bZ⁶) → py_ecc projective Z=1."""
    from py_ecc.optimized_bls12_381 import FQ, Z1
    X = _read_fp_mont(data, offset)
    Y = _read_fp_mont(data, offset + 48)
    Z = _read_fp_mont(data, offset + 96)
    if Z == 0:
        return Z1
    z_inv  = pow(Z, _P_FP - 2, _P_FP)
    z_inv2 = (z_inv * z_inv) % _P_FP
    z_inv3 = (z_inv2 * z_inv) % _P_FP
    return (FQ((X * z_inv2) % _P_FP), FQ((Y * z_inv3) % _P_FP), FQ(1))


def _read_fr(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 32], "little")


def _load_commitment(com_path: str) -> list:
    with open(com_path, "rb") as f:
        data = f.read()
    return [_read_g1_jacobian(data, i * 144) for i in range(len(data) // 144)]


def _eval_g1_ml(points: list, u_vec: list) -> object:
    from py_ecc.optimized_bls12_381 import add, multiply, Z1
    pts = list(points)
    for u in u_vec:
        omu = (1 - u) % _P_FR
        new = []
        for i in range(0, len(pts), 2):
            a = pts[i]
            b = pts[i + 1] if i + 1 < len(pts) else Z1
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
    with open(proof_path, "rb") as f:
        data = f.read()
    if data[:4] not in (b"IPA\x00", b"\x00API"):
        raise ValueError(f"Bad magic: {data[:4]!r}")
    k       = struct.unpack_from("<I", data, 4)[0]
    com_log = struct.unpack_from("<I", data, 8)[0]
    off     = 12
    C_init = _read_g1_jacobian(data, off); off += 144
    u_out  = [_read_fr(data, off + i * 32) for i in range(com_log)]; off += com_log * 32
    u_in   = [_read_fr(data, off + i * 32) for i in range(k)];       off += k * 32
    rounds = []
    for _ in range(k):
        L0 = _read_g1_jacobian(data, off); off += 144
        L1 = _read_g1_jacobian(data, off); off += 144
        rounds.append((L0, L1))
    g_final = _read_g1_jacobian(data, off); off += 144
    w_final = _read_fr(data, off)
    C = C_init
    for u, (L0, L1) in zip(u_in, rounds):
        omu = (1 - u) % _P_FR
        C = add(add(multiply(L0, (omu * omu) % _P_FR),
                    multiply(C,  (u * omu)   % _P_FR)),
                multiply(L1, (u * u) % _P_FR))
    fold_ok = eq(C, multiply(g_final, w_final))
    binding_ok = None
    if os.path.exists(com_path):
        try:
            binding_ok = eq(C_init, _eval_g1_ml(_load_commitment(com_path), u_out))
        except Exception as e:
            binding_ok = False
            print(f"  [BINDING ERROR] {e}", file=sys.stderr)
    return {"fold_ok": fold_ok, "binding_ok": binding_ok}


# ── 辅助：激活文件生成 & prover 运行 ────────────────────────────────────────

def _make_input(path: Path, rows: int = SEQ_LEN, cols: int = EMBED_DIM):
    """生成随机 int32 激活文件（已存在则复用）。"""
    if not path.exists():
        rng = np.random.default_rng(abs(hash(str(path))) % (2 ** 31))
        (rng.standard_normal((rows, cols)) * (1 << 16)).astype(np.int32).tofile(str(path))
    return path


def _rms_inv(input_path: Path, out_path: Path):
    """从 int32 激活文件计算 rms_inv，写到 out_path（按层命名，避免并行冲突）。"""
    X = np.fromfile(str(input_path), dtype=np.int32).reshape(SEQ_LEN, EMBED_DIM) / (1 << 16)
    rms_inv = 1.0 / np.sqrt((X ** 2).mean(axis=1) + RMS_EPS)   # (SEQ_LEN,)
    (rms_inv * (1 << 16)).round().astype(np.int32).tofile(str(out_path))


def _run(cmd: list, cwd: str, gpu_id: int = 0) -> tuple:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    r = subprocess.run(cmd, capture_output=True, cwd=cwd, env=env)
    return r.returncode, r.stderr.decode(errors="replace")




def _verify_proofs(proof_paths: list, gpu_id: int = 0) -> dict:
    """对一组 proof 文件执行 IPA 验证（C++ GPU binary），返回 {name: {fold_ok, binding_ok}}。"""
    results = {}
    for name, path in proof_paths:
        if not Path(path).exists():
            results[name] = None
            continue
        com_path = path.replace("-ipa-proof.bin", ".weight-commitment.bin")
        if not os.path.exists(com_path):
            results[name] = {"fold_ok": None, "binding_ok": None,
                             "error": "commitment not found"}
            continue
        try:
            results[name] = verify_ipa_cpp(path, com_path, gpu_id=gpu_id)
        except Exception as e:
            results[name] = {"fold_ok": False, "binding_ok": False, "error": str(e)}
    return results


# ── 单层完整 8 步证明 ────────────────────────────────────────────────────────

def verify_layer(layer_idx: int, workdir: Path, gpu_id: int = 0) -> dict:
    prefix  = f"layer-{layer_idx}"
    cwd     = str(ZKLLM_CWD)
    wd      = str(workdir)
    step_ms = {}
    step_rc = {}
    ipa     = {}

    # 工作文件（全部临时，结束后清理）
    h_in   = workdir / f"{prefix}-full-h_in.bin"    # 层输入
    h_mid  = workdir / f"{prefix}-full-h_mid.bin"   # attn skip 输出
    h_out  = workdir / f"{prefix}-full-h_out.bin"   # ffn skip 输出
    tmp_a  = workdir / f"{prefix}-full-tmp_a.bin"
    tmp_b  = workdir / f"{prefix}-full-tmp_b.bin"

    _make_input(h_in)   # 层输入：随机激活

    # ── Step 1: input_layernorm ─────────────────────────────────────────────
    rms_inv_pre = workdir / f"{prefix}-rms_inv_pre.bin"
    _rms_inv(h_in, rms_inv_pre)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "input",
                    str(h_in), str(SEQ_LEN), str(EMBED_DIM),
                    wd, prefix, str(tmp_a), str(rms_inv_pre)], cwd, gpu_id)
    step_ms["rmsnorm_pre"] = round((time.perf_counter() - t0) * 1000)
    step_rc["rmsnorm_pre"] = rc
    if rc != 0:
        print(f"    [Step1 FAIL] {err[-200:]}", file=sys.stderr)
    attn_in = tmp_a  # rmsnorm output → attn input

    # ── Step 2: self-attn linear (q/k/v IPA) ───────────────────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "linear",
                    str(attn_in), str(SEQ_LEN), str(EMBED_DIM),
                    wd, prefix, str(tmp_b), str(KV_DIM)], cwd, gpu_id)
    step_ms["attn_linear"] = round((time.perf_counter() - t0) * 1000)
    step_rc["attn_linear"] = rc
    if rc != 0:
        print(f"    [Step2 FAIL] {err[-200:]}", file=sys.stderr)

    # ── Step 3: zkAttn softmax ──────────────────────────────────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "attn",
                    str(attn_in), str(SEQ_LEN), str(EMBED_DIM),
                    wd, prefix, str(tmp_b), str(KV_DIM), str(NUM_KV_HEADS)], cwd, gpu_id)
    step_ms["zkAttn"] = round((time.perf_counter() - t0) * 1000)
    step_rc["zkAttn"] = rc
    if rc != 0:
        print(f"    [Step3 FAIL] {err[-200:]}", file=sys.stderr)

    # ── Step 4: o_proj IPA ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "self-attn"), "o_proj",
                    str(attn_in), str(SEQ_LEN), str(EMBED_DIM),
                    wd, prefix, str(tmp_b)], cwd, gpu_id)
    step_ms["o_proj"] = round((time.perf_counter() - t0) * 1000)
    step_rc["o_proj"] = rc
    if rc != 0:
        print(f"    [Step4 FAIL] {err[-200:]}", file=sys.stderr)

    # ── Step 5: skip connection 1 (h + attn_out → h_mid) ───────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "skip-connection"),
                    str(h_in), str(tmp_b), str(h_mid)], cwd, gpu_id)
    step_ms["skip1"] = round((time.perf_counter() - t0) * 1000)
    step_rc["skip1"] = rc

    # ── Step 6: post_attention_layernorm ────────────────────────────────────
    rms_inv_post = workdir / f"{prefix}-rms_inv_post.bin"
    _rms_inv(h_mid, rms_inv_post)
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "rmsnorm"), "post_attention",
                    str(h_mid), str(SEQ_LEN), str(EMBED_DIM),
                    wd, prefix, str(tmp_a), str(rms_inv_post)], cwd, gpu_id)
    step_ms["rmsnorm_post"] = round((time.perf_counter() - t0) * 1000)
    step_rc["rmsnorm_post"] = rc
    if rc != 0:
        print(f"    [Step6 FAIL] {err[-200:]}", file=sys.stderr)
    ffn_in = tmp_a  # post-norm output → ffn input

    # ── Step 7: FFN (gate/up/SwiGLU/down IPA) ──────────────────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "ffn"),
                    str(ffn_in), str(SEQ_LEN), str(EMBED_DIM), str(HIDDEN_DIM),
                    wd, prefix, str(tmp_b)], cwd, gpu_id)
    step_ms["ffn"] = round((time.perf_counter() - t0) * 1000)
    step_rc["ffn"] = rc
    if rc != 0:
        print(f"    [Step7 FAIL FULL STDERR]:\n{err}", file=sys.stderr)

    # ── Step 8: skip connection 2 (h_mid + ffn_out → h_out) ────────────────
    t0 = time.perf_counter()
    rc, err = _run([str(BIN_DIR / "skip-connection"),
                    str(h_mid), str(tmp_b), str(h_out)], cwd, gpu_id)
    step_ms["skip2"] = round((time.perf_counter() - t0) * 1000)
    step_rc["skip2"] = rc

    prover_ok = all(v == 0 for v in step_rc.values())

    # ── IPA 验证（C++ GPU binary，有权重承诺的步骤）──────────────────────
    if prover_ok:
        ipa = _verify_proofs([
            ("input_layernorm",          str(workdir / f"{prefix}-input_layernorm-ipa-proof.bin")),
            ("self_attn.k_proj",         str(workdir / f"{prefix}-self_attn.k_proj-ipa-proof.bin")),
            ("self_attn.q_proj",         str(workdir / f"{prefix}-self_attn.q_proj-ipa-proof.bin")),
            ("self_attn.v_proj",         str(workdir / f"{prefix}-self_attn.v_proj-ipa-proof.bin")),
            ("self_attn.o_proj",         str(workdir / f"{prefix}-self_attn.o_proj-ipa-proof.bin")),
            ("post_attention_layernorm", str(workdir / f"{prefix}-post_attention_layernorm-ipa-proof.bin")),
            ("mlp.gate_proj",            str(workdir / f"{prefix}-mlp.gate_proj-ipa-proof.bin")),
            ("mlp.up_proj",              str(workdir / f"{prefix}-mlp.up_proj-ipa-proof.bin")),
            ("mlp.down_proj",            str(workdir / f"{prefix}-mlp.down_proj-ipa-proof.bin")),
        ], gpu_id=gpu_id)

    # ── 清理临时文件 ────────────────────────────────────────────────────────
    for p in [h_in, h_mid, h_out, tmp_a, tmp_b, rms_inv_pre, rms_inv_post]:
        p.unlink(missing_ok=True)
    # temp_Q/K/V 现在写到 workdir，按前缀命名
    for suffix in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (workdir / f"{prefix}-{suffix}").unlink(missing_ok=True)

    fold_pass = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("fold_ok"))
    bind_pass = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("binding_ok"))
    bind_tot  = sum(1 for v in ipa.values() if isinstance(v, dict) and v.get("binding_ok") is not None)
    ipa_total = len([v for v in ipa.values() if v is not None])
    all_pass  = prover_ok and fold_pass == ipa_total and (bind_tot == 0 or bind_pass == bind_tot)

    failed_steps = [k for k, v in step_rc.items() if v != 0]
    status = "✓ PASS" if all_pass else "✗ FAIL"
    total_ms = sum(step_ms.values())
    print(f"  Layer {layer_idx}: {status}  "
          f"({total_ms}ms  fold={fold_pass}/{ipa_total}  binding={bind_pass}/{bind_tot})"
          + (f"  failed={failed_steps}" if failed_steps else ""))

    return {
        "layer": layer_idx,
        "all_pass": all_pass,
        "prover_ok": prover_ok,
        "step_rc": step_rc,
        "step_ms": step_ms,
        "ipa": ipa,
        "fold_pass": fold_pass,
        "bind_pass": bind_pass,
    }


# ── 主入口 ──────────────────────────────────────────────────────────────────

def _run_batch(layer_list: list, workdir_str: str, gpu_id: int) -> list:
    """在指定 GPU 上串行跑一批层（用于 ProcessPoolExecutor 子进程）。"""
    workdir = Path(workdir_str)
    results = []
    for li in layer_list:
        print(f"[GPU{gpu_id}] Layer {li} running ...", flush=True)
        results.append(verify_layer(li, workdir, gpu_id=gpu_id))
    return results


def main():
    print("开始测试")
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers",         nargs="+", type=int, default=None)
    parser.add_argument("--challenge-seed", default=None,
                        help="Fiat-Shamir challenge（SHA256 hex，64字符）；与 --layers 互斥")
    parser.add_argument("--k",              type=int, default=6,
                        help="Fiat-Shamir 模式下选取的层数（默认 6）")
    parser.add_argument("--workdir",  default="zkllm-workdir/jina-v4")
    parser.add_argument("--out",      default="notes/experiment_results/verify_layers.json")
    parser.add_argument("--parallel", action="store_true",
                        help="2-GPU 并行：偶数层 GPU0，奇数层 GPU1")
    args = parser.parse_args()

    # 层选择：--challenge-seed 优先，否则 --layers，否则默认 [30]
    if args.challenge_seed is not None:
        if len(args.challenge_seed) != 64:
            print("ERROR: --challenge-seed 须为 64 字符 hex（SHA256 输出）", file=sys.stderr)
            sys.exit(1)
        layer_list = _fiat_shamir_layers(args.challenge_seed, K=args.k)
        print(f"Fiat-Shamir 层选择: {layer_list}  (K={args.k}, seed={args.challenge_seed[:16]}…)")
    elif args.layers is not None:
        layer_list = args.layers
    else:
        layer_list = [30]

    workdir = (ROOT / args.workdir).resolve()
    if not workdir.exists():
        print(f"ERROR: workdir not found: {workdir}", file=sys.stderr)
        sys.exit(1)

    if not (BIN_DIR / "verify-ipa").exists():
        try:
            from py_ecc.optimized_bls12_381 import add, multiply, eq, Z1  # noqa: F401
        except ImportError:
            print("ERROR: verify-ipa binary not found and py_ecc not installed.", file=sys.stderr)
            sys.exit(1)

    print(f"zkLLM 完整层验证（8步）— layers {layer_list}")
    print(f"  步骤: input_layernorm → q/k/v linear → zkAttn → o_proj → skip1")
    print(f"        → post_attn_layernorm → FFN → skip2")
    if args.parallel:
        print("  模式: 2-GPU 并行（偶数层→GPU0，奇数层→GPU1）")
    print()

    t0 = time.perf_counter()
    results = []

    if args.parallel and len(layer_list) > 1:
        layers_gpu0 = [li for li in layer_list if li % 2 == 0]
        layers_gpu1 = [li for li in layer_list if li % 2 == 1]
        with ProcessPoolExecutor(max_workers=2) as ex:
            futures = {}
            if layers_gpu0:
                futures[ex.submit(_run_batch, layers_gpu0, str(workdir), 0)] = layers_gpu0
            if layers_gpu1:
                futures[ex.submit(_run_batch, layers_gpu1, str(workdir), 1)] = layers_gpu1
            batch_results = {}
            for fut in as_completed(futures):
                batch_results[id(futures[fut])] = (futures[fut], fut.result())
        # 按原始层顺序合并结果
        result_map = {}
        for _, (bl, res_list) in batch_results.items():
            for li, r in zip(bl, res_list):
                result_map[li] = r
        results = [result_map[li] for li in layer_list]
    else:
        for li in layer_list:
            print(f"[Layer {li}] running 8-step proof pipeline ...")
            results.append(verify_layer(li, workdir))

    elapsed = round((time.perf_counter() - t0) * 1000)
    n_pass  = sum(1 for r in results if r["all_pass"])

    print()
    print("━" * 56)
    print(f"Layers passed:  {n_pass}/{len(results)}")
    print(f"Total time:     {elapsed}ms")
    print("━" * 56)

    summary = {
        "layers": layer_list,
        "challenge_seed": args.challenge_seed,
        "layers_pass": n_pass,
        "elapsed_ms": elapsed,
        "results": results,
    }
    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results: {out}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()

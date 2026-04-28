"""
PatchMerger 证明 smoke test

jina-v4 PatchMerger 结构：
  input (n_patches, 1280)
  → RMSNorm(1280) → reshape(n/4, 5120) → Linear(5120,5120)+GELU → Linear(5120,2048)
  → output (n_patches/4, 2048)

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/prove_patchmerger.py [--n-patches 256] [--workdir ...]
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

VIT_DIM    = 1280
MERGED_DIM = 5120
OUT_DIM    = 2048
SCALE      = 1 << 16
RMS_EPS    = 1e-6
PREFIX     = "patchmerger"

_P_FP    = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP    = pow(2, 384, _P_FP)
_R_FP_INV = pow(_R_FP, -1, _P_FP)
_P_FR    = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


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

def verify_ipa(proof_path: str) -> dict:
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
    com_path = proof_path.replace("-ipa-proof.bin", ".weight-commitment.bin")
    if os.path.exists(com_path):
        try:
            binding_ok = eq(C_init, _eval_g1_ml(_load_commitment(com_path), u_out))
        except Exception:
            binding_ok = False
    return {"fold_ok": fold_ok, "binding_ok": binding_ok}


def prove_patchmerger(n_patches=256, workdir=None) -> dict:
    if workdir is None:
        workdir = ROOT / "zkllm-workdir" / "jina-v4"
    workdir = Path(workdir)

    # 生成随机输入 (smoke test)
    rng = np.random.default_rng(42)
    X_int = (rng.standard_normal((n_patches, VIT_DIM)) * SCALE).astype(np.int32)

    input_path  = workdir / f"{PREFIX}-input.bin"
    rms_inv_path = workdir / f"{PREFIX}-rms_inv.bin"
    output_path = workdir / f"{PREFIX}-output.bin"

    X_int.tofile(str(input_path))

    # 计算 rms_inv
    X = X_int / SCALE
    rms_inv = 1.0 / np.sqrt((X**2).mean(axis=1) + RMS_EPS)
    (rms_inv * SCALE).round().astype(np.int32).tofile(str(rms_inv_path))

    # 调用 patch-merger binary
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(BIN_DIR / "patch-merger"),
         str(input_path), str(n_patches), str(VIT_DIM),
         str(MERGED_DIM), str(OUT_DIM),
         str(workdir), PREFIX, str(output_path),
         str(rms_inv_path)],
        capture_output=True, cwd=str(ZKLLM_CWD), env=env
    )
    elapsed_ms = round((time.perf_counter()-t0)*1000)
    prover_ok = (r.returncode == 0)
    if not prover_ok:
        print(f"  [PROVER FAIL] {r.stderr.decode(errors='replace')[-400:]}", file=sys.stderr)

    # IPA 验证
    ipa = {}
    proof_names = ["patchmerger_layernorm", "patchmerger_fc1", "patchmerger_fc2"]
    if prover_ok:
        for name in proof_names:
            proof_path = str(workdir / f"{PREFIX}-{name}-ipa-proof.bin")
            if Path(proof_path).exists():
                try:
                    ipa[name] = verify_ipa(proof_path)
                except Exception as e:
                    ipa[name] = {"fold_ok": False, "error": str(e)}

    # 清理
    for p in [input_path, rms_inv_path, output_path]:
        p.unlink(missing_ok=True)

    fold_pass  = sum(1 for v in ipa.values() if v.get("fold_ok"))
    bind_pass  = sum(1 for v in ipa.values() if v.get("binding_ok"))
    bind_total = sum(1 for v in ipa.values() if v.get("binding_ok") is not None)
    ipa_total  = len(ipa)
    all_ok = prover_ok and fold_pass == ipa_total

    status = "✓ PASS" if all_ok else "✗ FAIL"
    print(f"PatchMerger proof: {status}  ({elapsed_ms}ms  "
          f"fold={fold_pass}/{ipa_total}  binding={bind_pass}/{bind_total})")
    return {
        "all_ok":     all_ok,
        "prover_ok":  prover_ok,
        "elapsed_ms": elapsed_ms,
        "n_patches":  n_patches,
        "ipa":        ipa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patches", type=int, default=256)
    parser.add_argument("--workdir",   default="zkllm-workdir/jina-v4")
    parser.add_argument("--out",       default="notes/experiment_results/prove_patchmerger.json")
    args = parser.parse_args()

    workdir = (ROOT / args.workdir).resolve()
    result = prove_patchmerger(args.n_patches, workdir)

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results: {out}")
    sys.exit(0 if result["all_ok"] else 1)


if __name__ == "__main__":
    main()

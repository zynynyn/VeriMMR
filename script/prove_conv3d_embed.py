"""
Conv3d Patch Embedding 证明流水线

步骤：
  1. im2col：将图像帧展开为 patch 矩阵 (N_patch, 1176)，量化为 int32
  2. 调用 conv3d-embed binary 生成 IPA 证明
  3. Python IPA 验证（fold + commitment binding）

图像输入格式：
  - 支持 npy 文件：float32, shape=(T, 3, H, W)，像素值 [0, 1] 或 [0, 255]
  - 或直接传入已量化的 patches 文件（跳过 im2col）

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/prove_conv3d_embed.py [--frames path.npy] [--workdir ...]
  # 不传 --frames 时用随机激活做 smoke test

pipeline 调用：
  from script.prove_conv3d_embed import prove_conv3d_embed
  result = prove_conv3d_embed(frames_npy=None, workdir=workdir)
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

PATCH_DIM = 1176    # 3 × 2 × 14 × 14
OUT_DIM   = 1280    # ViT hidden size
KT, KH, KW = 2, 14, 14
SCALE     = 1 << 16
PREFIX    = "conv3d"

_P_FP    = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP    = pow(2, 384, _P_FP)
_R_FP_INV = pow(_R_FP, -1, _P_FP)
_P_FR    = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


# ── im2col ────────────────────────────────────────────────────────────────────

def im2col(frames: np.ndarray) -> np.ndarray:
    """
    非重叠 Conv3d im2col：(T, C, H, W) → (N_patch, patch_dim)

    输出 patch 顺序与 PyTorch Conv3d 前向传播一致：
      时间维（慢）→ 空间行（中）→ 空间列（快）
    """
    T, C, H, W = frames.shape
    nT = T // KT
    nH = H // KH
    nW = W // KW
    N  = nT * nH * nW

    # (nT, KT, C, nH, KH, nW, KW) → (nT, nH, nW, KT, C, KH, KW) → (N, patch_dim)
    x = frames.reshape(nT, KT, C, nH, KH, nW, KW)
    x = x.transpose(0, 3, 5, 1, 2, 4, 6)
    return x.reshape(N, -1)   # (N, 1176)


def frames_to_patches_int(frames: np.ndarray) -> np.ndarray:
    """
    图像帧 → int32 patch 矩阵。
    frames: float32 (T, C, H, W)，像素值任意范围（会自动归一化到 [0,1]）
    """
    if frames.max() > 1.01:
        frames = frames / 255.0
    # 量化：实数 × SCALE → int32
    patches = im2col(frames)
    return np.round(patches * SCALE).astype(np.int32)


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
    return (FQ((X*zi*zi) % _P_FP), FQ((Y*zi*zi*zi) % _P_FP), FQ(1))

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
            a, b = pts[i], (pts[i+1] if i+1 < len(pts) else Z1)
            new.append(add(multiply(a, omu), multiply(b, u)))
        pts = new
    return pts[0]

def verify_ipa(proof_path: str, gpu_id: int = 1) -> dict:
    """优先使用 C++ verify-ipa binary；不可用时回退 Python py_ecc。"""
    cpp_bin = BIN_DIR / "verify-ipa"
    com_path = proof_path.replace("-ipa-proof.bin", ".weight-commitment.bin")
    if cpp_bin.exists() and os.path.exists(com_path):
        import subprocess as _sp, json as _json
        env = ({**os.environ} if "CUDA_VISIBLE_DEVICES" in os.environ else {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
        r = _sp.run([str(cpp_bin), proof_path, com_path],
                    capture_output=True, env=env)
        if r.returncode in (0, 1):
            try:
                return _json.loads(r.stdout.decode().strip())
            except Exception:
                pass
    # fallback: Python py_ecc
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
                    multiply(C,  (u*omu)%_P_FR)),
                multiply(L1, (u*u)%_P_FR))
    fold_ok = eq(C, multiply(g_final, w_final))

    binding_ok = None
    com_path = proof_path.replace("-ipa-proof.bin", ".weight-commitment.bin")
    if os.path.exists(com_path):
        try:
            pts = _load_commitment(com_path)
            binding_ok = eq(C_init, _eval_g1_ml(pts, u_out))
        except Exception as e:
            binding_ok = False
    return {"fold_ok": fold_ok, "binding_ok": binding_ok}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def prove_conv3d_embed(frames_npy=None, workdir=None,
                       patch_dim=PATCH_DIM, out_dim=OUT_DIM,
                       gpu_id: int = 1,
                       patches_int32: np.ndarray = None) -> dict:
    """
    frames_npy   : str | Path | None — float32 npy (T,C,H,W)；None 则随机 smoke test
    workdir      : Path | None
    gpu_id       : int — GPU index for prover and verifier
    patches_int32: (N, patch_dim) int32 — 预计算的量化 patch 矩阵，优先于 frames_npy
    """
    if workdir is None:
        workdir = ROOT / "zkllm-workdir" / "jina-v4"
    workdir = Path(workdir)

    # 1. 生成 patches int32 文件
    patches_path = workdir / f"{PREFIX}-patches.bin"
    if patches_int32 is not None:
        # 真实激活（由外部 hook 提供）
        patches_int = patches_int32.astype(np.int32)
        n_patches   = patches_int.shape[0]
    elif frames_npy is None:
        # smoke test：随机 patch 矩阵，模拟 32 patches (4 frames 8×8px)
        rng = np.random.default_rng(42)
        n_patches = 32
        patches_int = (rng.standard_normal((n_patches, patch_dim)) * SCALE).astype(np.int32)
    else:
        frames = np.load(str(frames_npy)).astype(np.float32)
        patches_int = frames_to_patches_int(frames)
        n_patches = patches_int.shape[0]

    patches_int.tofile(str(patches_path))
    output_path = workdir / f"{PREFIX}-embed_out.bin"

    # 2. 调用 conv3d-embed binary
    env = ({**os.environ} if "CUDA_VISIBLE_DEVICES" in os.environ else {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(BIN_DIR / "conv3d-embed"),
         str(patches_path), str(n_patches), str(patch_dim), str(out_dim),
         str(workdir), PREFIX, str(output_path)],
        capture_output=True, cwd=str(ZKLLM_CWD), env=env
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    prover_ok = (r.returncode == 0)
    stderr = r.stderr.decode(errors="replace")
    if not prover_ok:
        print(f"  [PROVER FAIL] {stderr[-300:]}", file=sys.stderr)

    # 3. IPA 验证
    ipa = {}
    proof_path = str(workdir / f"{PREFIX}-conv3d_embed-ipa-proof.bin")
    if prover_ok and Path(proof_path).exists():
        try:
            ipa = verify_ipa(proof_path, gpu_id=gpu_id)
        except Exception as e:
            ipa = {"fold_ok": False, "binding_ok": False, "error": str(e)}

    # 4. 清理临时文件
    patches_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    all_ok = prover_ok and ipa.get("fold_ok", False)
    status = "✓ PASS" if all_ok else "✗ FAIL"
    print(f"Conv3d embed proof: {status}  ({elapsed_ms}ms  "
          f"fold={'✓' if ipa.get('fold_ok') else '✗'}  "
          f"binding={'✓' if ipa.get('binding_ok') else ('✗' if ipa.get('binding_ok') is False else '—')})")

    return {
        "all_ok":     all_ok,
        "prover_ok":  prover_ok,
        "elapsed_ms": elapsed_ms,
        "n_patches":  n_patches,
        "ipa":        ipa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames",  default=None, help="float32 npy (T,C,H,W)")
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    parser.add_argument("--out",     default="notes/experiment_results/prove_conv3d.json")
    args = parser.parse_args()

    workdir = (ROOT / args.workdir).resolve()
    result  = prove_conv3d_embed(args.frames, workdir)

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results: {out}")
    sys.exit(0 if result["all_ok"] else 1)


if __name__ == "__main__":
    main()

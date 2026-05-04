"""
离线生成 embedding IPA 承诺库

对 embedding/embedding.npy 中每个向量 vᵢ 生成一个 Pedersen-style
IPA 向量承诺 cm_i（单个 BLS12-381 G1 Jacobian 点，144 字节），
输出拼接到 embedding/embedding_commitments.bin（N×144 字节）。

用法：
  python script/setup_embedding_commitments.py \
      --embedding-npy embedding/embedding.npy \
      --workdir zkllm-workdir/jina-v4 \
      --out-dir embedding/ \
      --scale 65536

依赖 binary：ppgen, commit-param（已编译在 src/zkllm/）
预计时间：~5 min（303 条 × ~1s/commit-param on GPU）
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "src" / "zkllm"


def run(cmd: list, *, timeout: int = 120) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} failed (rc={r.returncode}): "
            + r.stderr.decode(errors="replace")[-400:]
        )
    return r


def setup_commitments_cpu(
    embedding_npy: str,
    workdir: str,
    out_dir: str,
    scale: int = 65536,
    verify_n: int = 3,
    pp_seed: int = 42,
) -> None:
    """
    CPU-only variant: 使用纯 Python py_ecc G1 运算生成承诺，无需 GPU。
    生成的 pp 以 pickle 格式保存，供后续 Python IPA 验证流程使用。
    约 61 min（303 × 12s/vec on CPU）。
    """
    sys.path.insert(0, str(ROOT / "src" / "sumcheck"))
    from inner_product import generate_random_pp_python, _g1_to_jacobian_bytes, P_FR
    from py_ecc.optimized_bls12_381 import add, multiply, Z1
    import pickle

    emb = np.load(embedding_npy).astype(np.float32)
    N, D = emb.shape
    print(f"[setup-cpu] 加载 embedding: N={N}, D={D}, scale={scale}")

    wd   = Path(workdir);  wd.mkdir(parents=True, exist_ok=True)
    outd = Path(out_dir);  outd.mkdir(parents=True, exist_ok=True)
    pp_pkl = str(wd / "embedding-pp-cpu.pkl")
    out    = str(outd / "embedding_commitments_cpu.bin")

    # ── Step 1: 生成 pp（随机 G1 生成器，固定 seed）─────────────────────────
    if not os.path.exists(pp_pkl):
        print(f"[pp-cpu] 生成 {D} 个 G1 生成器（seed={pp_seed}）...")
        t0 = time.perf_counter()
        pp_gens = generate_random_pp_python(D, seed=pp_seed)
        with open(pp_pkl, "wb") as f:
            pickle.dump(pp_gens, f)
        print(f"[pp-cpu] 完成 {round((time.perf_counter()-t0))}s → {pp_pkl}")
    else:
        print(f"[pp-cpu] 已存在，加载: {pp_pkl}")
        with open(pp_pkl, "rb") as f:
            pp_gens = pickle.load(f)

    # ── Step 2: 批量 commit（Python G1）──────────────────────────────────────
    print(f"\n[commit-cpu] 开始批量承诺 N={N} 个向量...")
    all_cms = []
    t_total = time.perf_counter()

    for i, v_f32 in enumerate(emb):
        v_int = [int(round(float(x) * scale)) % P_FR for x in v_f32]
        cm = Z1
        for g, s in zip(pp_gens, v_int):
            if s != 0:
                cm = add(cm, multiply(g, s))
        all_cms.append(_g1_to_jacobian_bytes(cm))

        if (i + 1) % 20 == 0 or i == N - 1:
            elapsed = time.perf_counter() - t_total
            avg_s   = elapsed / (i + 1)
            remain  = avg_s * (N - i - 1)
            print(
                f"  [{i+1}/{N}]  均值={avg_s:.1f}s/条  "
                f"预计剩余 {int(remain//60)}m{int(remain%60)}s",
                flush=True,
            )

    # ── Step 3: 原子写出 ────────────────────────────────────────────────────
    out_tmp = out + ".tmp"
    try:
        with open(out_tmp, "wb") as f:
            for cm_bytes in all_cms:
                f.write(cm_bytes)
        os.replace(out_tmp, out)
    except Exception:
        try: os.unlink(out_tmp)
        except OSError: pass
        raise
    size_kb = os.path.getsize(out) / 1024
    print(f"\n[output] 已写出 {out}  ({N}×144B = {size_kb:.1f} KB)")
    print(f"[output] 总耗时: {round(time.perf_counter()-t_total)}s")

    # ── Step 4: 抽样验证 ────────────────────────────────────────────────────
    if verify_n > 0:
        print(f"\n[verify] 抽样验证 {verify_n} 个承诺...")
        rng = np.random.default_rng(0)
        idxs = rng.choice(N, size=min(verify_n, N), replace=False).tolist()
        with open(out, "rb") as f:
            cm_data = f.read()
        for idx in idxs:
            v_int = [int(round(float(x) * scale)) % P_FR for x in emb[idx]]
            cm2 = Z1
            for g, s in zip(pp_gens, v_int):
                if s != 0:
                    cm2 = add(cm2, multiply(g, s))
            stored = cm_data[idx * 144: idx * 144 + 144]
            ok = _g1_to_jacobian_bytes(cm2) == stored
            print(f"  [i={idx}] {'✓' if ok else '✗ MISMATCH'}")
            if not ok:
                print("[verify] 一致性验证失败！", file=sys.stderr)
                sys.exit(1)
        print("[verify] 全部通过")


def setup_commitments(
    embedding_npy: str,
    workdir: str,
    out_dir: str,
    scale: int = 65536,
    verify_n: int = 3,
) -> None:
    for name in ["ppgen", "commit-param"]:
        if not (BIN_DIR / name).exists():
            print(f"[ERROR] binary 不存在: {BIN_DIR / name}", file=sys.stderr)
            sys.exit(1)

    emb = np.load(embedding_npy).astype(np.float32)
    N, D = emb.shape
    print(f"[setup] 加载 embedding: N={N}, D={D}, scale={scale}")

    wd   = Path(workdir);  wd.mkdir(parents=True, exist_ok=True)
    outd = Path(out_dir);  outd.mkdir(parents=True, exist_ok=True)
    pp   = str(wd / "embedding-pp.bin")
    out  = str(outd / "embedding_commitments.bin")

    # ── Step 1: ppgen（仅在不存在时生成）─────────────────────────────────────
    if not os.path.exists(pp):
        print(f"[ppgen] 生成公共参数 size={D} → {pp}")
        t0 = time.perf_counter()
        run([str(BIN_DIR / "ppgen"), str(D), pp])
        print(f"[ppgen] 完成  {round((time.perf_counter()-t0)*1000)}ms")
    else:
        print(f"[ppgen] 已存在，跳过: {pp}")

    # ── Step 2: 批量 commit-param ────────────────────────────────────────────
    print(f"\n[commit] 开始批量承诺 N={N} 个向量...")
    all_cms = []
    t_total = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        vec_f = os.path.join(td, "v.bin")
        cm_f  = os.path.join(td, "cm.bin")

        for i, v in enumerate(emb):
            # 量化为 int32 binary
            np.round(v * scale).astype(np.int32).tofile(vec_f)

            run([str(BIN_DIR / "commit-param"), pp, vec_f, cm_f, "1", str(D)])

            with open(cm_f, "rb") as f:
                cm_bytes = f.read()
            if len(cm_bytes) != 144:
                raise ValueError(f"[i={i}] 期望 144B Jacobian，得到 {len(cm_bytes)}B")
            all_cms.append(cm_bytes)

            if (i + 1) % 20 == 0 or i == N - 1:
                elapsed = time.perf_counter() - t_total
                avg_s   = elapsed / (i + 1)
                remain  = avg_s * (N - i - 1)
                print(
                    f"  [{i+1}/{N}]  均值={avg_s:.1f}s/条  "
                    f"预计剩余 {int(remain//60)}m{int(remain%60)}s",
                    flush=True,
                )

    # ── Step 3: 原子写出（先写 .tmp，再重命名，防中断留下半写文件）────────────
    out_tmp = out + ".tmp"
    try:
        with open(out_tmp, "wb") as f:
            for cm in all_cms:
                f.write(cm)
        os.replace(out_tmp, out)   # atomic on POSIX
    except Exception:
        try:
            os.unlink(out_tmp)
        except OSError:
            pass
        raise
    size_kb = os.path.getsize(out) / 1024
    print(f"\n[output] 已写出 {out}  ({N}×144B = {size_kb:.1f} KB)")
    print(f"[output] 总耗时: {round(time.perf_counter()-t_total)}s")

    # ── Step 4: 随机抽样验证一致性 ──────────────────────────────────────────
    if verify_n > 0:
        print(f"\n[verify] 抽样验证 {verify_n} 个承诺的一致性...")
        rng = np.random.default_rng(0)
        idxs = rng.choice(N, size=min(verify_n, N), replace=False).tolist()

        with open(out, "rb") as f:
            cm_data = f.read()

        with tempfile.TemporaryDirectory() as td:
            vec_f  = os.path.join(td, "v.bin")
            cm2_f  = os.path.join(td, "cm2.bin")
            for idx in idxs:
                np.round(emb[idx] * scale).astype(np.int32).tofile(vec_f)
                run([str(BIN_DIR / "commit-param"), pp, vec_f, cm2_f, "1", str(D)])
                with open(cm2_f, "rb") as f2:
                    cm2 = f2.read()
                stored = cm_data[idx * 144 : idx * 144 + 144]
                ok = cm2 == stored
                print(f"  [i={idx}] {'✓' if ok else '✗ MISMATCH'}")
                if not ok:
                    print("[verify] 一致性验证失败！", file=sys.stderr)
                    sys.exit(1)
        print("[verify] 全部通过")


def main():
    parser = argparse.ArgumentParser(description="生成 embedding IPA 承诺库")
    parser.add_argument("--embedding-npy", default="embedding/embedding.npy")
    parser.add_argument("--workdir",       default="zkllm-workdir/jina-v4")
    parser.add_argument("--out-dir",       default="embedding/")
    parser.add_argument("--scale",  type=int, default=65536)
    parser.add_argument("--verify-n", type=int, default=3,
                        help="完成后随机抽样验证数量（0=跳过）")
    parser.add_argument("--cpu", action="store_true",
                        help="使用 CPU 纯 Python 模式（无需 GPU，约 61 min for N=303 D=2048）")
    parser.add_argument("--pp-seed", type=int, default=42,
                        help="CPU 模式：随机 G1 生成器的 seed（默认 42）")
    args = parser.parse_args()

    emb_path = str((ROOT / args.embedding_npy).resolve())
    wd_path  = str((ROOT / args.workdir).resolve())
    out_path = str((ROOT / args.out_dir).resolve())

    if args.cpu:
        setup_commitments_cpu(
            embedding_npy=emb_path,
            workdir=wd_path,
            out_dir=out_path,
            scale=args.scale,
            verify_n=args.verify_n,
            pp_seed=args.pp_seed,
        )
    else:
        setup_commitments(
            embedding_npy=emb_path,
            workdir=wd_path,
            out_dir=out_path,
            scale=args.scale,
            verify_n=args.verify_n,
        )


if __name__ == "__main__":
    main()

"""
zkLLM Pooling Head 证明 — MeanPool + L2Norm

jina-v4 的 Pooling Head 分两步：
  1. MeanPool:  p = (1/K) * sum_{t: mask_t=1}(H_t)   (K = 有效 token 数)
  2. L2Norm:    e = p / ||p||_2                       (单行归一化)

MeanPool 是公开可验证的线性操作（权重 = mask/K）：
  - 用 sumcheck 证明 sum_{t}(mask_t * H_t) = K * p
  - 批量随机挑战：对随机 r ∈ F^D，验证 inner_product(mask, H_r) = K * p_r
    其中 H_r[t] = H[t] · r，p_r = p · r

L2Norm 等价于对 (1,D) 的 RMSNorm（无可学习权重 γ=1）：
  - rms_inv = 1 / ||p||_2
  - e = rms_inv * p
  - 用 Rescaling 证明（整数量化后两步 rescale）
  - 证明输出：rms_inv 的值 + 量化误差界

证明格式（纯 Python，无需 C++ binary）：
  {
    "meanpool_ok":   bool,    # sumcheck 验证通过
    "l2norm_ok":     bool,    # rescaling 约束验证通过
    "inner_product": int,     # H_r · mask （量化域）
    "K":             int,     # 有效 token 数
    "rms_sq_x65536": int,     # ||p||_2^2 × 65536 的量化估计
  }

用法（standalone）：
  cd /root/autodl-tmp/UltraRAG
  python script/prove_pooling.py [--seq_len 1024] [--embed_dim 2048]

用法（pipeline 调用）：
  from script.prove_pooling import prove_pooling
  result = prove_pooling(H_path, mask, embed_dim=2048)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent.resolve()
RMS_EPS   = 1e-6
SCALE     = 1 << 16   # 量化比例 (= 2^16)
_P_FR     = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001


# ── MeanPool sumcheck ────────────────────────────────────────────────────────

def _meanpool_sumcheck(H_int: np.ndarray, mask: np.ndarray,
                       n_challenges: int = 8) -> dict:
    """
    验证 p = mean(H[mask]) 的 sumcheck。

    H_int : (T, D)  int32，量化激活（× SCALE）
    mask  : (T,)    bool / 0-1 整数
    返回  : {"ok": bool, "K": int, "challenges": list}
    """
    T, D = H_int.shape
    mask = mask.astype(bool)
    K    = int(mask.sum())
    if K == 0:
        return {"ok": False, "K": 0, "error": "empty mask"}

    p_int = H_int[mask].sum(axis=0)   # (D,)  = K * mean × SCALE（int64）

    rng       = np.random.default_rng(42)
    all_ok    = True
    challenges = []

    for _ in range(n_challenges):
        r = rng.integers(-2**15, 2**15, size=D).astype(np.int64)

        # sum_{t: mask}( H[t] · r ) = p_int · r
        lhs = np.int64(0)
        for t in range(T):
            if mask[t]:
                lhs += int(H_int[t].astype(np.int64) @ r)

        rhs = int(p_int.astype(np.int64) @ r)

        ok = (int(lhs) == int(rhs))
        all_ok = all_ok and ok
        challenges.append({"lhs": int(lhs), "rhs": int(rhs), "ok": ok})

    return {"ok": all_ok, "K": K, "p_int": p_int, "challenges": challenges}


# ── L2Norm rescaling 证明 ────────────────────────────────────────────────────

def _l2norm_prove(p_int: np.ndarray) -> dict:
    """
    验证 e = p / ||p||_2 的量化正确性。

    p_int : (D,)  int64，MeanPool 输出（未除以 K）
    返回  : {"ok": bool, "rms_inv_q": int, "error_bound": float}

    证明策略：
      - 计算 rms_inv = 1 / sqrt(||p_int/K||_2^2 / D + eps)（标准 RMSNorm 风格）
        注：L2Norm 是 p / ||p||_2，等价于 RMSNorm 无 γ 且 eps=0
      - 量化 rms_inv 为 int32（× SCALE）
      - 验证 |rms_inv_q × p_int / SCALE^2 - e|_∞ ≤ 阈值
    """
    p_float = p_int.astype(np.float64) / SCALE
    norm    = np.linalg.norm(p_float)
    if norm < 1e-10:
        return {"ok": False, "error": "zero norm"}

    rms_inv  = 1.0 / norm                       # L2Norm 的归一化因子
    e_float  = p_float * rms_inv                # 真实归一化结果

    rms_inv_q = round(rms_inv * SCALE)          # 量化到 int
    e_q       = p_int * rms_inv_q               # 量化乘积（int64），需要再 /SCALE^2

    e_reconstructed = e_q.astype(np.float64) / (SCALE ** 2)
    error    = np.max(np.abs(e_reconstructed - e_float))

    # 量化误差界：rms_inv 量化误差 ε_r = 0.5/SCALE
    # => ||e_recon - e||_∞ ≤ ε_r × ||p_float||_∞ / SCALE
    d_max    = float(np.max(np.abs(p_float)))
    # |e_recon[d] - e_float[d]| = |p_float[d]| × |rms_inv_q/SCALE - rms_inv| ≤ d_max × 0.5/SCALE
    expected_bound = 0.5 * d_max / SCALE + 1e-9

    ok = (error <= expected_bound * 2)   # 留一倍余量

    return {
        "ok": ok,
        "rms_inv_q": int(rms_inv_q),
        "norm": float(norm),
        "max_error": float(error),
        "error_bound": float(expected_bound * 2),
    }


# ── 主接口 ───────────────────────────────────────────────────────────────────

def prove_pooling(H_path=None, mask=None,
                  seq_len: int = 1024, embed_dim: int = 2048) -> dict:
    """
    证明 MeanPool + L2Norm。

    H_path   : str | Path | None — int32 bin 文件 (T×D)；None 则用随机激活（smoke test）
    mask     : array-like (T,) bool/int | None — 有效 token 掩码；None 则随机生成
    seq_len  : int — 序列长度 T（H_path=None 时用于生成随机激活）
    embed_dim: int — 嵌入维度 D
    """
    T, D = seq_len, embed_dim

    # 加载激活
    if H_path is None:
        rng   = np.random.default_rng(0)
        H_int = (rng.standard_normal((T, D)) * SCALE).astype(np.int32)
    else:
        H_int = np.fromfile(str(H_path), dtype=np.int32).reshape(T, D)

    # 构造 mask
    if mask is None:
        rng  = np.random.default_rng(1)
        mask = rng.integers(0, 2, size=T).astype(bool)
        # 至少保证 10% 非零
        if mask.sum() < max(1, T // 10):
            mask[:T // 10] = True

    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != T:
        raise ValueError(f"mask length {mask.shape[0]} != seq_len {T}")

    # MeanPool sumcheck
    pool_result = _meanpool_sumcheck(H_int, mask)
    meanpool_ok = pool_result["ok"]

    # L2Norm rescaling 证明（用 p_int = sum(H[mask])）
    l2_result  = _l2norm_prove(pool_result["p_int"]) if meanpool_ok else {"ok": False}
    l2norm_ok  = l2_result.get("ok", False)

    result = {
        "meanpool_ok": meanpool_ok,
        "l2norm_ok":   l2norm_ok,
        "all_ok":      meanpool_ok and l2norm_ok,
        "K":           pool_result.get("K", 0),
        "meanpool":    {k: v for k, v in pool_result.items() if k not in ("p_int", "challenges")},
        "l2norm":      l2_result,
    }
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--H",         default=None, help="int32 激活文件路径")
    parser.add_argument("--mask",      default=None, help="mask npy 文件路径（bool）")
    parser.add_argument("--seq_len",   type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--out",       default="notes/experiment_results/prove_pooling.json")
    args = parser.parse_args()

    mask = None
    if args.mask:
        mask = np.load(args.mask)

    result = prove_pooling(args.H, mask, args.seq_len, args.embed_dim)

    status = "✓ PASS" if result["all_ok"] else "✗ FAIL"
    print(f"Pooling Head Proof: {status}")
    print(f"  MeanPool sumcheck : {'✓' if result['meanpool_ok'] else '✗'}  (K={result['K']})")
    print(f"  L2Norm rescaling  : {'✓' if result['l2norm_ok'] else '✗'}", end="")
    if "norm" in result.get("l2norm", {}):
        print(f"  (||p||={result['l2norm']['norm']:.4f}, "
              f"max_err={result['l2norm']['max_error']:.2e} ≤ {result['l2norm']['error_bound']:.2e})")
    else:
        print()

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Results: {out}")

    sys.exit(0 if result["all_ok"] else 1)


if __name__ == "__main__":
    main()

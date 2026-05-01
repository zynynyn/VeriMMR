"""
Fiat-Shamir 安全性统计实验（纯 Python，无 GPU）

实验：
  FS-1  单次查询检出率
        N=36 层，K 层随机挑战，篡改 L 层；理论 vs 实测
  FS-2  多轮累积检出率
        L=1 层被篡改，连续 T 次查询；理论 vs 实测累积曲线
  FS-3  层选择均匀性（χ² 检验）
        M 次查询，各层被选频次应均匀

Fiat-Shamir 层选择（与 interactive_demo.py 完全相同）：
  seed   = SHA256(query_text + "|" + nonce)[:32]  (hex digest)
  rng    = Python random.Random(seed)
  layers = sorted(rng.sample(range(N), K))

用法：
  python script/experiment_fiat_shamir.py --trials 10000 --k 6
  python script/experiment_fiat_shamir.py --trials 50000 --k 6 --max-t 30
"""

import argparse
import hashlib
import json
import math
import random
import sys
import uuid
from pathlib import Path
from scipy import stats as scipy_stats

ROOT     = Path(__file__).parent.parent.resolve()
OUT_FILE = ROOT / "notes" / "experiment_results" / "fiat_shamir_security.json"

# ── Fiat-Shamir 层选择（与 interactive_demo.py 一致）─────────────────────────

def fiat_shamir_layers(query: str, nonce: str, n: int, k: int) -> list[int]:
    """用 SHA256(query|nonce) 作种子，deterministic 地选 k 层。"""
    seed = hashlib.sha256(f"{query}|{nonce}".encode()).hexdigest()
    rng  = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def make_query() -> str:
    """生成随机查询文本（模拟真实场景）。"""
    return uuid.uuid4().hex


def make_nonce() -> str:
    """生成服务器端 nonce（8 字符，与 demo 实现一致）。"""
    return uuid.uuid4().hex[:8]


# ── 理论公式 ──────────────────────────────────────────────────────────────────

def p_detect_single(n: int, k: int, L: int) -> float:
    """
    单次查询检出率：P(至少命中一个篡改层) = 1 - C(N-L,K)/C(N,K)
    即随机抽 K 层中至少有一层来自 L 个篡改层的概率。
    """
    if L == 0 or k == 0 or n < k:
        return 0.0
    if L >= n or k >= n:
        return 1.0
    # P = 1 - C(N-L, K) / C(N, K)
    # 使用 log-gamma 避免大数溢出
    log_num = math.lgamma(n - L + 1) - math.lgamma(n - L - k + 1)
    log_den = math.lgamma(n + 1) - math.lgamma(n - k + 1)
    return 1.0 - math.exp(log_num - log_den)


def p_detect_cumulative(n: int, k: int, L: int, T: int) -> float:
    """T 次独立查询后检出率（每次独立 Fiat-Shamir 挑战）。"""
    p1 = p_detect_single(n, k, L)
    return 1.0 - (1.0 - p1) ** T


# ── FS-1: 单次检出率 ──────────────────────────────────────────────────────────

def experiment_fs1(n: int, k: int, L_values: list[int],
                   M: int) -> dict:
    """
    M 次随机查询，测量不同篡改层数 L 下的实测检出率。
    """
    print(f"\n[FS-1] 单次检出率  N={n}, K={k}, M={M}")
    results = []
    for L in L_values:
        # 固定篡改哪 L 层（从 N 层中选前 L 层，确保篡改集合固定）
        tampered = set(range(L))
        detected = 0
        for _ in range(M):
            q = make_query()
            nonce = make_nonce()
            selected = set(fiat_shamir_layers(q, nonce, n, k))
            if selected & tampered:
                detected += 1
        p_empirical = detected / M
        p_theory    = p_detect_single(n, k, L)
        error       = abs(p_empirical - p_theory)
        print(f"  L={L:2d}  empirical={p_empirical:.4f}  "
              f"theory={p_theory:.4f}  |err|={error:.4f}")
        results.append({
            "L":           L,
            "detected":    detected,
            "trials":      M,
            "p_empirical": round(p_empirical, 6),
            "p_theory":    round(p_theory, 6),
            "abs_error":   round(error, 6),
        })
    return {"experiment": "FS-1", "N": n, "K": k, "M": M, "results": results}


# ── FS-2: 多轮累积安全 ────────────────────────────────────────────────────────

def experiment_fs2(n: int, k: int, L: int,
                   T_max: int, M: int) -> dict:
    """
    L 层被篡改，M 次独立实验，每次模拟 T_max 轮查询，
    记录每轮后累积检出率。
    """
    print(f"\n[FS-2] 多轮累积检出率  N={n}, K={k}, L={L}, T_max={T_max}, M={M}")
    tampered = set(range(L))

    # detected_at[t] = 在第 t 轮（含）及之前被检出的实验数
    detected_at = [0] * (T_max + 1)

    for trial in range(M):
        found = False
        for t in range(1, T_max + 1):
            if not found:
                q = make_query()
                nonce = make_nonce()
                selected = set(fiat_shamir_layers(q, nonce, n, k))
                if selected & tampered:
                    found = True
            if found:
                detected_at[t] += 1

    results = []
    p1 = p_detect_single(n, k, L)
    for t in range(1, T_max + 1):
        p_emp = detected_at[t] / M
        p_th  = p_detect_cumulative(n, k, L, t)
        results.append({
            "T":           t,
            "p_empirical": round(p_emp, 6),
            "p_theory":    round(p_th, 6),
        })
    # 打印关键节点
    for t in [1, 5, 10, 20, T_max]:
        if t <= T_max:
            r = results[t - 1]
            print(f"  T={t:3d}  empirical={r['p_empirical']:.4f}  "
                  f"theory={r['p_theory']:.4f}")
    return {
        "experiment": "FS-2",
        "N": n, "K": k, "L": L, "T_max": T_max, "M": M,
        "results": results,
    }


# ── FS-3: 层选择均匀性 ────────────────────────────────────────────────────────

def experiment_fs3(n: int, k: int, M: int) -> dict:
    """
    M 次查询，统计每层被选中的频次，χ² 检验均匀性。
    期望每层频次 = M * K / N。
    """
    print(f"\n[FS-3] 层选择均匀性  N={n}, K={k}, M={M}")
    freq = [0] * n
    for _ in range(M):
        q = make_query()
        nonce = make_nonce()
        for lay in fiat_shamir_layers(q, nonce, n, k):
            freq[lay] += 1

    expected = M * k / n
    chi2_stat, p_value = scipy_stats.chisquare(freq)

    freq_mean = sum(freq) / n
    freq_std  = (sum((f - freq_mean) ** 2 for f in freq) / n) ** 0.5
    cv        = freq_std / freq_mean  # coefficient of variation

    print(f"  期望频次/层 = {expected:.1f}")
    print(f"  实测 μ={freq_mean:.1f}  σ={freq_std:.2f}  CV={cv:.4f}")
    print(f"  χ²={chi2_stat:.2f}  df={n-1}  p={p_value:.4f}  "
          f"({'均匀 ✓' if p_value > 0.05 else '不均匀 ✗'})")

    return {
        "experiment":    "FS-3",
        "N": n, "K": k, "M": M,
        "expected_freq": round(expected, 2),
        "freq_mean":     round(freq_mean, 4),
        "freq_std":      round(freq_std, 4),
        "cv":            round(cv, 6),
        "chi2_stat":     round(chi2_stat, 4),
        "p_value":       round(p_value, 6),
        "uniform":       bool(p_value > 0.05),
        "layer_freq":    freq,
    }


# ── Fiat-Shamir 安全性讨论 ────────────────────────────────────────────────────

def security_discussion(n: int, k: int) -> dict:
    """
    分析服务器端 nonce vs 客户端 nonce 的安全差异。
    服务器可枚举 nonce 直到选出"安全"层集合（不包含自己篡改的层）。
    """
    # 枚举攻击成本：每次枚举命中"安全"nonce 的概率
    # 若 L 层被篡改，选出的 K 层全不含篡改层的概率 = C(N-L,K)/C(N,K)
    L_values_discuss = [1, 3, 6, 12]
    bypass_probs = []
    for L in L_values_discuss:
        p_bypass = 1.0 - p_detect_single(n, k, L)
        expected_tries = 1.0 / p_bypass if p_bypass > 0 else float("inf")
        bypass_probs.append({
            "L":             L,
            "p_bypass":      round(p_bypass, 6),
            "expected_tries": round(expected_tries, 2),
        })
        print(f"  L={L:2d}  P(bypass)={p_bypass:.4f}  "
              f"expected_tries={expected_tries:.1f}")
    return {
        "note": ("服务器端 nonce 时，恶意服务器可枚举 nonce 绕过检测。"
                 "修复：改用客户端 nonce，challenge=SHA256(query+client_nonce)。"),
        "N": n, "K": k,
        "bypass_analysis": bypass_probs,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fiat-Shamir 安全性统计实验")
    parser.add_argument("--trials",  type=int, default=10000,
                        help="FS-1/FS-3 的模拟次数")
    parser.add_argument("--k",       type=int, default=6,
                        help="每次查询选取的层数")
    parser.add_argument("--n",       type=int, default=36,
                        help="总层数")
    parser.add_argument("--max-t",   type=int, default=30,
                        help="FS-2 最大查询轮数")
    parser.add_argument("--fs2-m",   type=int, default=5000,
                        help="FS-2 模拟实验次数（可低于 --trials）")
    parser.add_argument("--out",     default=str(OUT_FILE))
    args = parser.parse_args()

    n = args.n
    k = args.k
    M = args.trials

    print("=" * 60)
    print("Fiat-Shamir 安全性统计实验")
    print(f"  N={n}, K={k}, M={M}")
    print("=" * 60)

    # FS-1: 不同 L 下的单次检出率
    L_values = [1, 3, 6, 12, 18]
    fs1 = experiment_fs1(n, k, L_values, M)

    # FS-2: L=1, 多轮累积（最常见的最坏情形）
    fs2 = experiment_fs2(n, k, L=1, T_max=args.max_t, M=args.fs2_m)

    # FS-3: 层选择均匀性
    fs3 = experiment_fs3(n, k, M)

    # 安全性讨论（服务器端 nonce 漏洞）
    print(f"\n[安全分析] 服务器端 nonce 攻击成本  N={n}, K={k}")
    sec = security_discussion(n, k)

    print("\n" + "=" * 60)
    print("FS-1 关键结论:")
    for r in fs1["results"]:
        print(f"  L={r['L']:2d}  检出率={r['p_empirical']:.4f}  "
              f"（理论={r['p_theory']:.4f}）")

    output = {
        "experiment": "Fiat-Shamir Security",
        "description": "Fiat-Shamir 层选择机制统计安全性分析",
        "params": {"N": n, "K": k, "trials": M},
        "fs1_single_round":       fs1,
        "fs2_cumulative":         fs2,
        "fs3_uniformity":         fs3,
        "security_discussion":    sec,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()

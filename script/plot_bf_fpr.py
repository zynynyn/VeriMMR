"""
BF 误报率分析图 — 原版样式，生成 4 张独立 PNG
  bf_fpr_seed42.png / bf_fpr_seed123.png / bf_fpr_seed999.png
  bf_fpr_combined.png
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import binom

NOTES   = Path(__file__).parent.parent / "notes"
OUT_DIR = NOTES / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPS = 0.01

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, center - margin), center + margin

# ── 数据 ─────────────────────────────────────────────────────────────────────
runs_raw = {}
for key in ["seed42", "seed123", "seed999"]:
    d = json.loads((NOTES / f"experiment_c1_bf_fpr_{key}.json").read_text())
    r = d["results"]
    runs_raw[key] = {
        "b1": r["B1_image_replace"]["fp"],
        "b2": r["B2_embedding_replace"]["fp"],
    }

total_b1 = sum(v["b1"] for v in runs_raw.values())
total_b2 = sum(v["b2"] for v in runs_raw.values())

TASKS = [
    ("seed42",    runs_raw["seed42"]["b1"],  runs_raw["seed42"]["b2"],  400,  "seed=42"),
    ("seed123",   runs_raw["seed123"]["b1"], runs_raw["seed123"]["b2"], 400,  "seed=123"),
    ("seed999",   runs_raw["seed999"]["b1"], runs_raw["seed999"]["b2"], 400,  "seed=999"),
    ("combined",  total_b1,                  total_b2,                  1200, "Combined (3 runs)"),
]

# ── 画图函数（原版样式）───────────────────────────────────────────────────────
def make_figure(tag, fp_b1, fp_b2, N, label):
    fp_tot   = fp_b1 + fp_b2
    N2       = N * 2
    fpr_b1   = fp_b1 / N
    fpr_b2   = fp_b2 / N
    fpr_tot  = fp_tot / N2
    b1_lo, b1_hi = wilson_ci(fp_b1, N)
    b2_lo, b2_hi = wilson_ci(fp_b2, N)
    t_lo,  t_hi  = wilson_ci(fp_tot, N2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Bloom Filter False Positive Rate Analysis — {label}\n"
        f"ZAC Accumulator — B1 (Image Replace) & B2 (Embedding Replace)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ══════════════════════════════════════════════════════
    # 左图：FPR 柱状图
    # ══════════════════════════════════════════════════════
    labels  = ["B1\nImage Replace", "B2\nEmbedding Replace", f"Combined\n(B1+B2)"]
    fprs    = [fpr_b1,  fpr_b2,  fpr_tot]
    ci_los  = [b1_lo,   b2_lo,   t_lo]
    ci_his  = [b1_hi,   b2_hi,   t_hi]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    counts  = [f"({fp_b1}/{N})", f"({fp_b2}/{N})", f"({fp_tot}/{N2})"]

    x = np.arange(3)
    ax1.bar(x, [v*100 for v in fprs], width=0.5,
            color=colors, alpha=0.85, zorder=3)

    for i, (fpr, lo, hi, cnt) in enumerate(zip(fprs, ci_los, ci_his, counts)):
        ax1.errorbar(x[i], fpr*100,
                     yerr=[[fpr*100 - lo*100], [hi*100 - fpr*100]],
                     fmt="none", color="black", capsize=6, capthick=1.5,
                     linewidth=1.5, zorder=4)
        ax1.text(x[i], hi*100 + 0.1,
                 f"{fpr*100:.2f}%\n{cnt}",
                 ha="center", va="bottom", fontsize=9,
                 color=colors[i], fontweight="bold")

    ax1.axhline(EPS*100, color="crimson", linewidth=1.8, linestyle="--",
                zorder=5, label=f"Theoretical ε = 1%")
    ax1.axhspan(t_lo*100, t_hi*100, alpha=0.10, color="#55A868",
                label=f"Combined 95% CI [{t_lo*100:.2f}%, {t_hi*100:.2f}%]")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("False Positive Rate (%)", fontsize=11)
    ax1.set_title(f"Observed FPR with 95% Wilson CI\n"
                  f"(n={N} per type, n={N2} combined)", fontsize=10)
    ax1.set_ylim(0, max(b1_hi, b2_hi, t_hi)*100 * 1.5 + 0.3)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    for i, (lo, hi) in enumerate(zip(ci_los, ci_his)):
        in_ci = lo <= EPS <= hi
        ax1.text(x[i], -0.22, "✓ ε ∈ CI" if in_ci else "✗ ε ∉ CI",
                 ha="center", va="top", fontsize=8.5,
                 color="green" if in_ci else "red",
                 transform=ax1.get_xaxis_transform())

    # ══════════════════════════════════════════════════════
    # 右图：二项分布 PMF
    # ══════════════════════════════════════════════════════
    k_max   = max(int(N * EPS * 3.5), fp_b1 + 3, fp_b2 + 3)
    k_range = np.arange(0, k_max + 1)
    pmf     = binom.pmf(k_range, N, EPS)
    exp_k   = N * EPS
    lo_k    = int(binom.ppf(0.025, N, EPS))
    hi_k    = int(binom.ppf(0.975, N, EPS))

    # 95% 接受域底色
    ax2.axvspan(lo_k - 0.5, hi_k + 0.5, alpha=0.12, color="crimson",
                label=f"95% acceptance region [{lo_k}, {hi_k}]", zorder=1)

    # PMF 柱
    ax2.bar(k_range, pmf*100, width=0.75, color="#9ecae1", alpha=0.8,
            zorder=2, label=f"Binom(n={N}, p={EPS}) PMF")

    # 观测值高亮柱（叠在 PMF 上，标签含完整信息）
    ax2.bar(fp_b1, pmf[fp_b1]*100, width=0.75, color="#4C72B0",
            alpha=0.9, zorder=3,
            label=f"B1 obs={fp_b1}  FPR={fpr_b1*100:.2f}%  p(k={fp_b1})={pmf[fp_b1]*100:.2f}%")
    ax2.bar(fp_b2, pmf[fp_b2]*100, width=0.75, color="#DD8452",
            alpha=0.9, zorder=3,
            label=f"B2 obs={fp_b2}  FPR={fpr_b2*100:.2f}%  p(k={fp_b2})={pmf[fp_b2]*100:.2f}%")

    # 期望值线
    ax2.axvline(exp_k, color="crimson", linewidth=1.8, linestyle="--",
                zorder=4, label=f"E[k] = n·ε = {exp_k:.1f}")

    ax2.set_xlabel(f"Number of False Positives k  (out of {N} trials)", fontsize=10)
    ax2.set_ylabel("Probability (%)", fontsize=11)
    ax2.set_title(
        f"Binomial Distribution: Binom(n={N}, p={EPS})\n"
        f"Expected false positives = {exp_k:.1f};  "
        f"B1={fp_b1}, B2={fp_b2} both plausible",
        fontsize=10,
    )
    ax2.set_xlim(-0.5, k_max + 0.5)
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.grid(axis="y", alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)

    # 底部说明
    fig.text(
        0.5, -0.04,
        f"Left: Error bars = 95% Wilson CI.  "
        f"Right: Shaded = 95% acceptance region under H₀: p={EPS}.\n"
        f"B1 ({fp_b1}/{N}={fpr_b1*100:.2f}%) and "
        f"B2 ({fp_b2}/{N}={fpr_b2*100:.2f}%) "
        f"{'are' if (b1_lo<=EPS<=b1_hi and b2_lo<=EPS<=b2_hi) else 'not both'} "
        f"consistent with ε={EPS}.  "
        f"Combined: {fp_tot}/{N2} = {fpr_tot*100:.2f}% "
        f"({'≈' if abs(fpr_tot-EPS)<0.002 else '≠'} ε).",
        ha="center", fontsize=9, color="gray",
    )

    plt.tight_layout()
    out_path = OUT_DIR / f"bf_fpr_{tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out_path.name}  "
          f"B1={fp_b1}/{N} ({fpr_b1*100:.2f}%)  "
          f"B2={fp_b2}/{N} ({fpr_b2*100:.2f}%)  "
          f"Combined={fp_tot}/{N2} ({fpr_tot*100:.2f}%)")

# ── 生成 4 张 ─────────────────────────────────────────────────────────────────
for tag, fp_b1, fp_b2, N, label in TASKS:
    make_figure(tag, fp_b1, fp_b2, N, label)

print(f"\n输出目录：{OUT_DIR}")

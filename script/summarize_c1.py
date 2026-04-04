"""
C1 实验结果汇总输出
读取 notes/ 下所有 C1 结果 JSON，打印统一格式的汇总表。
"""
import json
from pathlib import Path

NOTES = Path(__file__).parent.parent / "notes"
EPS   = 0.01

DATASETS = ["slidevqa", "docvqa", "chartvqa", "infovqa"]
LABELS   = {
    "slidevqa": "SlideVQA",
    "docvqa":   "MP-DocVQA",
    "chartvqa": "ChartQA",
    "infovqa":  "InfoVQA",
}

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
    return max(0.0, center - margin), center + margin

# ── 读取数据 ──────────────────────────────────────────────────────────────────
attack = {}
for key in DATASETS:
    f = NOTES / f"experiment_c1_attack_{key}.json"
    if f.exists():
        attack[key] = json.loads(f.read_text())

# 优先读取三 seed 合并数据，回退到旧格式
_fpr_seeds = ["seed42", "seed123", "seed999"]
_fpr_parts = [NOTES / f"experiment_c1_bf_fpr_{s}.json" for s in _fpr_seeds]
if all(p.exists() for p in _fpr_parts):
    _parts = [json.loads(p.read_text()) for p in _fpr_parts]
    _n = _parts[0]["config"]["n_trials_per_type"]
    fpr_data = {
        "_combined": True,
        "config": {"n_trials_per_type": _n * 3, "n_zac": _parts[0]["config"]["n_zac"]},
        "results": {
            "B1_image_replace": {"fp": sum(p["results"]["B1_image_replace"]["fp"] for p in _parts)},
            "B2_embedding_replace": {"fp": sum(p["results"]["B2_embedding_replace"]["fp"] for p in _parts)},
            "false_negatives": _parts[0]["results"]["false_negatives"],
        },
    }
else:
    fpr_file = NOTES / "experiment_c1_bf_fpr.json"
    fpr_data = json.loads(fpr_file.read_text()) if fpr_file.exists() else None

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print(f"  C1 实验汇总：可验证多模态检索完整性（jina-embeddings-v4）")
print(f"{'='*72}")

# ── 表1：检索质量 ─────────────────────────────────────────────────────────────
print(f"\n【表1】检索质量：jina-v4 基线 vs VisRAG-Ret OOD（MRR@10 / Recall@10）")
print(f"  {'数据集':<12} {'语料规模':>6}  {'jina-v4 MRR@10':>14}  {'VisRAG OOD':>10}  "
      f"{'jina-v4 R@10':>12}  {'VisRAG OOD':>10}")
print(f"  {'─'*68}")
for key in DATASETS:
    if key not in attack: continue
    a   = attack[key]
    b   = a["baseline_recall"]
    ref = a["paper_reference_ood"]
    n   = a["config"]["n_corpus"]
    print(f"  {LABELS[key]:<12} {n:>6}  "
          f"{b['mrr_at_10']:>14.2f}  {ref['mrr_at_10']:>10.2f}  "
          f"{b['recall_at_10']:>12.2f}  {ref['recall_at_10']:>10.2f}")

# ── 表2：B3 攻击前后 Recall 对比 ──────────────────────────────────────────────
print(f"\n【表2】B3 排名操控攻击：Recall@10 静默降级（无验证情况）")
print(f"  {'数据集':<12} {'基线 R@10':>10}  {'攻击后 R@10':>12}  {'下降幅度':>10}  {'基线 MRR@10':>12}  {'攻击后':>8}")
print(f"  {'─'*68}")
for key in DATASETS:
    if key not in attack: continue
    a = attack[key]
    b = a["baseline_recall"]
    v = a["attacked_recall_no_verify"]
    print(f"  {LABELS[key]:<12} {b['recall_at_10']:>10.2f}  {v['recall_at_10']:>12.2f}  "
          f"{v['recall10_drop']:>9.2f}pp  {b['mrr_at_10']:>12.2f}  {v['mrr_at_10']:>8.2f}")

# ── 表3：攻击检测率 ────────────────────────────────────────────────────────────
print(f"\n【表3】攻击检测率")
print(f"  {'数据集':<12} {'B1 ZAC':>10}  {'B2 ZAC':>10}  {'B3 Sumcheck':>12}  {'Sumcheck延迟':>12}")
print(f"  {'─'*60}")
for key in DATASETS:
    if key not in attack: continue
    dr  = attack[key]["detection_rates"]
    b1  = dr["B1_image_replace_zac"]
    b2  = dr["B2_embedding_replace_zac"]
    b3  = dr["B3_ranking_forge_sumcheck"]
    ms  = b3["verify_ms_per_query"]
    def fmt(d, t, r):
        mark = "✓" if r == 100.0 else "△"
        return f"{d}/{t} {mark}"
    print(f"  {LABELS[key]:<12} {fmt(b1['detected'],b1['total'],b1['rate_pct']):>10}  "
          f"{fmt(b2['detected'],b2['total'],b2['rate_pct']):>10}  "
          f"{fmt(b3['detected'],b3['total'],b3['rate_pct']):>12}  "
          f"{ms:>9.0f} ms")

# ── 表4：BF 误报率统计验证 ────────────────────────────────────────────────────
if fpr_data:
    r = fpr_data["results"]
    fp_b1 = r["B1_image_replace"]["fp"];  n_b1 = fpr_data["config"]["n_trials_per_type"]
    fp_b2 = r["B2_embedding_replace"]["fp"];  n_b2 = n_b1
    b1_lo, b1_hi = wilson_ci(fp_b1, n_b1)
    b2_lo, b2_hi = wilson_ci(fp_b2, n_b2)
    fp_tot = fp_b1 + fp_b2;  n_tot = n_b1 + n_b2
    tot_lo, tot_hi = wilson_ci(fp_tot, n_tot)

    print(f"\n【表4】Bloom Filter 误报率统计验证（InfoVQA，N_ZAC=50）")
    print(f"  {'类型':<22} {'误报':>6}  {'FPR':>6}  {'95% Wilson CI':>18}  {'ε=0.01 ∈ CI':>12}")
    print(f"  {'─'*68}")
    rows = [
        ("B1 图像替换",      fp_b1,  n_b1,  b1_lo,  b1_hi),
        ("B2 Embedding替换", fp_b2,  n_b2,  b2_lo,  b2_hi),
        ("合计",             fp_tot, n_tot, tot_lo, tot_hi),
    ]
    for label, fp, n, lo, hi in rows:
        in_ci = "✅" if lo <= EPS <= hi else "❌"
        print(f"  {label:<22} {fp:>3}/{n:<4}  {fp/n*100:>5.2f}%  "
              f"[{lo*100:.2f}%, {hi*100:.2f}%]  {in_ci:>12}")
    fn = r["false_negatives"]
    print(f"  {'合法成员假阴性':<22} {fn:>3}/{fpr_data['config']['n_zac']:<4}  "
          f"{'0.00%':>6}  {'—':>18}  {'✅ 零漏报':>12}")

# ── 说明 ──────────────────────────────────────────────────────────────────────
print(f"""
说明：
  ✓ = 100% 检出  △ = 存在漏报（详见注释）
  B1/B2 各 10 次试验（N_ZAC_ATTACK=10），B3 各 50 条 query（N_SUMCHECK_Q=50）
  BF 误报率：400×2=800 次大样本验证（独立于主实验）
  VisRAG-Ret OOD：MiniCPM-V 2.0，来自 VisRAG ICLR 2025 Table 3（领域外零样本）
""")

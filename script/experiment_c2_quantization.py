"""
实验 C2：Sumcheck 量化误差分析

目标：验证 float32 embedding 在 Sumcheck 整数域量化后，内积精度损失在可接受范围内，
     且 top-k 检索排名不受影响。

量化方案（与 src/sumcheck/inner_product.py 一致）：
  - 缩放因子 scale=256（jina-v4 embedding 值域约 ±1，量化后约 ±256）
  - 域：Z_p，p = 2^61 - 1（Mersenne 素数）
  - 量化：int(round(x * scale)) mod p
"""

import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "sumcheck"))
from inner_product import quantize, P

EMB_PATH = ROOT / "embedding" / "embedding.npy"
TOP_K    = 5
SCALES   = [256, 65536]   # 256=原始选择，65536=zkLLM对齐方案

# ── 加载 corpus embedding ─────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"实验 C2：Sumcheck 量化误差分析")
print(f"{'='*55}")

corpus = np.load(EMB_PATH).astype(np.float32)   # (N, D)
N, D = corpus.shape
queries = {
    "corpus[0]（语料库内 query）": corpus[0],
    "corpus[100]":                corpus[100],
    "随机单位向量":               (lambda v: v / np.linalg.norm(v))(
                                    np.random.default_rng(42).standard_normal(D).astype(np.float32)),
}

import time
scale_results = {}

for SCALE in SCALES:
    print(f"\n{'─'*55}")
    print(f"scale = {SCALE}  (溢出上界校验: {SCALE**2 * D:.2e} << p={P:.2e}  {'✅' if SCALE**2 * D < P else '❌ 溢出！'})")
    print(f"{'─'*55}")

    all_results = []
    t0 = time.perf_counter()

    for qname, q_float in queries.items():
        print(f"\n  Query: {qname}")

        scores_f32 = corpus @ q_float

        q_int    = np.array(quantize(q_float.tolist(), SCALE), dtype=object)
        corp_int = np.array([quantize(v.tolist(), SCALE) for v in corpus], dtype=object)
        scores_int_raw    = corp_int @ q_int
        scores_int_mod    = scores_int_raw % P
        half_p            = P // 2
        scores_int_signed = np.array([
            int(s) if int(s) <= half_p else int(s) - P
            for s in scores_int_mod
        ], dtype=np.float64)
        scores_int_f = scores_int_signed / (SCALE ** 2)

        diff     = np.abs(scores_f32.astype(np.float64) - scores_int_f)
        l_inf    = diff.max()
        l1_mean  = diff.mean()
        rel_err  = (diff / (np.abs(scores_f32) + 1e-8)).mean()

        top5_f   = set(np.argsort(scores_f32)[-TOP_K:])
        top5_int = set(np.argsort(scores_int_f)[-TOP_K:])
        top10_f  = set(np.argsort(scores_f32)[-10:])
        top10_int= set(np.argsort(scores_int_f)[-10:])

        print(f"    L∞={l_inf:.2e}  L1={l1_mean:.2e}  rel={rel_err:.2e}  "
              f"top5={'✅' if top5_f==top5_int else '❌'}  "
              f"top10={'✅' if top10_f==top10_int else '❌'}")

        all_results.append({
            "query": qname, "l_inf": float(l_inf), "l1_mean": float(l1_mean),
            "rel_err": float(rel_err),
            "top5_match": bool(top5_f == top5_int),
            "top10_match": bool(top10_f == top10_int),
        })

    elapsed = time.perf_counter() - t0
    all_top5  = all(r["top5_match"]  for r in all_results)
    all_top10 = all(r["top10_match"] for r in all_results)
    max_linf  = max(r["l_inf"]  for r in all_results)
    max_rel   = max(r["rel_err"] for r in all_results)

    print(f"\n  [scale={SCALE} 汇总]  耗时={elapsed:.1f}s  "
          f"top5={'✅' if all_top5 else '❌'}  top10={'✅' if all_top10 else '❌'}  "
          f"max_L∞={max_linf:.2e}  max_rel={max_rel:.2e}")

    scale_results[SCALE] = {
        "elapsed_s": elapsed, "all_top5": all_top5, "all_top10": all_top10,
        "max_l_inf": max_linf, "max_rel_err": max_rel, "queries": all_results,
    }

# ── 对比总结 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"scale 对比总结（N={N}, D={D}）")
print(f"{'='*55}")
print(f"  {'scale':>8}  {'耗时':>8}  {'top-5':>6}  {'top-10':>7}  {'max L∞':>10}  {'max rel':>10}")
for s, r in scale_results.items():
    print(f"  {s:>8}  {r['elapsed_s']:>7.1f}s  "
          f"{'✅' if r['all_top5'] else '❌':>6}  "
          f"{'✅' if r['all_top10'] else '❌':>7}  "
          f"{r['max_l_inf']:>10.2e}  {r['max_rel_err']:>10.2e}")
print(f"{'='*55}\n")

out_path = ROOT / "notes" / "experiment_c2_result.json"
with open(out_path, "w") as f:
    json.dump({"N": N, "D": D, "top_k": TOP_K, "scales": scale_results}, f, indent=2)
print(f"结果已保存至：{out_path}")

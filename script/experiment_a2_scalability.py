"""
实验 A2：N 扩展性
目标：量化语料库规模 N 对各验证组件延迟的影响。

各组件复杂度理论：
  FAISS  IndexFlatIP : O(N·D)  但 D=2048 固定，N<1000 时 GPU 上极快
  Sumcheck            : O(N·D)  prove_global_batch 对 N 条向量做内积再证明
  ZAC 成员证明         : O(k)   k=top-k=5，与 N 无关（ZAC 优于 Merkle 的关键性质）

实验设计：
  N ∈ {50, 100, 200, 303, 500, 1000}
  N ≤ 303 : 直接切片已有 embedding.npy
  N > 303 : 补充随机单位向量（归一化，仅用于计时，语义无意义）
  5 条 query × 10 次重复，取中位数 ± IQR
  ZAC 始终使用真实语料库的 k=5 个元素（证明 O(k) 常数性）
"""

import sys, json, time, statistics
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zac.accumulator import ZACAccumulator
from sumcheck.inner_product import prove_global_batch, verify_global_batch

# ── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = "/root/autodl-tmp/models/jina-embeddings-v4"
INDEX_PATH    = str(ROOT / "index" / "index.index")
CORPUS_JSONL  = ROOT / "corpora" / "image.jsonl"
EMBEDDING_NPY = ROOT / "embedding" / "embedding.npy"
PROVER_STATE  = ROOT / "output" / "phase1" / "prover_state.json"

TOP_K    = 5
N_REPEAT = 10   # 每个 N × 每条 query 重复次数
N_VALUES = [50, 100, 200, 303, 500, 1000]

QUERIES = [
    "What is the sensor resolution of the Nikon Z8?",
    "How to set ISO sensitivity in manual exposure mode?",
    "Battery life and USB-C charging specifications",
    "AF tracking performance for 4K 60fps video recording",
    "Differences between Nikon Z8 and Z9 in weight and price",
]

print(f"\n{'='*64}")
print("实验 A2：N 扩展性")
print(f"{'='*64}")
print(f"  N 值   : {N_VALUES}")
print(f"  Queries: {len(QUERIES)}")
print(f"  Repeats: {N_REPEAT} 次/query")

# ── 加载模型与基础数据 ────────────────────────────────────────────────────────
print(f"\n[加载] 模型和索引...")

import torch
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cuda:0")
model.eval()
print(f"  jina-v4 loaded  ({time.time()-t0:.1f}s)")

# 原始 embedding（N_real=303）
t0 = time.time()
emb_real = np.load(EMBEDDING_NPY).astype(np.float32)
N_real, D = emb_real.shape
corpus_paths = [json.loads(l)["image_path"] for l in CORPUS_JSONL.read_text().splitlines()]
base_dir     = CORPUS_JSONL.parent
all_paths    = [str(base_dir / p) for p in corpus_paths]
print(f"  Embeddings loaded  N_real={N_real}  D={D}  ({time.time()-t0:.1f}s)")

# ZAC（仅用于 k=5 成员证明计时，不重建）
t0 = time.time()
acc = ZACAccumulator.load_prover_state(str(PROVER_STATE))
print(f"  ZAC prover state loaded  ({time.time()-t0:.1f}s)")

# 预计算 ZAC elements（真实的 k=5 个元素，固定用于所有 N 的 ZAC 计时）
# ZAC 成员证明复杂度 O(k)，与 N 无关，此处用固定元素集演示常数性
_fixed_ids   = list(range(TOP_K))   # 固定取前 5 张图
zac_elements = [
    ZACAccumulator.image_embedding_hash(all_paths[i], emb_real[i])
    for i in _fixed_ids
]
print(f"  ZAC elements 预计算完成（{TOP_K} 个）")

# 预先编码所有 queries（不计入每轮计时）
print(f"\n[预编码] {len(QUERIES)} 条 query...")
with torch.no_grad():
    q_embs = model.encode(
        QUERIES,
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=False,
        batch_size=len(QUERIES),
    )
q_vecs = [q_embs[i].tolist() for i in range(len(QUERIES))]
print(f"  完成  shape={q_embs.shape}")

# ── 计时辅助 ──────────────────────────────────────────────────────────────────
def median_iqr(vals):
    s  = sorted(vals)
    med = statistics.median(s)
    q1  = s[max(0, len(s) // 4 - 1)]
    q3  = s[min(len(s)-1, 3 * len(s) // 4)]
    return med, q3 - q1

import faiss as _faiss

# ── 主循环：逐 N 计时 ─────────────────────────────────────────────────────────
results = {}   # N -> {"faiss": [...], "sumcheck": [...], "zac": [...]}

for N in N_VALUES:
    print(f"\n{'─'*64}")
    print(f"N = {N}")
    print(f"{'─'*64}")

    # ── 构建该 N 的 embedding 矩阵 ──────────────────────────────────────────
    if N <= N_real:
        emb_N = emb_real[:N].copy()
    else:
        # 补充随机单位向量
        rng   = np.random.default_rng(42)
        extra = rng.standard_normal((N - N_real, D)).astype(np.float32)
        norms = np.linalg.norm(extra, axis=1, keepdims=True)
        extra /= norms
        emb_N = np.vstack([emb_real, extra])

    corpus_vecs_N = emb_N.tolist()   # Sumcheck 输入

    # ── 构建该 N 的 FAISS 索引（CPU，N<1000 差异可忽略）──────────────────────
    idx = _faiss.IndexFlatIP(D)
    idx = _faiss.IndexIDMap2(idx)
    idx.add_with_ids(emb_N, np.arange(N, dtype=np.int64))
    print(f"  FAISS index built  ntotal={idx.ntotal}")

    t_faiss_all    = []
    t_sumcheck_all = []
    t_zac_all      = []

    for qi, (query, q_vec) in enumerate(zip(QUERIES, q_vecs)):
        q_np = np.array([q_embs[qi]], dtype=np.float32)

        tf, ts, tz = [], [], []
        for _ in range(N_REPEAT):
            # FAISS
            t = time.perf_counter()
            scores_f, ids_f = idx.search(q_np, min(TOP_K, N))
            tf.append((time.perf_counter() - t) * 1000)

            # Sumcheck
            t = time.perf_counter()
            proof = prove_global_batch(q_vec, corpus_vecs_N)
            vr    = verify_global_batch(q_vec, corpus_vecs_N, proof, TOP_K)
            ts.append((time.perf_counter() - t) * 1000)

            # ZAC（固定 k=5 元素，与 N 无关）
            t = time.perf_counter()
            zac_proof = acc.prove_membership_batch(zac_elements)
            zac_ok    = acc.verify_membership_batch(zac_elements, zac_proof)
            tz.append((time.perf_counter() - t) * 1000)

        t_faiss_all.extend(tf)
        t_sumcheck_all.extend(ts)
        t_zac_all.extend(tz)

        med_f, _ = median_iqr(tf)
        med_s, _ = median_iqr(ts)
        med_z, _ = median_iqr(tz)
        print(f"  Q{qi+1}: faiss={med_f:.1f}ms  sumcheck={med_s:.0f}ms  zac={med_z:.0f}ms")

    med_f, iqr_f = median_iqr(t_faiss_all)
    med_s, iqr_s = median_iqr(t_sumcheck_all)
    med_z, iqr_z = median_iqr(t_zac_all)

    results[N] = {
        "faiss_median_ms":    round(med_f, 2),
        "faiss_iqr_ms":       round(iqr_f, 2),
        "sumcheck_median_ms": round(med_s, 2),
        "sumcheck_iqr_ms":    round(iqr_s, 2),
        "zac_median_ms":      round(med_z, 2),
        "zac_iqr_ms":         round(iqr_z, 2),
        "raw_faiss_ms":       [round(x, 2) for x in t_faiss_all],
        "raw_sumcheck_ms":    [round(x, 2) for x in t_sumcheck_all],
        "raw_zac_ms":         [round(x, 2) for x in t_zac_all],
    }
    print(f"  ── 汇总 N={N} ──")
    print(f"  FAISS    中位数={med_f:.1f}ms  IQR={iqr_f:.1f}ms")
    print(f"  Sumcheck 中位数={med_s:.0f}ms  IQR={iqr_s:.0f}ms")
    print(f"  ZAC      中位数={med_z:.0f}ms  IQR={iqr_z:.0f}ms  （O(k)，k={TOP_K}）")

# ── 汇总报告 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("A2 N 扩展性汇总（中位数，ms）")
print(f"{'='*64}")
print(f"  {'N':>6}  {'FAISS':>8}  {'Sumcheck':>10}  {'ZAC':>8}")
print(f"  {'─'*44}")
for N in N_VALUES:
    r = results[N]
    print(f"  {N:>6}  {r['faiss_median_ms']:>8.1f}  "
          f"{r['sumcheck_median_ms']:>10.0f}  {r['zac_median_ms']:>8.0f}")

# 计算 Sumcheck 拟合斜率（log-log，验证线性关系）
import math
ns   = [N for N in N_VALUES if N <= N_real or True]
scs  = [results[N]["sumcheck_median_ms"] for N in ns]
log_ns  = [math.log(N) for N in ns]
log_scs = [math.log(s) for s in scs]
# 简单线性回归 log(sc) = a * log(N) + b
n_pts = len(log_ns)
mean_x = sum(log_ns) / n_pts
mean_y = sum(log_scs) / n_pts
slope  = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ns, log_scs)) / \
         sum((x - mean_x) ** 2 for x in log_ns)
print(f"\n  Sumcheck log-log 拟合斜率 = {slope:.3f}（理论 O(N^1.0) → 斜率≈1.0）")
print(f"  ZAC 中位数范围：{min(results[N]['zac_median_ms'] for N in N_VALUES):.0f}"
      f"–{max(results[N]['zac_median_ms'] for N in N_VALUES):.0f}ms（验证 O(k) 常数性）")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
def _json_safe(obj):
    if isinstance(obj, (np.integer, np.bool_)):  return int(obj)
    if isinstance(obj, np.floating):             return float(obj)
    if isinstance(obj, dict):   return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_json_safe(v) for v in obj]
    return obj

output = {
    "config": {
        "n_values":  N_VALUES,
        "n_queries": len(QUERIES),
        "n_repeat":  N_REPEAT,
        "top_k":     TOP_K,
        "D":         D,
        "N_real":    N_real,
        "note":      "N>303 补充随机单位向量；ZAC 始终用固定 k=5 真实元素（验证 O(k) 常数性）",
    },
    "results": {str(N): results[N] for N in N_VALUES},
    "sumcheck_loglog_slope": round(slope, 4),
}

out_path = ROOT / "notes" / "experiment_a2_result.json"
out_path.write_text(json.dumps(_json_safe(output), indent=2))
print(f"\n结果已保存至：{out_path}")

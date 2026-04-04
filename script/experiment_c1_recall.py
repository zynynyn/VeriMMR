"""
实验 C1：Recall@K / MRR@10 单数据集评测
用法：
  python script/experiment_c1_recall.py --dataset <key> [--no-clean]

  key 可选：slidevqa | docvqa | chartvqa | infovqa

  默认评测完成后自动清理 HuggingFace 缓存（加 --no-clean 可保留）。

VisRAG-Ret OOD 基准（论文 Table 3，ICLR 2025）：
  slidevqa   MRR@10=45.57  Recall@10=67.70
  docvqa     MRR@10=74.60  Recall@10=89.65
  chartvqa   MRR@10=75.99  Recall@10=91.40
  infovqa    MRR@10=67.26  Recall@10=87.05

编码参数与 VisRAG 流水线保持一致（visrag_parameter.yaml）：
  语料图像：task=retrieval，无 prompt_name
  查询文本：task=retrieval，prompt_name=query
"""

import argparse, json, shutil, sys, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── 数据集配置 ────────────────────────────────────────────────────────────────
DATASETS = {
    "slidevqa": {
        "hf_name":       "openbmb/VisRAG-Ret-Test-SlideVQA",
        "paper_mrr10":   45.57,
        "paper_recall10": 67.70,
    },
    "docvqa": {
        "hf_name":       "openbmb/VisRAG-Ret-Test-MP-DocVQA",
        "paper_mrr10":   74.60,
        "paper_recall10": 89.65,
    },
    "chartvqa": {
        "hf_name":       "openbmb/VisRAG-Ret-Test-ChartQA",
        "paper_mrr10":   75.99,
        "paper_recall10": 91.40,
    },
    "infovqa": {
        "hf_name":       "openbmb/VisRAG-Ret-Test-InfoVQA",
        "paper_mrr10":   67.26,
        "paper_recall10": 87.05,
    },
}

MODEL_PATH   = "/root/autodl-tmp/models/jina-embeddings-v4"
CACHE_DIR    = ROOT / "data" / "huggingface_cache"
NOTES_DIR    = ROOT / "notes"

EVAL_KS      = [1, 3, 5, 10]
TOP_K_SEARCH = 10
BATCH_IMG    = 4     # 图像编码批大小（显存约束）
BATCH_TXT    = 32    # 文本编码批大小

# ── 参数解析 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                    help="要评测的数据集")
parser.add_argument("--no-clean", action="store_true",
                    help="评测完成后保留 HuggingFace 缓存（默认自动清理）")
args = parser.parse_args()

key = args.dataset
cfg = DATASETS[key]
clean_cache = not args.no_clean

out_path = NOTES_DIR / f"experiment_c1_{key}.json"

def ts():
    return time.strftime("%H:%M:%S")

print(f"\n{'='*64}")
print(f"实验 C1：{key}  ({cfg['hf_name']})")
print(f"{'='*64}")
print(f"  模型      : {MODEL_PATH}")
print(f"  缓存目录  : {CACHE_DIR}")
print(f"  结果输出  : {out_path}")
print(f"  清理缓存  : {'是' if clean_cache else '否'}")
print(f"  开始时间  : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载 jina-v4 模型...")
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import io, faiss
from datasets import load_dataset

t0 = time.time()
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cuda:0")
model.eval()
print(f"  loaded  ({time.time()-t0:.1f}s)")

# ── 加载数据集 ────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载数据集...")
t0 = time.time()
ds_corpus  = load_dataset(cfg["hf_name"], "corpus",  split="train",
                          cache_dir=str(CACHE_DIR))
ds_queries = load_dataset(cfg["hf_name"], "queries", split="train",
                          cache_dir=str(CACHE_DIR))
ds_qrels   = load_dataset(cfg["hf_name"], "qrels",   split="train",
                          cache_dir=str(CACHE_DIR))
print(f"  corpus={len(ds_corpus)}  queries={len(ds_queries)}  qrels={len(ds_qrels)}"
      f"  ({time.time()-t0:.1f}s)")
print(f"  corpus 字段 : {ds_corpus.column_names}")
print(f"  qrels  字段 : {ds_qrels.column_names}")

# ── 构建 qrels 字典 ───────────────────────────────────────────────────────────
print(f"\n[{ts()}] 构建 qrels 索引...")
qid_f   = next(f for f in ds_qrels.column_names if "query"  in f.lower())
cid_f   = next(f for f in ds_qrels.column_names if "corpus" in f.lower())
score_f = next((f for f in ds_qrels.column_names if "score"  in f.lower()), None)

qrels_dict: dict[str, set] = {}
for row in ds_qrels:
    qid = str(row[qid_f])
    cid = str(row[cid_f])
    rel = int(row[score_f]) if score_f else 1
    if rel > 0:
        qrels_dict.setdefault(qid, set()).add(cid)
print(f"  有标注查询：{len(qrels_dict)} 条")

# ── 识别字段名（兼容 _id / id / doc_id 等变体）──────────────────────────────
corpus_id_field = next(
    (f for f in ds_corpus.column_names
     if f in ("_id", "id", "doc_id", "corpus_id", "corpus-id")),
    ds_corpus.column_names[0],
)
query_id_field = next(
    (f for f in ds_queries.column_names
     if f in ("_id", "id", "query_id", "query-id")),
    ds_queries.column_names[0],
)
query_text_field = next(
    (f for f in ds_queries.column_names
     if f in ("text", "query", "question")),
    None,
)
if query_text_field is None:
    raise ValueError(f"未找到查询文本字段，queries 字段为：{ds_queries.column_names}")

img_field = next(
    (f for f in ds_corpus.column_names
     if f in ("image", "img", "pixel_values", "image_bytes")),
    None,
)
if img_field is None:
    raise ValueError(f"未找到图像字段，corpus 字段为：{ds_corpus.column_names}")

print(f"  corpus_id 字段 : {corpus_id_field}")
print(f"  query_id  字段 : {query_id_field}")
print(f"  query_text字段 : {query_text_field}")
print(f"  图像字段       : {img_field}")

# ── 编码语料图像（流式批处理，避免全量载入内存）────────────────────────────────
print(f"\n[{ts()}] 编码语料图像（{len(ds_corpus)} 张，batch={BATCH_IMG}）...")
t_corpus = time.time()
corpus_ids_list  = []
corpus_embs_list = []

_batch_ids  = []
_batch_imgs = []

def _flush():
    if not _batch_imgs:
        return
    with torch.no_grad():
        emb = model.encode(
            _batch_imgs,
            task="retrieval",
            normalize_embeddings=True,
            batch_size=len(_batch_imgs),
            show_progress_bar=False,
        )
    corpus_embs_list.append(emb)
    corpus_ids_list.extend(_batch_ids)
    _batch_ids.clear()
    _batch_imgs.clear()

for i, row in enumerate(ds_corpus):
    img_data = row[img_field]
    if isinstance(img_data, Image.Image):
        pil = img_data.convert("RGB")
    elif isinstance(img_data, bytes):
        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
    elif isinstance(img_data, dict) and "bytes" in img_data:
        pil = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    else:
        raise TypeError(f"未知图像类型：{type(img_data)}")

    _batch_ids.append(str(row[corpus_id_field]))
    _batch_imgs.append(pil)

    if len(_batch_imgs) >= BATCH_IMG:
        _flush()

    if (i + 1) % 100 == 0 or (i + 1) == len(ds_corpus):
        elapsed = time.time() - t_corpus
        speed   = (i + 1) / elapsed if elapsed > 0 else 0
        eta     = (len(ds_corpus) - i - 1) / speed if speed > 0 else 0
        print(f"  [{ts()}] {i+1}/{len(ds_corpus)}  {speed:.2f} img/s  ETA {eta/60:.1f} min")

_flush()
corpus_matrix = np.vstack(corpus_embs_list).astype(np.float32)
encode_corpus_s = time.time() - t_corpus
print(f"  完成  shape={corpus_matrix.shape}  耗时={encode_corpus_s:.1f}s")

# ── 编码查询文本 ──────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 编码查询文本（{len(ds_queries)} 条）...")
t0 = time.time()
query_ids   = [str(row[query_id_field])   for row in ds_queries]
query_texts = [str(row[query_text_field]) for row in ds_queries]
with torch.no_grad():
    query_matrix = model.encode(
        query_texts,
        task="retrieval",
        prompt_name="query",
        normalize_embeddings=True,
        batch_size=BATCH_TXT,
        show_progress_bar=False,
    ).astype(np.float32)
encode_query_s = time.time() - t0
print(f"  完成  shape={query_matrix.shape}  耗时={encode_query_s:.1f}s")

# ── FAISS 构建与搜索 ──────────────────────────────────────────────────────────
print(f"\n[{ts()}] FAISS 构建索引并搜索...")
D = corpus_matrix.shape[1]
index = faiss.IndexFlatIP(D)
index = faiss.IndexIDMap2(index)
index.add_with_ids(corpus_matrix, np.arange(len(corpus_ids_list), dtype=np.int64))
print(f"  ntotal={index.ntotal}  D={D}")

t0 = time.time()
scores, int_ids = index.search(query_matrix, TOP_K_SEARCH)
search_s = time.time() - t0
print(f"  搜索完成  ({search_s:.3f}s)")

int_to_cid = dict(enumerate(corpus_ids_list))

# ── 计算指标 ──────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 计算指标...")
recalls    = {k: [] for k in EVAL_KS}
reciprocal = []
evaluated  = 0
skipped    = 0

for qi, qid in enumerate(query_ids):
    if qid not in qrels_dict:
        skipped += 1
        continue
    relevant = qrels_dict[qid]
    top_ids  = [int_to_cid[iid] for iid in int_ids[qi] if iid >= 0]

    rr = 0.0
    for rank, cid in enumerate(top_ids[:10], 1):
        if cid in relevant:
            rr = 1.0 / rank
            break
    reciprocal.append(rr)

    for k in EVAL_KS:
        recalls[k].append(float(bool(set(top_ids[:k]) & relevant)))

    evaluated += 1

mrr10       = float(np.mean(reciprocal) * 100)
recall_vals = {k: float(np.mean(recalls[k]) * 100) for k in EVAL_KS}

# ── 打印结果 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"结果：{key}")
print(f"{'='*64}")
print(f"  评测查询：{evaluated}  跳过：{skipped}")
print(f"  {'指标':<14} {'jina-v4':>10}  {'VisRAG-Ret OOD':>14}")
print(f"  {'─'*42}")
print(f"  {'MRR@10':<14} {mrr10:>10.2f}  {cfg['paper_mrr10']:>14.2f}")
for k in EVAL_KS:
    ref = cfg["paper_recall10"] if k == 10 else None
    ref_s = f"{ref:.2f}" if ref is not None else "—"
    print(f"  {'Recall@'+str(k):<14} {recall_vals[k]:>10.2f}  {ref_s:>14}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
NOTES_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "dataset": key,
    "hf_name": cfg["hf_name"],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "model": MODEL_PATH,
        "eval_ks": EVAL_KS,
        "n_corpus": len(corpus_ids_list),
        "n_queries_total": len(query_ids),
        "n_queries_evaluated": evaluated,
        "n_queries_skipped": skipped,
        "embedding_dim": D,
        "encode_corpus_s": round(encode_corpus_s, 1),
        "encode_query_s": round(encode_query_s, 1),
        "search_s": round(search_s, 4),
    },
    "results": {
        "mrr_at_10": round(mrr10, 4),
        **{f"recall_at_{k}": round(recall_vals[k], 4) for k in EVAL_KS},
    },
    "paper_reference_ood": {
        "model": "VisRAG-Ret (MiniCPM-V 2.0)",
        "setting": "out-of-domain",
        "source": "VisRAG ICLR 2025 Table 3",
        "mrr_at_10": cfg["paper_mrr10"],
        "recall_at_10": cfg["paper_recall10"],
    },
}
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n结果已保存：{out_path}")

# ── 清理缓存 ──────────────────────────────────────────────────────────────────
if clean_cache and CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)
    print(f"缓存已清理：{CACHE_DIR}")
elif not clean_cache:
    print(f"缓存保留：{CACHE_DIR}")

print(f"\n完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

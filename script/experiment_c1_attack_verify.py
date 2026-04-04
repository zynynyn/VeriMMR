"""
实验 C1（攻击验证版）：B1 + B2 + B3 三种攻击 + Sumcheck/ZAC 检测
用法：
  python script/experiment_c1_attack_verify.py --dataset <key> [--no-clean]
  key: slidevqa | docvqa | chartvqa | infovqa

攻击方案（与 B 组定义完全一致，在公开多模态数据集上重现）：
  B1 图像替换攻击  → Phase 1 ZAC 检测
      攻击：将某语料图像内容替换为另一张图像，embedding 不变
      检测：SHA256(new_bytes ∥ old_emb) ∉ BF → ZAC 拒绝

  B2 Embedding 替换攻击 → Phase 1 ZAC 跨层绑定检测
      攻击：保持图像文件不变，将 FAISS 中 embedding 替换为随机单位向量
      检测：SHA256(old_bytes ∥ fake_emb) ∉ BF → ZAC 拒绝

  B3 排名操控攻击  → Phase 2 Sumcheck 检测 + Recall@K 质量影响
      攻击：将所有相关语料项的 FAISS embedding 替换为随机向量
      质量影响：无验证时 Recall@K 大幅下降（静默降级）
      检测：Sumcheck 用承诺向量独立验证，top-k 与 FAISS 不一致 → 报警

报告格式：
  - 基线 Recall@K（干净语料）
  - B3 攻击后 Recall@K（无验证情况下用户实际得到的结果）
  - B1/B2/B3 各自的攻击检出率

注：验证机制检出攻击后系统拒绝响应，而非"修复" Recall。
"""

import argparse, copy, hashlib, io, json, shutil, sys, tempfile, time
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zac.accumulator import ZACAccumulator
from sumcheck.inner_product import prove_global_batch, verify_global_batch

# ── 数据集配置 ────────────────────────────────────────────────────────────────
DATASETS = {
    "slidevqa": {
        "hf_name":        "openbmb/VisRAG-Ret-Test-SlideVQA",
        "paper_mrr10":    45.57,
        "paper_recall10": 67.70,
    },
    "docvqa": {
        "hf_name":        "openbmb/VisRAG-Ret-Test-MP-DocVQA",
        "paper_mrr10":    74.60,
        "paper_recall10": 89.65,
    },
    "chartvqa": {
        "hf_name":        "openbmb/VisRAG-Ret-Test-ChartQA",
        "paper_mrr10":    75.99,
        "paper_recall10": 91.40,
    },
    "infovqa": {
        "hf_name":        "openbmb/VisRAG-Ret-Test-InfoVQA",
        "paper_mrr10":    67.26,
        "paper_recall10": 87.05,
    },
}

MODEL_PATH      = "/root/autodl-tmp/models/jina-embeddings-v4"
CACHE_DIR       = ROOT / "data" / "huggingface_cache"
NOTES_DIR       = ROOT / "notes"

EVAL_KS         = [1, 3, 5, 10]
TOP_K_SEARCH    = 10
BATCH_IMG       = 4
BATCH_TXT       = 32
N_ZAC_SAMPLE    = 50    # 用于 ZAC 构建的图像数（B1/B2 演示）
N_ZAC_ATTACK    = 10    # B1/B2 各自的攻击样本数
N_SUMCHECK_Q    = 50    # B3 Sumcheck 验证的 query 数（抽样）

# ── 参数解析 ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
parser.add_argument("--no-clean", action="store_true")
args       = parser.parse_args()
key        = args.dataset
cfg        = DATASETS[key]
clean_cache = not args.no_clean
out_path   = NOTES_DIR / f"experiment_c1_attack_{key}.json"

def ts():
    return time.strftime("%H:%M:%S")

print(f"\n{'='*64}")
print(f"实验 C1（攻击验证版）：{key}  B1+B2+B3")
print(f"{'='*64}")
print(f"  数据集  : {cfg['hf_name']}")
print(f"  ZAC 样本: {N_ZAC_SAMPLE} 张（B1/B2 各 {N_ZAC_ATTACK} 个攻击样本）")
print(f"  Sumcheck: {N_SUMCHECK_Q} queries（B3）")
print(f"  开始    : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载 jina-v4 ...")
import torch, faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

t0 = time.time()
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cuda:0")
model.eval()
print(f"  loaded  ({time.time()-t0:.1f}s)")

# ── 加载数据集 ────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载数据集 ...")
t0 = time.time()
ds_corpus  = load_dataset(cfg["hf_name"], "corpus",  split="train", cache_dir=str(CACHE_DIR))
ds_queries = load_dataset(cfg["hf_name"], "queries", split="train", cache_dir=str(CACHE_DIR))
ds_qrels   = load_dataset(cfg["hf_name"], "qrels",   split="train", cache_dir=str(CACHE_DIR))
print(f"  corpus={len(ds_corpus)}  queries={len(ds_queries)}  qrels={len(ds_qrels)}  ({time.time()-t0:.1f}s)")
print(f"  corpus 字段: {ds_corpus.column_names}")
print(f"  qrels  字段: {ds_qrels.column_names}")

# ── 字段自适应 ────────────────────────────────────────────────────────────────
corpus_id_f = next(f for f in ds_corpus.column_names  if f in ("_id","id","doc_id","corpus_id","corpus-id"))
query_id_f  = next(f for f in ds_queries.column_names if f in ("_id","id","query_id","query-id"))
query_txt_f = next(f for f in ds_queries.column_names if f in ("text","query","question"))
img_f       = next(f for f in ds_corpus.column_names  if f in ("image","img","pixel_values","image_bytes"))
qid_f       = next(f for f in ds_qrels.column_names   if "query"  in f.lower())
cid_f       = next(f for f in ds_qrels.column_names   if "corpus" in f.lower())
score_f     = next((f for f in ds_qrels.column_names  if "score"  in f.lower()), None)
print(f"  corpus_id={corpus_id_f}  img={img_f}  query_text={query_txt_f}")

# ── qrels ─────────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 构建 qrels ...")
qrels_dict: dict[str, set] = {}
for row in ds_qrels:
    qid = str(row[qid_f]); cid = str(row[cid_f])
    rel = int(row[score_f]) if score_f else 1
    if rel > 0:
        qrels_dict.setdefault(qid, set()).add(cid)
print(f"  有标注查询：{len(qrels_dict)} 条")

# ── 编码语料图像（承诺向量，全量）────────────────────────────────────────────
print(f"\n[{ts()}] 编码语料图像（{len(ds_corpus)} 张）...")
t_enc = time.time()
corpus_ids   = []
corpus_embs  = []
corpus_pils  = []          # 保存 PIL Image（供 ZAC 使用）
_bids, _bimgs = [], []

def _to_pil(img_data) -> Image.Image:
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    raise TypeError(f"未知图像类型：{type(img_data)}")

def _flush():
    if not _bimgs:
        return
    with torch.no_grad():
        emb = model.encode(_bimgs, task="retrieval", normalize_embeddings=True,
                           batch_size=len(_bimgs), show_progress_bar=False)
    corpus_embs.append(emb)
    corpus_ids.extend(_bids)
    _bids.clear(); _bimgs.clear()

for i, row in enumerate(ds_corpus):
    pil = _to_pil(row[img_f])
    corpus_pils.append(pil)
    _bids.append(str(row[corpus_id_f]))
    _bimgs.append(pil)
    if len(_bimgs) >= BATCH_IMG:
        _flush()
    if (i+1) % 100 == 0 or (i+1) == len(ds_corpus):
        elapsed = time.time() - t_enc
        speed = (i+1)/elapsed
        eta = (len(ds_corpus)-i-1)/speed
        print(f"  [{ts()}] {i+1}/{len(ds_corpus)}  {speed:.2f} img/s  ETA {eta/60:.1f}min")

_flush()
committed_matrix = np.vstack(corpus_embs).astype(np.float32)
D = committed_matrix.shape[1]
N = len(corpus_ids)
cid_to_int = {cid: i for i, cid in enumerate(corpus_ids)}
int_to_cid = dict(enumerate(corpus_ids))
print(f"  承诺向量 shape={committed_matrix.shape}  ({time.time()-t_enc:.1f}s)")

# ── 编码查询 ──────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 编码查询（{len(ds_queries)} 条）...")
t0 = time.time()
query_ids   = [str(row[query_id_f])  for row in ds_queries]
query_texts = [str(row[query_txt_f]) for row in ds_queries]
with torch.no_grad():
    query_matrix = model.encode(
        query_texts, task="retrieval", prompt_name="query",
        normalize_embeddings=True, batch_size=BATCH_TXT, show_progress_bar=False,
    ).astype(np.float32)
print(f"  shape={query_matrix.shape}  ({time.time()-t0:.1f}s)")

# ── 工具函数：Recall@K 计算 ───────────────────────────────────────────────────
def compute_recall(search_ids, eval_ks=EVAL_KS):
    recalls = {k: [] for k in eval_ks}
    rr_list = []
    ev = sk = 0
    for qi, qid in enumerate(query_ids):
        if qid not in qrels_dict:
            sk += 1; continue
        relevant = qrels_dict[qid]
        top = [int_to_cid[iid] for iid in search_ids[qi] if iid >= 0]
        rr = next((1.0/(r+1) for r, cid in enumerate(top[:10]) if cid in relevant), 0.0)
        rr_list.append(rr)
        for k in eval_ks:
            recalls[k].append(float(bool(set(top[:k]) & relevant)))
        ev += 1
    mrr = float(np.mean(rr_list)*100) if rr_list else 0.0
    rec = {k: float(np.mean(recalls[k])*100) for k in eval_ks}
    return mrr, rec, ev, sk

# ── 工具函数：构建 FAISS ──────────────────────────────────────────────────────
def build_faiss(matrix: np.ndarray):
    idx = faiss.IndexFlatIP(D)
    idx = faiss.IndexIDMap2(idx)
    idx.add_with_ids(matrix, np.arange(len(matrix), dtype=np.int64))
    return idx

# ════════════════════════════════════════════════════════════════════════════
# 基线：干净 FAISS，测 Recall@K
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] 基线：干净 FAISS 搜索 ...")
clean_idx = build_faiss(committed_matrix)
_, ids_clean = clean_idx.search(query_matrix, TOP_K_SEARCH)
mrr_clean, rec_clean, ev, sk = compute_recall(ids_clean)
print(f"  基线 MRR@10={mrr_clean:.2f}  R@1={rec_clean[1]:.2f}  "
      f"R@5={rec_clean[5]:.2f}  R@10={rec_clean[10]:.2f}  "
      f"（评测 {ev} 条，跳过 {sk} 条）")

# ════════════════════════════════════════════════════════════════════════════
# B1 + B2：构建 ZAC，在样本图像上演示攻击检测
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] 构建 ZAC（{N_ZAC_SAMPLE} 张样本图像）...")

rng = np.random.default_rng(42)
zac_indices = list(range(min(N_ZAC_SAMPLE, N)))  # 取前 N_ZAC_SAMPLE 张

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    # 将 PIL 图像保存为 JPEG 临时文件
    tmp_paths = []
    for i in zac_indices:
        p = tmp / f"img_{i:04d}.jpg"
        corpus_pils[i].save(str(p), format="JPEG", quality=95)
        tmp_paths.append(str(p))

    # 构建 ZAC 承诺集合 S
    zac_embs = committed_matrix[zac_indices]
    S = set()
    for path, emb in zip(tmp_paths, zac_embs):
        S.add(ZACAccumulator.image_embedding_hash(path, emb))

    t0 = time.time()
    acc = ZACAccumulator(S=S)
    print(f"  ZAC Root: {acc.root_hex()[:32]}…  ({time.time()-t0:.1f}s)")

    # ── B1：图像替换攻击 ──────────────────────────────────────────────────────
    print(f"\n[{ts()}] B1 图像替换攻击（{N_ZAC_ATTACK} 个样本）...")
    print("  攻击：将目标图像文件替换为另一张图像内容，embedding 不变")
    b1_detected = 0
    t_b1 = time.time()

    for i in range(min(N_ZAC_ATTACK, len(zac_indices))):
        vic_local = i
        donor_local = (i + len(zac_indices)//2) % len(zac_indices)
        if donor_local == vic_local:
            donor_local = (vic_local + 1) % len(zac_indices)

        vic_path   = tmp_paths[vic_local]
        donor_path = tmp_paths[donor_local]
        vic_emb    = zac_embs[vic_local]

        elem_orig  = ZACAccumulator.image_embedding_hash(vic_path, vic_emb)
        ok_before  = acc.verify_membership_batch([elem_orig],
                                                  acc.prove_membership_batch([elem_orig]))

        # 攻击：用 donor 图像覆盖 victim 文件
        bak = str(vic_path) + ".b1bak"
        shutil.copy2(vic_path, bak)
        try:
            shutil.copy2(donor_path, vic_path)
            elem_atk  = ZACAccumulator.image_embedding_hash(vic_path, vic_emb)
            ok_after  = acc.verify_membership_batch([elem_atk],
                                                     acc.prove_membership_batch([elem_atk]))
        finally:
            shutil.copy2(bak, vic_path)
            Path(bak).unlink()

        detected = (ok_before is True) and (ok_after is False)
        if detected:
            b1_detected += 1
        print(f"  [{i+1:02d}/{N_ZAC_ATTACK}] vic={zac_indices[vic_local]:3d} "
              f"donor={zac_indices[donor_local]:3d}  "
              f"before={ok_before} after={ok_after}  {'✓' if detected else '✗'}")

    b1_rate = b1_detected / N_ZAC_ATTACK
    print(f"  B1 检出率：{b1_detected}/{N_ZAC_ATTACK} = {b1_rate*100:.1f}%  "
          f"耗时 {time.time()-t_b1:.1f}s")

    # ── B2：Embedding 替换攻击 ─────────────────────────────────────────────────
    print(f"\n[{ts()}] B2 Embedding 替换攻击（{N_ZAC_ATTACK} 个样本）...")
    print("  攻击：图像文件不变，FAISS embedding 替换为随机单位向量")
    b2_detected = 0
    t_b2 = time.time()

    attack_indices_b2 = rng.choice(len(zac_indices), size=N_ZAC_ATTACK, replace=False)
    for i, ai in enumerate(attack_indices_b2):
        vic_path = tmp_paths[ai]
        vic_emb  = zac_embs[ai]

        elem_orig = ZACAccumulator.image_embedding_hash(vic_path, vic_emb)
        ok_before = acc.verify_membership_batch([elem_orig],
                                                acc.prove_membership_batch([elem_orig]))

        # 攻击：替换 embedding 为随机单位向量
        fake_emb  = rng.standard_normal(D).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        elem_atk  = ZACAccumulator.image_embedding_hash(vic_path, fake_emb)
        ok_after  = acc.verify_membership_batch([elem_atk],
                                                acc.prove_membership_batch([elem_atk]))

        detected = (ok_before is True) and (ok_after is False)
        if detected:
            b2_detected += 1
        print(f"  [{i+1:02d}/{N_ZAC_ATTACK}] tgt={zac_indices[ai]:3d}  "
              f"before={ok_before} after={ok_after}  {'✓' if detected else '✗'}")

    b2_rate = b2_detected / N_ZAC_ATTACK
    print(f"  B2 检出率：{b2_detected}/{N_ZAC_ATTACK} = {b2_rate*100:.1f}%  "
          f"耗时 {time.time()-t_b2:.1f}s")

# tmpdir 在此已自动清理

# ════════════════════════════════════════════════════════════════════════════
# B3：FAISS Embedding 替换 → Recall@K 质量影响 + Sumcheck 检测
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[{ts()}] B3 FAISS Embedding 替换攻击（相关语料项）...")
print("  攻击：将所有相关语料项的 FAISS embedding 替换为随机单位向量")

# 收集所有 qrels 相关语料项
all_relevant_cids = set()
for rel_set in qrels_dict.values():
    all_relevant_cids |= rel_set

tampered_matrix = committed_matrix.copy()
attacked_count  = 0
for cid in all_relevant_cids:
    if cid not in cid_to_int:
        continue
    idx_i = cid_to_int[cid]
    rand_vec = rng.standard_normal(D).astype(np.float32)
    rand_vec /= np.linalg.norm(rand_vec)
    tampered_matrix[idx_i] = rand_vec
    attacked_count += 1

print(f"  替换了 {attacked_count} 个相关语料项的 embedding"
      f"（corpus 占比 {attacked_count/N*100:.1f}%）")

# 无验证场景：用篡改 FAISS 搜索
tampered_idx = build_faiss(tampered_matrix)
_, ids_atk = tampered_idx.search(query_matrix, TOP_K_SEARCH)
mrr_atk, rec_atk, _, _ = compute_recall(ids_atk)
print(f"  攻击后（无验证）MRR@10={mrr_atk:.2f}  "
      f"R@1={rec_atk[1]:.2f}  R@5={rec_atk[5]:.2f}  R@10={rec_atk[10]:.2f}")
print(f"  Recall@10 下降：{rec_clean[10]-rec_atk[10]:.2f}pp  "
      f"MRR@10 下降：{mrr_clean-mrr_atk:.2f}pp")

# Sumcheck 验证：抽样 N_SUMCHECK_Q 条有标注 query
print(f"\n[{ts()}] Sumcheck 验证（{N_SUMCHECK_Q} 条 query，承诺向量）...")
attacked_qpairs = [(qi, qid) for qi, qid in enumerate(query_ids)
                   if qid in qrels_dict][:N_SUMCHECK_Q]
committed_vecs  = committed_matrix.tolist()

b3_detected = 0
t_b3 = time.time()
for qi, qid in attacked_qpairs:
    q_vec     = query_matrix[qi].tolist()
    atk_topk  = set(int_to_cid[iid] for iid in ids_atk[qi] if iid >= 0)

    proof  = prove_global_batch(q_vec, committed_vecs)
    result = verify_global_batch(q_vec, committed_vecs, proof, TOP_K_SEARCH)

    if not result["verified"]:
        b3_detected += 1
        continue

    verified_topk = set(int_to_cid[idx] for idx in result["top_k_indices"])
    if atk_topk != verified_topk:
        b3_detected += 1

b3_total  = len(attacked_qpairs)
b3_rate   = b3_detected / b3_total if b3_total > 0 else 0.0
verify_s  = time.time() - t_b3
print(f"  B3 检出率：{b3_detected}/{b3_total} = {b3_rate*100:.1f}%  "
      f"耗时 {verify_s:.1f}s  ({verify_s/b3_total*1000:.0f}ms/query)")

# ════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print(f"C1 攻击验证汇总：{key}")
print(f"{'='*64}")
print(f"\n  检索质量（B3 攻击前后对比）")
print(f"  {'指标':<14} {'基线（干净）':>12}  {'攻击后（无验证）':>16}")
print(f"  {'─'*46}")
print(f"  {'MRR@10':<14} {mrr_clean:>12.2f}  {mrr_atk:>16.2f}")
for k in EVAL_KS:
    print(f"  {'Recall@'+str(k):<14} {rec_clean[k]:>12.2f}  {rec_atk[k]:>16.2f}")

print(f"\n  攻击检出率")
print(f"  {'─'*46}")
print(f"  B1 图像替换（ZAC）              {b1_detected}/{N_ZAC_ATTACK}  ({b1_rate*100:.1f}%)")
print(f"  B2 Embedding 替换（ZAC 绑定）   {b2_detected}/{N_ZAC_ATTACK}  ({b2_rate*100:.1f}%)")
print(f"  B3 排名操控（Sumcheck）          {b3_detected}/{b3_total}  ({b3_rate*100:.1f}%)")
print(f"\n  VisRAG-Ret OOD 参照：MRR@10={cfg['paper_mrr10']}  Recall@10={cfg['paper_recall10']}")

# ── 保存结果 ──────────────────────────────────────────────────────────────────
NOTES_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "dataset": key,
    "hf_name": cfg["hf_name"],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "model": MODEL_PATH,
        "eval_ks": EVAL_KS,
        "n_corpus": N,
        "n_queries_evaluated": ev,
        "n_attacked_corpus_items": attacked_count,
        "attacked_ratio": round(attacked_count / N, 4),
        "n_zac_sample": N_ZAC_SAMPLE,
        "n_zac_attack": N_ZAC_ATTACK,
        "n_sumcheck_queries": b3_total,
        "embedding_dim": D,
    },
    "baseline_recall": {
        "mrr_at_10": round(mrr_clean, 4),
        **{f"recall_at_{k}": round(rec_clean[k], 4) for k in EVAL_KS},
    },
    "attacked_recall_no_verify": {
        "mrr_at_10": round(mrr_atk, 4),
        **{f"recall_at_{k}": round(rec_atk[k], 4) for k in EVAL_KS},
        "mrr_drop": round(mrr_clean - mrr_atk, 4),
        "recall10_drop": round(rec_clean[10] - rec_atk[10], 4),
    },
    "detection_rates": {
        "B1_image_replace_zac": {
            "detected": b1_detected, "total": N_ZAC_ATTACK,
            "rate_pct": round(b1_rate * 100, 2),
        },
        "B2_embedding_replace_zac": {
            "detected": b2_detected, "total": N_ZAC_ATTACK,
            "rate_pct": round(b2_rate * 100, 2),
        },
        "B3_ranking_forge_sumcheck": {
            "detected": b3_detected, "total": b3_total,
            "rate_pct": round(b3_rate * 100, 2),
            "verify_ms_per_query": round(verify_s / b3_total * 1000, 1),
        },
    },
    "paper_reference_ood": {
        "model": "VisRAG-Ret (MiniCPM-V 2.0, out-of-domain)",
        "source": "VisRAG ICLR 2025 Table 3",
        "mrr_at_10": cfg["paper_mrr10"],
        "recall_at_10": cfg["paper_recall10"],
    },
}
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n结果已保存：{out_path}")

if clean_cache and CACHE_DIR.exists():
    shutil.rmtree(CACHE_DIR)
    print(f"缓存已清理：{CACHE_DIR}")

print(f"完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

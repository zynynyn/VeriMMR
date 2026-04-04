"""
实验 C1 补充：Bloom Filter 误报率（False Positive Rate）统计验证

目标：用大样本实验验证 ZAC 内置 BF 的实际误报率是否符合理论值 ε=0.01。

方法：
  ZAC 集合 S：corpus 前 N_ZAC 张图像（50 张）
  B1-style 测试：对 S 之外的图像，测试 SHA256(donor_bytes ∥ member_emb) 是否被误判为成员
                 → 模拟图像替换攻击中 ZAC 未能检出的情形
  B2-style 测试：对 S 内图像，测试 SHA256(member_bytes ∥ fake_emb) 是否被误判为成员
                 → 模拟 Embedding 替换攻击中 ZAC 未能检出的情形

  每种测试各 N_TRIALS 次，共 2×N_TRIALS 次独立试验。

预期结果：
  实际误报率 ≈ ε=0.01（95% Wilson CI 应包含 0.01）
  零假阴性：S 中所有合法成员均通过验证

数据集：infovqa（corpus=459，ZAC 用前50张，剩余409张作非成员测试）
"""

import io, json, sys, time, tempfile
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from zac.accumulator import ZACAccumulator

# ── 配置 ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/root/autodl-tmp/models/jina-embeddings-v4"
CACHE_DIR    = ROOT / "data" / "huggingface_cache"
NOTES_DIR    = ROOT / "notes"

HF_NAME      = "openbmb/VisRAG-Ret-Test-InfoVQA"
N_ZAC        = 50       # ZAC 集合大小
N_TRIALS     = 400      # B1 / B2 各自的试验次数
SEED         = 42

def ts():
    return time.strftime("%H:%M:%S")

print(f"\n{'='*64}")
print(f"实验 C1 补充：Bloom Filter 误报率统计验证")
print(f"{'='*64}")
print(f"  数据集   : {HF_NAME}")
print(f"  ZAC 大小 : {N_ZAC}")
print(f"  试验次数 : {N_TRIALS} × 2 = {N_TRIALS*2} 次")
print(f"  理论 ε   : 0.01")
print(f"  开始     : {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── 加载模型 ─────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载 jina-v4 ...")
import torch
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cuda:0")
model.eval()
print(f"  loaded  ({time.time()-t0:.1f}s)")

# ── 加载数据集 ────────────────────────────────────────────────────────────────
print(f"\n[{ts()}] 加载数据集 ...")
from datasets import load_dataset
t0 = time.time()
ds_corpus = load_dataset(HF_NAME, "corpus", split="train", cache_dir=str(CACHE_DIR))
N_corpus  = len(ds_corpus)
print(f"  corpus={N_corpus}  ({time.time()-t0:.1f}s)")
assert N_corpus >= N_ZAC + N_TRIALS, \
    f"语料库不足：需要 {N_ZAC + N_TRIALS} 张，实际 {N_corpus} 张"

img_f = next(f for f in ds_corpus.column_names if f in ("image","img","pixel_values","image_bytes"))

def to_pil(img_data) -> Image.Image:
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    raise TypeError(f"未知图像类型：{type(img_data)}")

# ── 编码全部 N_ZAC + N_TRIALS 张图像 ─────────────────────────────────────────
N_encode = N_ZAC + N_TRIALS
print(f"\n[{ts()}] 编码前 {N_encode} 张图像 ...")
t_enc = time.time()
pils, embs = [], []
batch_imgs, BATCH = [], 4

def flush():
    if not batch_imgs: return
    with torch.no_grad():
        e = model.encode(batch_imgs, task="retrieval", normalize_embeddings=True,
                         batch_size=len(batch_imgs), show_progress_bar=False)
    embs.append(e)
    batch_imgs.clear()

for i in range(N_encode):
    pil = to_pil(ds_corpus[i][img_f])
    pils.append(pil)
    batch_imgs.append(pil)
    if len(batch_imgs) >= BATCH:
        flush()
    if (i+1) % 100 == 0 or (i+1) == N_encode:
        elapsed = time.time() - t_enc
        print(f"  [{ts()}] {i+1}/{N_encode}  {(i+1)/elapsed:.2f} img/s")
flush()
all_embs = np.vstack(embs).astype(np.float32)
print(f"  编码完成  shape={all_embs.shape}  ({time.time()-t_enc:.1f}s)")

# ── 构建 ZAC（前 N_ZAC 张）────────────────────────────────────────────────────
print(f"\n[{ts()}] 构建 ZAC（{N_ZAC} 张）...")
rng = np.random.default_rng(SEED)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    zac_paths = []
    for i in range(N_ZAC):
        p = tmp / f"zac_{i:04d}.jpg"
        pils[i].save(str(p), format="JPEG", quality=95)
        zac_paths.append(str(p))

    # 构建集合 S
    S = set()
    for path, emb in zip(zac_paths, all_embs[:N_ZAC]):
        S.add(ZACAccumulator.image_embedding_hash(path, emb))

    t0 = time.time()
    acc = ZACAccumulator(S)
    print(f"  ZAC Root: {acc.root_hex()[:32]}…  ({time.time()-t0:.1f}s)")

    # ── 零假阴性验证：所有合法成员必须通过 ────────────────────────────────────
    print(f"\n[{ts()}] 零假阴性验证（合法成员全量检查）...")
    fn_count = 0
    for path, emb in zip(zac_paths, all_embs[:N_ZAC]):
        elem = ZACAccumulator.image_embedding_hash(path, emb)
        ok   = acc.verify_membership_batch([elem], acc.prove_membership_batch([elem]))
        if not ok:
            fn_count += 1
    print(f"  假阴性（漏报）：{fn_count}/{N_ZAC}  {'✅ 零漏报' if fn_count==0 else '❌ 存在漏报'}")

    # ── B1-style：非成员图像 + 成员 embedding ────────────────────────────────
    print(f"\n[{ts()}] B1-style 误报测试（{N_TRIALS} 次）...")
    print(f"  测试元素：SHA256(donor_bytes[N_ZAC..N_ZAC+N_TRIALS] ∥ emb[0])")
    t_b1 = time.time()
    fp_b1 = 0
    # 为非成员图像保存临时文件
    nonmember_paths = []
    for i in range(N_ZAC, N_ZAC + N_TRIALS):
        p = tmp / f"nm_{i:04d}.jpg"
        pils[i].save(str(p), format="JPEG", quality=95)
        nonmember_paths.append(str(p))

    # 固定使用 emb[0]（ZAC 成员的 embedding），测试非成员图像字节
    ref_emb = all_embs[0]
    for ti, nm_path in enumerate(nonmember_paths):
        elem = ZACAccumulator.image_embedding_hash(nm_path, ref_emb)
        ok   = acc.verify_membership_batch([elem], acc.prove_membership_batch([elem]))
        if ok:
            fp_b1 += 1
        if (ti+1) % 50 == 0 or (ti+1) == N_TRIALS:
            elapsed = time.time() - t_b1
            print(f"  [{ts()}] {ti+1}/{N_TRIALS}  误报={fp_b1}  ({elapsed:.0f}s)")

    fpr_b1 = fp_b1 / N_TRIALS
    print(f"  B1-style 误报率：{fp_b1}/{N_TRIALS} = {fpr_b1*100:.2f}%")

    # ── B2-style：成员图像 + 随机 embedding ──────────────────────────────────
    print(f"\n[{ts()}] B2-style 误报测试（{N_TRIALS} 次）...")
    print(f"  测试元素：SHA256(member_bytes[0] ∥ random_emb_k)")
    t_b2  = time.time()
    fp_b2 = 0
    ref_path = zac_paths[0]
    D = all_embs.shape[1]

    for ti in range(N_TRIALS):
        fake_emb = rng.standard_normal(D).astype(np.float32)
        fake_emb /= np.linalg.norm(fake_emb)
        elem = ZACAccumulator.image_embedding_hash(ref_path, fake_emb)
        ok   = acc.verify_membership_batch([elem], acc.prove_membership_batch([elem]))
        if ok:
            fp_b2 += 1
        if (ti+1) % 50 == 0 or (ti+1) == N_TRIALS:
            elapsed = time.time() - t_b2
            print(f"  [{ts()}] {ti+1}/{N_TRIALS}  误报={fp_b2}  ({elapsed:.0f}s)")

    fpr_b2 = fp_b2 / N_TRIALS
    print(f"  B2-style 误报率：{fp_b2}/{N_TRIALS} = {fpr_b2*100:.2f}%")

# ── 汇总 ──────────────────────────────────────────────────────────────────────
total_fp  = fp_b1 + fp_b2
total_n   = N_TRIALS * 2
fpr_total = total_fp / total_n

# 95% Wilson 置信区间
z = 1.96
n, p = total_n, fpr_total
denom = 1 + z**2/n
center = (p + z**2/(2*n)) / denom
margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
ci_lo  = max(0.0, center - margin)
ci_hi  = center + margin

print(f"\n{'='*64}")
print(f"Bloom Filter 误报率统计验证结果")
print(f"{'='*64}")
print(f"  合法成员假阴性（漏报）  : {fn_count}/{N_ZAC}  ({'✅ 零漏报' if fn_count==0 else '❌'})")
print(f"  B1-style 误报           : {fp_b1}/{N_TRIALS}  ({fpr_b1*100:.2f}%)")
print(f"  B2-style 误报           : {fp_b2}/{N_TRIALS}  ({fpr_b2*100:.2f}%)")
print(f"  合计误报                : {total_fp}/{total_n}  ({fpr_total*100:.2f}%)")
print(f"  95% Wilson CI           : [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")
print(f"  理论 ε=0.01 {'✅ 在 CI 内' if ci_lo <= 0.01 <= ci_hi else '❌ 不在 CI 内'}")

# ── 保存 ──────────────────────────────────────────────────────────────────────
result = {
    "dataset": "infovqa",
    "hf_name": HF_NAME,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "n_zac": N_ZAC,
        "n_trials_per_type": N_TRIALS,
        "n_trials_total": total_n,
        "theoretical_epsilon": 0.01,
        "seed": SEED,
    },
    "results": {
        "false_negatives": fn_count,
        "fp_b1_style": fp_b1,
        "fp_b2_style": fp_b2,
        "fpr_b1": round(fpr_b1, 6),
        "fpr_b2": round(fpr_b2, 6),
        "fpr_total": round(fpr_total, 6),
        "ci_95_lo": round(ci_lo, 6),
        "ci_95_hi": round(ci_hi, 6),
        "theoretical_epsilon_in_ci": bool(ci_lo <= 0.01 <= ci_hi),
    },
}
out_path = NOTES_DIR / "experiment_c1_bf_fpr.json"
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
print(f"\n结果已保存：{out_path}")
print(f"完成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

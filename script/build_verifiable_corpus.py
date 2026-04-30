"""
可验证语料库一键构建脚本

两种模式：
  全量构建（默认）：从零建库
  增量添加（--incremental）：仅处理新 PDF，不重算已有部分

增量模式各步骤：
  corpus    → append-only（仅新图像）
  embedding → 仅算新图像，concat 到现有 .npy
  FAISS     → index.add_with_ids()，不重建
  ZAC       → 必须全量重建（集合 S 扩展了）
  zkLLM     → 仅新图像（默认跳过已有证明文件）

用法：
  # 全量建库
  python script/build_verifiable_corpus.py --pdf data/nikon.pdf

  # 增量添加新 PDF
  python script/build_verifiable_corpus.py --pdf data/new_doc.pdf --incremental
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_MODEL = "/root/autodl-tmp/models/jina-embeddings-v4"


def run(cmd: list, desc: str, check=True) -> int:
    print(f"\n{'='*60}")
    print(f"[步骤] {desc}")
    print(f"  命令: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"[错误] {desc} 失败（returncode={result.returncode}）", file=sys.stderr)
        sys.exit(result.returncode)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# 增量模式
# ─────────────────────────────────────────────────────────────────────────────
def run_incremental(args):
    corpus_path  = ROOT / args.corpus
    emb_path     = ROOT / args.embedding
    index_path   = ROOT / args.index
    pdf_path     = ROOT / args.pdf

    print(f"\n[增量] 目标 PDF : {pdf_path}")
    print(f"[增量] 当前语料 : {corpus_path}")

    # ── 1. 读取已有 corpus，收集 image_id 集合 ────────────────────────────────
    existing_ids:    set  = set()
    existing_entries: list = []
    if corpus_path.exists():
        with open(corpus_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    existing_ids.add(item.get("image_id", ""))
                    existing_entries.append(item)
    n_existing = len(existing_entries)
    print(f"[增量] 已有语料: {n_existing} 条")

    # ── 2. 转换新 PDF → 图像（仅尚未存在的页） ────────────────────────────────
    try:
        import fitz
    except ImportError:
        print("[错误] 需要 pymupdf：pip install pymupdf", file=sys.stderr)
        sys.exit(1)

    pdf_stem  = pdf_path.stem
    image_dir = corpus_path.parent / "image" / pdf_stem
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[增量] 转换 PDF → 图像...")
    doc = fitz.open(str(pdf_path))
    new_entries = []
    for page_idx in range(len(doc)):
        img_name  = f"page_{page_idx}.jpg"
        image_id  = f"{pdf_stem}/{img_name}"
        if image_id in existing_ids:
            continue
        img_path = image_dir / img_name
        if not img_path.exists():
            pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            pix.save(str(img_path))
        new_entries.append({
            "id":         n_existing + len(new_entries),
            "image_id":   image_id,
            "image_path": f"image/{pdf_stem}/{img_name}",
        })
    doc.close()

    if not new_entries:
        print("[增量] 无新图像（已全部入库），退出。")
        return
    print(f"[增量] 新增图像: {len(new_entries)} 张")

    # ── 3. Append 新条目到 corpus jsonl ──────────────────────────────────────
    with open(corpus_path, "a", encoding="utf-8") as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[增量] Corpus 更新 → {corpus_path}（共 {n_existing + len(new_entries)} 条）")

    # ── 4. 计算新图像 embedding（仅新增部分） ────────────────────────────────
    print("[增量] 计算新图像 embedding（jina-v4）...")
    new_img_paths = [str(corpus_path.parent / e["image_path"]) for e in new_entries]
    temp_emb_path = ROOT / "embedding" / "_incremental_tmp.npy"

    embed_code = (
        "import numpy as np\n"
        "from PIL import Image\n"
        "from pathlib import Path\n"
        "from sentence_transformers import SentenceTransformer\n"
        f"model = SentenceTransformer({args.model_path!r}, trust_remote_code=True, device='cuda:0')\n"
        f"paths = {new_img_paths!r}\n"
        "imgs = [Image.open(p).convert('RGB') for p in paths if Path(p).exists()]\n"
        "embs = model.encode(imgs, task='retrieval', normalize_embeddings=False,\n"
        "                    batch_size=4, show_progress_bar=True)\n"
        f"np.save({str(temp_emb_path)!r}, embs.astype('float32'))\n"
        f"print(f'[embed] 保存 {{len(embs)}} 条 embedding')\n"
    )

    rc = subprocess.run([sys.executable, "-c", embed_code], cwd=str(ROOT)).returncode
    if rc != 0 or not temp_emb_path.exists():
        print("[错误] embedding 计算失败", file=sys.stderr)
        sys.exit(1)

    new_embs = np.load(str(temp_emb_path)).astype(np.float32)
    temp_emb_path.unlink()
    print(f"[增量] 新 embedding shape: {new_embs.shape}")

    # ── 5. Concat → embedding.npy ────────────────────────────────────────────
    if emb_path.exists():
        old_embs = np.load(str(emb_path)).astype(np.float32)
        all_embs = np.concatenate([old_embs, new_embs], axis=0)
    else:
        all_embs = new_embs
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), all_embs)
    print(f"[增量] Embedding 更新 → {emb_path}  shape={all_embs.shape}")

    # ── 6. FAISS：add_with_ids（不重建索引） ─────────────────────────────────
    try:
        import faiss
    except ImportError:
        print("[错误] 需要 faiss：pip install faiss-gpu", file=sys.stderr)
        sys.exit(1)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        start_id = n_existing
        new_ids  = np.arange(start_id, start_id + len(new_embs), dtype=np.int64)
        index.add_with_ids(new_embs, new_ids)
    else:
        inner = faiss.IndexFlatIP(new_embs.shape[1])
        index = faiss.IndexIDMap2(inner)
        index.add_with_ids(new_embs, np.arange(len(new_embs), dtype=np.int64))
    faiss.write_index(index, str(index_path))
    print(f"[增量] FAISS 索引更新 → {index_path}  ntotal={index.ntotal}")

    # ── 7. ZAC 全量重建（S 已扩展，必须重新承诺） ────────────────────────────
    if not args.skip_zac:
        zac_out = ROOT / args.zac_output
        zac_out.parent.mkdir(parents=True, exist_ok=True)
        run(["python", "script/phase1_corpus_fingerprint.py",
             "--zac-only",
             "--corpus-jsonl",  args.corpus,
             "--embedding-npy", args.embedding,
             "--output",        args.zac_output,
             "--n-filters",     str(args.n_filters)],
            "ZAC 承诺全量重建（集合 S 已扩展）")
        print(f"[ZAC] ⚠️  cm_hex 已更新！请重新通过可信渠道发布新的 ZAC Root！")
    else:
        print("\n[跳过] ZAC 重建（--skip-zac）——注意：旧 cm_hex 已失效！")

    # ── 8. zkLLM 仅新图像（默认跳过已有证明，无需 --overwrite） ───────────────
    if not args.skip_zkllm:
        _run_zkllm(args, new_only=True)

    # ── 完成 ─────────────────────────────────────────────────────────────────
    n_new = len(new_entries)
    n_total = n_existing + n_new
    print(f"\n{'='*60}")
    print(f"增量更新完成！新增 {n_new} 张图像，语料库共 {n_total} 条")
    print(f"{'='*60}")
    print(f"  查看 zkLLM 后台进度：tail -f /tmp/corpus_zkllm_proof.log")
    print(f"  ⚠️  ZAC Root 已变更，请发布新的 cm_hex")


# ─────────────────────────────────────────────────────────────────────────────
# zkLLM 后台预计算（全量 / 仅新图像）
# ─────────────────────────────────────────────────────────────────────────────
def _run_zkllm(args, new_only: bool = False):
    zkllm_dir  = ROOT / args.zkllm_workdir
    first_layer = 36 - args.k_layers
    probe = zkllm_dir / f"layer-{first_layer}-self_attn.q_proj.weight-commitment.bin"
    if not zkllm_dir.exists() or not probe.exists():
        print(f"\n[警告] zkLLM 权重未提交，跳过预计算（K={args.k_layers}）", file=sys.stderr)
        print(f"  请先运行 /root/autodl-tmp/zkllm-ccs2024/load_jina_weights.py")
        return

    overwrite_flag = ["--overwrite"] if (args.overwrite and not new_only) else []
    cmd = [
        sys.executable, "script/build_corpus_zkllm_proofs.py",
        "--corpus",   args.corpus,
        "--workdir",  args.zkllm_workdir,
        "--k_layers", str(args.k_layers),
    ] + overwrite_flag

    log_path = "/tmp/corpus_zkllm_proof.log"
    print(f"\n[zkLLM] 后台启动{'（仅新图像）' if new_only else '（全量）'} K={args.k_layers}")
    print(f"  日志：tail -f {log_path}")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"  PID: {proc.pid}（后台运行，退出后不中断）")


# ─────────────────────────────────────────────────────────────────────────────
# 全量模式
# ─────────────────────────────────────────────────────────────────────────────
def run_full(args):
    # ── 步骤 1：PDF → image corpus ────────────────────────────────────────────
    if args.skip_corpus:
        print("\n[跳过] 步骤1 PDF→corpus（--skip-corpus）")
    else:
        corpus_path = ROOT / args.corpus
        if corpus_path.exists() and not args.overwrite:
            print(f"\n[跳过] 步骤1 corpus 已存在（传 --overwrite 强制重建）")
        else:
            run(["ultrarag", "build", "examples/build_image_corpus.yaml"], "构建 image corpus（build）")
            run(["ultrarag", "run",   "examples/build_image_corpus.yaml"], "构建 image corpus（run）")

    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"[错误] corpus 不存在：{corpus_path}", file=sys.stderr)
        sys.exit(1)

    # ── 步骤 2+3：embedding + FAISS index ────────────────────────────────────
    if args.skip_embed:
        print("\n[跳过] 步骤2+3 embedding+index（--skip-embed）")
    else:
        emb_path = ROOT / args.embedding
        if emb_path.exists() and not args.overwrite:
            print(f"\n[跳过] 步骤2+3 embedding 已存在（传 --overwrite 强制重建）")
        else:
            run(["ultrarag", "build", "examples/corpus_index.yaml"], "构建 embedding+index（build）")
            run(["ultrarag", "run",   "examples/corpus_index.yaml"], "构建 embedding+index（run）")

    # ── 步骤 4：ZAC 语料库承诺 ───────────────────────────────────────────────
    if args.skip_zac:
        print("\n[跳过] 步骤4 ZAC 承诺（--skip-zac）")
    else:
        zac_out = ROOT / args.zac_output
        zac_out.parent.mkdir(parents=True, exist_ok=True)
        run(["python", "script/phase1_corpus_fingerprint.py",
             "--zac-only",
             "--corpus-jsonl",  args.corpus,
             "--embedding-npy", args.embedding,
             "--output",        args.zac_output,
             "--n-filters",     str(args.n_filters)],
            "生成 ZAC 语料库承诺（Phase 1）")
        print(f"\n[ZAC] ⚠️  请将 ZAC Root (cm_hex) 通过可信渠道发布")

    # ── 步骤 5：zkLLM corpus-side 证明（后台） ────────────────────────────────
    if args.skip_zkllm:
        print("\n[跳过] 步骤5 zkLLM 证明预计算（--skip-zkllm）")
    else:
        _run_zkllm(args, new_only=False)

    print(f"\n{'='*60}")
    print(f"构建完成！可验证检索系统就绪")
    print(f"{'='*60}")
    print(f"  查看 zkLLM 进度：tail -f /tmp/corpus_zkllm_proof.log")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="可验证语料库一键构建（全量/增量）")
    parser.add_argument("--pdf",           default="data/nikon.pdf")
    parser.add_argument("--corpus",        default="corpora/image.jsonl")
    parser.add_argument("--embedding",     default="embedding/embedding.npy")
    parser.add_argument("--index",         default="index/index.index")
    parser.add_argument("--zac-output",    default="output/phase1/fingerprint.json")
    parser.add_argument("--zkllm-workdir", default="zkllm-workdir/jina-v4")
    parser.add_argument("--model-path",    default=DEFAULT_MODEL,
                        help="jina-v4 模型路径（增量模式下用于计算新 embedding）")
    parser.add_argument("--k-layers",      type=int, default=36,
                        help="语料库侧证明层数（全量=36，即全部 LLM decoder 层）")
    parser.add_argument("--incremental",   action="store_true",
                        help="增量模式：仅处理新 PDF，不重算已有部分")
    parser.add_argument("--n-filters",     type=int, default=1,
                        help="ZAC 串联 BF 层数（复合 FPR = ε^n，默认 1）")
    parser.add_argument("--skip-corpus",   action="store_true")
    parser.add_argument("--skip-embed",    action="store_true")
    parser.add_argument("--skip-zac",      action="store_true")
    parser.add_argument("--skip-zkllm",    action="store_true")
    parser.add_argument("--overwrite",     action="store_true")
    args = parser.parse_args()

    print(f"\n可验证语料库{'增量更新' if args.incremental else '全量构建'}")
    print(f"  PDF        : {args.pdf}")
    print(f"  Corpus     : {args.corpus}")
    print(f"  Embedding  : {args.embedding}")
    print(f"  Index      : {args.index}")
    print(f"  ZAC out    : {args.zac_output}")
    print(f"  zkLLM dir  : {args.zkllm_workdir} (K={args.k_layers})")

    if args.incremental:
        run_incremental(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()

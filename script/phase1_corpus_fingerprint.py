"""
Phase 1 — Corpus Fingerprint Generator
=======================================
Automation script for Stage 1 of the verifiable retrieval system.

Pipeline:
  1. (Optional) Build image corpus from PDF via UltraRAG corpus server
  2. (Optional) Compute embeddings via UltraRAG retriever server
  3. Build ZAC accumulator over (image, vector) pairs
  4. Save 48-byte ZAC root + full Merkle manifest

Usage examples:

  # Full pipeline: PDF → corpus → embeddings → ZAC root
  python script/phase1_corpus_fingerprint.py \
    --pdf data/nikon.pdf \
    --corpus-dir output/phase1/corpora \
    --embedding-dir output/phase1/embedding \
    --output output/phase1/fingerprint.json \
    --model /root/autodl-tmp/models/jina-embeddings-v4 \
    --gpu 0

  # ZAC only (corpus + embeddings already exist)
  python script/phase1_corpus_fingerprint.py \
    --corpus-jsonl corpora/image.jsonl \
    --embedding-npy embedding/embedding.npy \
    --output output/phase1/fingerprint.json \
    --zac-only

  # Verify a membership proof
  python script/phase1_corpus_fingerprint.py \
    --verify output/phase1/fingerprint.json \
    --proof-index 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zac.accumulator import ZACAccumulator


# ---------------------------------------------------------------------------
# Sub-steps (call UltraRAG pipeline programmatically)
# ---------------------------------------------------------------------------

def _build_image_corpus(pdf_path: str, corpus_dir: str) -> str:
    """
    Build image corpus from PDF using pymupdf (mirrors corpus.build_image_corpus).
    Returns path to the generated image.jsonl.
    """
    try:
        import pymupdf
    except ImportError:
        sys.exit("ERROR: pymupdf not installed. Run: pip install pymupdf")

    from PIL import Image

    pdf_path = Path(pdf_path).resolve()
    corpus_dir = Path(corpus_dir).resolve()
    img_dir = corpus_dir / "image" / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = corpus_dir / "image.jsonl"

    doc = pymupdf.open(str(pdf_path))
    zoom = 144 / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    rows = []
    gid = 0
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=pymupdf.csRGB)
        filename = f"page_{i}.jpg"
        save_path = img_dir / filename
        pix.save(str(save_path), jpg_quality=90)
        pix = None

        # Verify
        try:
            with Image.open(save_path) as im:
                im.verify()
        except Exception as e:
            print(f"  [WARN] Skip page {i}: {e}")
            save_path.unlink(missing_ok=True)
            continue

        rel_path = Path("image") / pdf_path.stem / filename
        rows.append({
            "id": gid,
            "image_id": str(Path(pdf_path.stem) / filename),
            "image_path": rel_path.as_posix(),
        })
        gid += 1

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[corpus] Built {len(rows)} pages from {pdf_path.name} → {jsonl_path}")
    return str(jsonl_path)


def _compute_embeddings(
    corpus_jsonl: str,
    embedding_path: str,
    model_path: str,
    gpu_id: int = 0,
    batch_size: int = 8,
) -> None:
    """
    Compute multimodal embeddings using jina-embeddings-v4 via sentence_transformers.
    Saves to embedding_path (.npy).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")

    import numpy as np
    from PIL import Image

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[embed] Loading model from {model_path} ...")
    model = SentenceTransformer(
        model_name_or_path=model_path,
        device="cuda",
        trust_remote_code=True,
    )

    corpus_dir = Path(corpus_jsonl).parent
    image_paths = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rel = item.get("image_path", "")
            image_paths.append(str((corpus_dir / rel).resolve()))

    print(f"[embed] Encoding {len(image_paths)} images ...")
    images = []
    for p in image_paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())

    embeddings = model.encode(
        images,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        precision="float32",
        task="retrieval",
    )

    emb_path = Path(embedding_path)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    np.save(str(emb_path), embeddings_arr)
    print(f"[embed] Saved embeddings {embeddings_arr.shape} → {emb_path}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def cmd_generate(args: argparse.Namespace) -> None:
    """Generate corpus fingerprint (ZAC root)."""

    corpus_jsonl = args.corpus_jsonl
    embedding_npy = args.embedding_npy

    # Step 1: Build corpus (if needed)
    if not args.zac_only:
        if not args.pdf:
            sys.exit("ERROR: --pdf is required unless --zac-only is set.")
        corpus_dir = args.corpus_dir or "output/phase1/corpora"
        corpus_jsonl = _build_image_corpus(args.pdf, corpus_dir)

    if not corpus_jsonl or not Path(corpus_jsonl).exists():
        sys.exit(f"ERROR: corpus JSONL not found: {corpus_jsonl}")

    # Step 2: Compute embeddings (if needed)
    if not args.zac_only:
        emb_dir = args.embedding_dir or "output/phase1/embedding"
        embedding_npy = str(Path(emb_dir) / "embedding.npy")
        _compute_embeddings(
            corpus_jsonl=corpus_jsonl,
            embedding_path=embedding_npy,
            model_path=args.model,
            gpu_id=args.gpu,
            batch_size=args.batch_size,
        )

    if not embedding_npy or not Path(embedding_npy).exists():
        sys.exit(f"ERROR: embedding .npy not found: {embedding_npy}")

    # Step 3: Build ZAC accumulator
    corpus_base_dir = str(Path(corpus_jsonl).parent)
    print(f"\n[ZAC] Building accumulator ...")
    print(f"      embeddings : {embedding_npy}")
    print(f"      corpus     : {corpus_jsonl}")
    print(f"      base_dir   : {corpus_base_dir}")

    t0 = time.time()
    acc = ZACAccumulator.from_corpus(
        embeddings_path=embedding_npy,
        corpus_jsonl=corpus_jsonl,
        corpus_base_dir=corpus_base_dir,
        n_filters=args.n_filters,
    )
    elapsed = time.time() - t0

    root_bytes = len(acc.root_hex()) // 2
    print(f"\n[ZAC] Done in {elapsed:.2f}s")
    print(f"      items      : {len(acc._S)}")
    print(f"      n_filters  : {acc._n_filters}")
    print(f"      ZAC root   : {acc.root_hex()}  ({root_bytes} bytes)")
    print(f"      BF params  : q={acc._bf.q}, k={acc._bf.k}")

    # Step 4: Save manifest + prover state
    output = args.output or "output/phase1/fingerprint.json"
    acc.save(output)
    state_path = str(Path(output).with_name("prover_state.json"))
    acc.save_prover_state(state_path)
    print(f"\n[ZAC] Fingerprint saved    → {output}")
    print(f"[ZAC] Prover state saved   → {state_path}")

    # Sample proof for first element
    if len(acc._S) > 0:
        sample_elem = next(iter(acc._S))
        proof = acc.prove_membership(sample_elem)
        proof_path = str(Path(output).with_name("proof_sample_0.json"))
        with open(proof_path, "w") as f:
            json.dump(proof, f, indent=2)
        print(f"[ZAC] Sample membership proof → {proof_path}")

        ok = acc.verify_membership(sample_elem, proof)
        print(f"[ZAC] Self-verify proof[0]: {'PASS' if ok else 'FAIL'}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify a membership proof against a saved prover state."""
    import hashlib as _hl

    # Try prover_state.json next to the manifest
    manifest_path = Path(args.verify)
    state_path = manifest_path.parent / "prover_state.json"
    if not state_path.exists():
        state_path = manifest_path  # fallback: user passed state directly

    print(f"[ZAC] Loading prover state from {state_path} ...")
    loaded = ZACAccumulator.load_prover_state(str(state_path))
    print(f"      ZAC root   : {loaded.root_hex()}")
    print(f"      Items      : {len(loaded._S)}")
    print(f"      BF params  : q={loaded._bf.q}, k={loaded._bf.k}")

    # Pick element by index
    elements = sorted(loaded._S)
    index = min(args.proof_index, len(elements) - 1)
    elem = elements[index]

    print(f"\n[ZAC] Proving membership for element index {index} ...")
    proof = loaded.prove_membership(elem)
    ok = loaded.verify_membership(elem, proof)
    print(f"[ZAC] Membership proof verification: {'PASS' if ok else 'FAIL'}")

    # Tamper test: find a confirmed non-member
    fake = None
    for i in range(500):
        h = _hl.sha256(f"tamper_test_{i}".encode()).digest()
        if not loaded._bf.check(loaded._v, h):
            fake = h
            break
    if fake:
        ok_tamper = loaded.verify_membership(fake, proof)
        print(f"[ZAC] Tamper detection test: {'PASS (correctly rejected)' if not ok_tamper else 'FAIL (should have been rejected)'}")
    else:
        print("[ZAC] Tamper test skipped: could not find non-member within 500 tries")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate/verify UltraRAG corpus ZAC fingerprint."
    )

    # Mode
    parser.add_argument("--verify", metavar="MANIFEST_JSON",
                        help="Verify mode: load manifest and test proof for --proof-index.")
    parser.add_argument("--proof-index", type=int, default=0,
                        help="Index to generate/verify proof for (default: 0).")

    # Generation inputs
    parser.add_argument("--pdf", help="Path to PDF file (required unless --zac-only).")
    parser.add_argument("--corpus-jsonl", help="Path to existing image.jsonl (--zac-only).")
    parser.add_argument("--embedding-npy", help="Path to existing embedding.npy (--zac-only).")
    parser.add_argument("--zac-only", action="store_true",
                        help="Skip corpus/embedding steps; use existing files.")

    # Generation outputs
    parser.add_argument("--corpus-dir", default="output/phase1/corpora",
                        help="Output directory for corpus (default: output/phase1/corpora).")
    parser.add_argument("--embedding-dir", default="output/phase1/embedding",
                        help="Output directory for embeddings (default: output/phase1/embedding).")
    parser.add_argument("--output", default="output/phase1/fingerprint.json",
                        help="Output path for fingerprint JSON (default: output/phase1/fingerprint.json).")

    # ZAC options
    parser.add_argument("--n-filters", type=int, default=1,
                        help="串联 BF 层数（复合 FPR = ε^n，默认 1）。")

    # Model / hardware
    parser.add_argument("--model", default="/root/autodl-tmp/models/jina-embeddings-v4",
                        help="Path or name of embedding model.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (default: 0).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Embedding batch size (default: 8).")

    args = parser.parse_args()

    if args.verify:
        cmd_verify(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()

"""
Phase 2 — Sumcheck 可验证检索 Demo & 实验脚本

用法
----
# Demo：加载已有 embedding，对 top-5 生成内积证明并验证
python script/phase2_sumcheck.py --demo \
    --embedding-npy  embedding/embedding.npy \
    --corpus-jsonl   corpora/image.jsonl \
    --k 5

# 实验：生成不同 k 下的性能指标表
python script/phase2_sumcheck.py --experiment \
    --embedding-npy  embedding/embedding.npy \
    --corpus-jsonl   corpora/image.jsonl \
    --output         output/phase2/sumcheck_experiment.json

功能
----
  1. 从语料库随机采样 query embedding（模拟实时 query）
  2. 用 FAISS/numpy 计算 top-k 最近邻
  3. 对每对 (q, v_i) 生成 Sumcheck 内积证明
  4. 生成排序证明（delta 见证）
  5. 验证所有证明
  6. 篡改测试：修改一个返回向量，证明验证失败
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# 把 src/ 加进路径
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sumcheck.inner_product import (
    prove_retrieval,
    verify_retrieval,
    prove_inner_product,
    verify_inner_product,
    prove_ranking,
    verify_ranking,
    prove_global_batch,
    verify_global_batch,
    _field_to_signed,
)


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_corpus_paths(corpus_jsonl: str) -> List[str]:
    paths = []
    base = Path(corpus_jsonl).parent
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rel = item.get("image_path", "")
            paths.append(str((base / rel).resolve()))
    return paths


def top_k_by_ip(query: np.ndarray, corpus: np.ndarray, k: int):
    """
    Brute-force inner-product top-k (mirrors FAISS IndexFlatIP).
    Returns (indices, scores) both sorted descending by score.
    """
    scores = corpus @ query          # shape (N,)
    idx = np.argsort(scores)[::-1][:k]
    return idx.tolist(), scores[idx].tolist()


def proof_size_bytes(proof: dict) -> int:
    """Rough serialized size of a proof dict (JSON)."""
    return len(json.dumps(proof).encode())


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo(embedding_npy: str, corpus_jsonl: str, k: int, scale: int = 256):
    print("=" * 60)
    print("Phase 2 — Sumcheck 可验证检索 Demo")
    print("=" * 60)

    embeddings = np.load(embedding_npy).astype(np.float32)
    N, D = embeddings.shape
    print(f"\n语料库：{N} 个向量，维度 D={D}")
    print(f"量化精度：scale={scale}（int≈{scale}×float）")

    # 随机选一个语料库向量作为 query（模拟文字 query 的 embedding）
    rng = np.random.default_rng(42)
    query_idx = int(rng.integers(0, N))
    query = embeddings[query_idx]
    print(f"\n[1/6] 查询向量（来自语料库 index={query_idx}，模拟 query embedding）")

    # Top-k 检索
    t0 = time.perf_counter()
    indices, float_scores = top_k_by_ip(query, embeddings, k)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[2/6] FAISS-like 检索（brute-force inner product）")
    print(f"      top-{k} indices : {indices}")
    print(f"      float scores   : {[round(s, 4) for s in float_scores]}")
    print(f"      检索时间       : {retrieve_ms:.1f} ms")

    corpus_vecs = [embeddings[i].tolist() for i in indices]
    q_list = query.tolist()

    # 生成 Sumcheck 证明
    t0 = time.perf_counter()
    retrieval_proof = prove_retrieval(q_list, corpus_vecs, scale=scale)
    prove_ms = (time.perf_counter() - t0) * 1000
    int_scores = retrieval_proof["rank_proof"]["scores"]

    print(f"\n[3/6] Sumcheck 证明生成")
    print(f"      int scores (×scale²): {int_scores}")
    print(f"      ell (rounds/IP)      : {retrieval_proof['ip_proofs'][0]['ell']}")
    print(f"      单个 IP proof 大小   : {proof_size_bytes(retrieval_proof['ip_proofs'][0])} B")
    print(f"      全部证明大小         : {proof_size_bytes(retrieval_proof)} B")
    print(f"      Prove 时间           : {prove_ms:.1f} ms（{k} 个 IP + 1 ranking）")

    # 验证
    t0 = time.perf_counter()
    ok = verify_retrieval(q_list, corpus_vecs, retrieval_proof)
    verify_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[4/6] 验证")
    print(f"      结果     : {'✓ PASS' if ok else '✗ FAIL'}")
    print(f"      Verify 时间 : {verify_ms:.1f} ms")

    # 篡改测试：把第一个结果向量的某个分量改掉
    print(f"\n[5/6] 篡改测试（修改返回向量第一个分量）")
    tampered_vecs = [v.copy() for v in corpus_vecs]
    tampered_vecs[0][0] += 1.0      # 轻微篡改
    ok_tamper = verify_retrieval(q_list, tampered_vecs, retrieval_proof)
    print(f"      篡改后验证 : {'✗ FAIL (correctly rejected)' if not ok_tamper else '✓ PASS (should have failed!)'}")

    # 排序篡改测试：调换前两个结果顺序
    print(f"\n[6/6] 排序篡改测试（调换 top-1 和 top-2 向量顺序）")
    swapped_vecs = [corpus_vecs[1], corpus_vecs[0]] + corpus_vecs[2:]
    ok_swap = verify_retrieval(q_list, swapped_vecs, retrieval_proof)
    print(f"      调换后验证 : {'✗ FAIL (correctly rejected)' if not ok_swap else '✓ PASS (should have failed!)'}")

    print("\n" + "=" * 60)
    print("Demo 完成")
    print("=" * 60)


# ── 实验：不同 k 下的性能指标 ───────────────────────────────────────────────────

def run_experiment(
    embedding_npy: str,
    corpus_jsonl: str,
    output: str,
    k_values: List[int] = None,
    n_trials: int = 3,
    scale: int = 256,
):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print("=" * 60)
    print("Phase 2 — Sumcheck 性能实验")
    print("=" * 60)

    embeddings = np.load(embedding_npy).astype(np.float32)
    N, D = embeddings.shape
    print(f"语料库：N={N}，D={D}，scale={scale}\n")

    rng = np.random.default_rng(0)
    results = []

    for k in k_values:
        prove_times, verify_times, proof_sizes = [], [], []
        all_pass = True

        for trial in range(n_trials):
            q_idx = int(rng.integers(0, N))
            query = embeddings[q_idx]
            indices, _ = top_k_by_ip(query, embeddings, k)
            corpus_vecs = [embeddings[i].tolist() for i in indices]
            q_list = query.tolist()

            t0 = time.perf_counter()
            proof = prove_retrieval(q_list, corpus_vecs, scale=scale)
            prove_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            ok = verify_retrieval(q_list, corpus_vecs, proof)
            verify_ms = (time.perf_counter() - t0) * 1000

            if not ok:
                all_pass = False

            prove_times.append(prove_ms)
            verify_times.append(verify_ms)
            proof_sizes.append(proof_size_bytes(proof))

        med_prove  = round(sorted(prove_times)[n_trials // 2], 1)
        med_verify = round(sorted(verify_times)[n_trials // 2], 1)
        med_size   = sorted(proof_sizes)[n_trials // 2]

        # Merkle 对比：每条路径 ceil(log2(N))×32B，k 条互相独立
        import math
        merkle_bytes = k * math.ceil(math.log2(max(N, 2))) * 32

        print(
            f"  k={k:2d}  prove={med_prove:7.1f} ms  verify={med_verify:7.1f} ms  "
            f"proof={med_size} B  Merkle={merkle_bytes} B  {'✓' if all_pass else '✗'}"
        )
        results.append({
            "k": k,
            "N": N,
            "D": D,
            "scale": scale,
            "prove_ms_median": med_prove,
            "verify_ms_median": med_verify,
            "proof_bytes": med_size,
            "merkle_bytes": merkle_bytes,
            "all_pass": all_pass,
        })

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n实验结果保存至: {output}")


# ── 单元测试（无需语料库，直接运行） ───────────────────────────────────────────

def run_unit_tests():
    import random
    rng = random.Random(42)

    print("=== Sumcheck 单元测试 ===\n")

    def rand_vec(d, scale=1.0):
        return [rng.uniform(-scale, scale) for _ in range(d)]

    passed = 0
    total = 0

    for d in [4, 16, 128, 256, 2048]:
        q = rand_vec(d)
        v = rand_vec(d)
        proof = prove_inner_product(q, v, scale=65536)
        ok = verify_inner_product(q, v, proof, scale=65536)
        # Check claimed H matches direct computation
        from sumcheck.inner_product import quantize, _m
        q_int = quantize(q, 256)
        v_int = quantize(v, 256)
        expected_H = _m(sum(_m(qi * vi) for qi, vi in zip(q_int, v_int)))
        H_ok = proof["H"] == expected_H
        status = "PASS" if (ok and H_ok) else "FAIL"
        print(f"  d={d:5d}  verify={status}  H_check={'OK' if H_ok else 'WRONG'}")
        passed += (ok and H_ok)
        total += 1

    # Tamper test
    q = rand_vec(64)
    v = rand_vec(64)
    proof = prove_inner_product(q, v, scale=65536)
    v_bad = v.copy(); v_bad[0] += 0.1
    ok_tamper = verify_inner_product(q, v_bad, proof, scale=65536)
    print(f"  Tamper test (d=64, v[0]+=0.1): {'PASS (rejected)' if not ok_tamper else 'FAIL (accepted!)'}")
    passed += (not ok_tamper)
    total += 1

    # Ranking proof
    scores = [1000, 800, 600, 400, 200]
    rp = prove_ranking(scores)
    rp_ok = verify_ranking(scores, rp)
    scores_wrong = [800, 1000, 600, 400, 200]
    rp_wrong = verify_ranking(scores_wrong, rp)
    print(f"  Ranking proof (correct order): {'PASS' if rp_ok else 'FAIL'}")
    print(f"  Ranking proof (wrong order)  : {'PASS (rejected)' if not rp_wrong else 'FAIL (accepted!)'}")
    passed += rp_ok + (not rp_wrong)
    total += 2

    # ── Global batch Sumcheck tests ──────────────────────────────────────────
    print("\n--- Global Batch Sumcheck ---")

    # Basic: N=8, d=16, top-k=3
    N_g, d_g, k_g = 8, 16, 3
    corpus_g = [rand_vec(d_g) for _ in range(N_g)]
    q_g = rand_vec(d_g)
    proof_g = prove_global_batch(q_g, corpus_g, scale=65536)
    result_g = verify_global_batch(q_g, corpus_g, proof_g, top_k=k_g)
    gb_ok = result_g["verified"] and len(result_g["top_k_indices"]) == k_g
    print(f"  Global batch N={N_g}, d={d_g}, k={k_g}: {'PASS' if gb_ok else 'FAIL'}")
    passed += gb_ok
    total += 1

    # Larger: N=20, d=64, top-k=5
    N_g2, d_g2, k_g2 = 20, 64, 5
    corpus_g2 = [rand_vec(d_g2) for _ in range(N_g2)]
    q_g2 = rand_vec(d_g2)
    proof_g2 = prove_global_batch(q_g2, corpus_g2, scale=65536)
    result_g2 = verify_global_batch(q_g2, corpus_g2, proof_g2, top_k=k_g2)
    gb2_ok = result_g2["verified"] and len(result_g2["top_k_indices"]) == k_g2
    print(f"  Global batch N={N_g2}, d={d_g2}, k={k_g2}: {'PASS' if gb2_ok else 'FAIL'}")
    passed += gb2_ok
    total += 1

    # Verifier selects top-k correctly: cross-check with numpy
    scores_signed = result_g2["top_k_scores"]
    all_scores_np = [float(q_g2[j]) * float(corpus_g2[i][j]) for i in range(N_g2) for j in range(d_g2)]
    # just verify top indices are the highest
    all_dots = [sum(q_g2[j] * corpus_g2[i][j] for j in range(d_g2)) for i in range(N_g2)]
    expected_topk = sorted(range(N_g2), key=lambda i: all_dots[i], reverse=True)[:k_g2]
    topk_match = set(result_g2["top_k_indices"]) == set(expected_topk)
    print(f"  Top-k indices match numpy ground truth (N={N_g2}, k={k_g2}): {'PASS' if topk_match else 'FAIL'}")
    passed += topk_match
    total += 1

    # Tamper test: modify one corpus vector → should fail
    corpus_g_tampered = [v.copy() for v in corpus_g]
    corpus_g_tampered[0][0] += 0.5
    result_g_tamper = verify_global_batch(q_g, corpus_g_tampered, proof_g, top_k=k_g)
    gb_tamper_ok = not result_g_tamper["verified"]
    print(f"  Global tamper test (corpus[0][0]+=0.5): {'PASS (rejected)' if gb_tamper_ok else 'FAIL (accepted!)'}")
    passed += gb_tamper_ok
    total += 1

    # Score tampering: modify announced scores in proof → should fail
    proof_g_bad = dict(proof_g)
    proof_g_bad["scores"] = list(proof_g["scores"])
    proof_g_bad["scores"][0] = (proof_g_bad["scores"][0] + 999999) % ((1 << 61) - 1)
    result_g_score_tamper = verify_global_batch(q_g, corpus_g, proof_g_bad, top_k=k_g)
    gb_score_ok = not result_g_score_tamper["verified"]
    print(f"  Global score tamper test: {'PASS (rejected)' if gb_score_ok else 'FAIL (accepted!)'}")
    passed += gb_score_ok
    total += 1

    print(f"\n结果：{passed}/{total} 通过")
    return passed == total


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Sumcheck 可验证检索"
    )
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--test",       action="store_true", help="运行单元测试（无需语料库）")
    parser.add_argument("--embedding-npy",  default="embedding/embedding.npy")
    parser.add_argument("--corpus-jsonl",   default="corpora/image.jsonl")
    parser.add_argument("--k",          type=int, default=5)
    parser.add_argument("--scale",      type=int, default=65536)
    parser.add_argument("--output",     default="output/phase2/sumcheck_experiment.json")
    args = parser.parse_args()

    if args.test:
        ok = run_unit_tests()
        sys.exit(0 if ok else 1)

    if args.demo:
        run_demo(args.embedding_npy, args.corpus_jsonl, args.k, args.scale)

    if args.experiment:
        run_experiment(
            args.embedding_npy,
            args.corpus_jsonl,
            args.output,
            scale=args.scale,
        )

    if not (args.demo or args.experiment or args.test):
        parser.print_help()


if __name__ == "__main__":
    main()

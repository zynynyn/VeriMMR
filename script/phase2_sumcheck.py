"""
Phase 2 — Sumcheck 可验证检索 Demo & 实验脚本

用法
----
# Demo：加载已有 embedding，对 top-5 生成内积证明并验证
python script/phase2_sumcheck.py --demo \
    --embedding-npy  embedding/embedding.npy \
    --corpus-jsonl   corpora/image.jsonl \
    --k 5

# IPA 模式 Demo（不依赖原始 embedding）
python script/phase2_sumcheck.py --demo --ipa \
    --commitment-path embedding/embedding_commitments.bin \
    --workdir         zkllm-workdir/jina-v4 \
    --embedding-npy   embedding/embedding.npy --k 5

# 实验：生成不同 k 下的性能指标表
python script/phase2_sumcheck.py --experiment \
    --embedding-npy  embedding/embedding.npy \
    --output         output/phase2/sumcheck_experiment.json

# IPA 实验（额外测量 cm_w 聚合 + oracle proof 开销）
python script/phase2_sumcheck.py --experiment --ipa \
    --commitment-path embedding/embedding_commitments.bin \
    --workdir zkllm-workdir/jina-v4 \
    --output  output/phase2/sumcheck_experiment_ipa.json

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
    P_FR,
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
    ipa_mode: bool = False,
    commitment_path: str = None,
    workdir: str = None,
    pp_path: str = None,
):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    print("=" * 60)
    print(f"Phase 2 — Sumcheck 性能实验{'（IPA 模式）' if ipa_mode else ''}")
    print("=" * 60)

    embeddings = np.load(embedding_npy).astype(np.float32)
    N, D = embeddings.shape
    print(f"语料库：N={N}，D={D}，scale={scale}\n")

    rng = np.random.default_rng(0)
    results = []

    for k in k_values:
        prove_times, verify_times, proof_sizes = [], [], []
        oracle_proof_sizes, cm_agg_times, oracle_verify_times = [], [], []
        all_pass = True

        for trial in range(n_trials):
            q_idx = int(rng.integers(0, N))
            query = embeddings[q_idx]
            # Global batch: use all N corpus vectors
            corpus_vecs = embeddings.tolist()
            q_list = query.tolist()

            t0 = time.perf_counter()
            if ipa_mode:
                proof = prove_global_batch(q_list, corpus_vecs, scale=scale,
                                           ipa_mode=True, pp_path=pp_path,
                                           workdir=workdir)
            else:
                proof = prove_global_batch(q_list, corpus_vecs, scale=scale)
            prove_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            if ipa_mode:
                # Measure cm_w aggregation time
                t_cm = time.perf_counter()
                vr = verify_global_batch(q_list, None, proof, top_k=k,
                                         commitment_path=commitment_path)
                cm_agg_ms = round((time.perf_counter() - t_cm) * 1000, 1)
                oracle_verify_times.append(cm_agg_ms)
                op_bytes = len(proof.get("oracle_proof", b""))
                oracle_proof_sizes.append(op_bytes)
            else:
                vr = verify_global_batch(q_list, corpus_vecs, proof, top_k=k)
            verify_ms = (time.perf_counter() - t0) * 1000

            if not vr.get("verified", False):
                all_pass = False

            prove_times.append(prove_ms)
            verify_times.append(verify_ms)
            proof_sizes.append(proof_size_bytes(proof))

        med_prove  = round(sorted(prove_times)[n_trials // 2], 1)
        med_verify = round(sorted(verify_times)[n_trials // 2], 1)
        med_size   = sorted(proof_sizes)[n_trials // 2]

        import math
        merkle_bytes = k * math.ceil(math.log2(max(N, 2))) * 32

        row = {
            "k": k, "N": N, "D": D, "scale": scale,
            "ipa_mode": ipa_mode,
            "prove_ms_median": med_prove,
            "verify_ms_median": med_verify,
            "proof_bytes": med_size,
            "merkle_bytes": merkle_bytes,
            "all_pass": all_pass,
        }
        line = (
            f"  k={k:2d}  prove={med_prove:7.1f} ms  verify={med_verify:7.1f} ms  "
            f"proof={med_size} B  Merkle={merkle_bytes} B  {'✓' if all_pass else '✗'}"
        )
        if ipa_mode and oracle_proof_sizes:
            med_op   = sorted(oracle_proof_sizes)[n_trials // 2]
            med_ovm  = round(sorted(oracle_verify_times)[n_trials // 2], 1)
            row["oracle_proof_bytes"] = med_op
            row["verifier_ipa_ms"]    = med_ovm
            line += f"  oracle={med_op}B  ipa_verify={med_ovm}ms"
        print(line)
        results.append(row)

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

    from sumcheck.inner_product import quantize, _m
    for d in [4, 16, 128, 256, 2048]:
        q = rand_vec(d)
        v = rand_vec(d)
        proof = prove_inner_product(q, v, scale=65536)
        ok = verify_inner_product(q, v, proof, scale=65536)
        # Check claimed H matches direct computation (same scale as prove)
        q_int = quantize(q, 65536)
        v_int = quantize(v, 65536)
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

    # ── IPA mode unit tests ─────────────────────────────────────────────────
    print("\n--- IPA Oracle 模式单元测试 ---")

    bin_dir = Path(__file__).resolve().parents[1] / "src" / "zkllm"
    ppgen_ok = (bin_dir / "ppgen").exists() and (bin_dir / "commit-param").exists() \
               and (bin_dir / "open-ipa").exists()

    if not ppgen_ok:
        print("  [SKIP] IPA binary 未找到（ppgen/commit-param/open-ipa），跳过 IPA 测试")
    else:
        import subprocess
        import tempfile as _tmpmod
        import numpy as _np

        d_ipa = 16   # small dim for speed (D=16 → ell=4 rounds)
        N_ipa = 5

        with _tmpmod.TemporaryDirectory() as td:
            td = Path(td)

            # Generate pp
            pp_f = str(td / "pp.bin")
            subprocess.run([str(bin_dir / "ppgen"), str(d_ipa), pp_f],
                           capture_output=True, check=True)

            # Generate corpus vectors and commitments
            corpus_ipa = [rand_vec(d_ipa) for _ in range(N_ipa)]
            q_ipa = rand_vec(d_ipa)
            cm_all = []
            for v in corpus_ipa:
                v_int = _np.round(_np.array(v) * 65536).astype(_np.int32)
                vf = str(td / "v.bin"); cmf = str(td / "cm.bin")
                v_int.tofile(vf)
                subprocess.run([str(bin_dir / "commit-param"), pp_f, vf, cmf, "1", str(d_ipa)],
                               capture_output=True, check=True)
                with open(cmf, "rb") as f:
                    cm_all.append(f.read())
            cm_path = str(td / "coms.bin")
            with open(cm_path, "wb") as f:
                for cm in cm_all:
                    f.write(cm)

            # Test IPA prove+verify (correct)
            proof_ipa = prove_global_batch(q_ipa, corpus_ipa, scale=65536,
                                           ipa_mode=True,
                                           pp_path=pp_f,
                                           workdir=str(td))
            r_ipa = verify_global_batch(q_ipa, None, proof_ipa, top_k=3,
                                        commitment_path=cm_path)
            ipa_ok = r_ipa["verified"] and r_ipa.get("oracle_ok")
            print(f"  IPA prove+verify (N={N_ipa}, d={d_ipa}): "
                  f"{'PASS' if ipa_ok else 'FAIL'}  "
                  f"oracle_proof_bytes={len(proof_ipa.get('oracle_proof', b''))}")
            passed += ipa_ok
            total += 1

            # Test IPA tamper: corrupt commitment → binding_ok should fail
            cm_bad_path = str(td / "coms_bad.bin")
            import shutil
            shutil.copy(cm_path, cm_bad_path)
            with open(cm_bad_path, "r+b") as f:
                f.seek(10)
                f.write(b"\xff\xff\xff\xff")
            r_ipa_bad = verify_global_batch(q_ipa, None, proof_ipa, top_k=3,
                                            commitment_path=cm_bad_path)
            ipa_tamper_ok = not r_ipa_bad["verified"]
            print(f"  IPA tamper (corrupt commitment): "
                  f"{'PASS (rejected)' if ipa_tamper_ok else 'FAIL (accepted!)'}")
            passed += ipa_tamper_ok
            total += 1

            # Consistency: IPA top-k should match non-IPA top-k
            proof_nipa = prove_global_batch(q_ipa, corpus_ipa, scale=65536)
            r_nipa = verify_global_batch(q_ipa, corpus_ipa, proof_nipa, top_k=3)
            topk_consistent = set(r_ipa["top_k_indices"]) == set(r_nipa["top_k_indices"])
            print(f"  IPA vs non-IPA top-k consistency: "
                  f"{'PASS' if topk_consistent else 'FAIL'}  "
                  f"ipa={r_ipa['top_k_indices']} nipa={r_nipa['top_k_indices']}")
            passed += topk_consistent
            total += 1

    print(f"\n结果：{passed}/{total} 通过")
    return passed == total

# ── main ───────────────────────────────────────────────────────────────────────
    top_k: int = 5,
    output: str = "output/phase2/b3_detection.json",
):
    """
    B3 攻击检测实验：Prover 将非 top-k 向量的分值改为 max+1 注入到 top-k。

    对每个 query：
      1. 正常计算 top-k（FAISS）
      2. 从非 top-k 随机选 n_victims 个 victim
      3. 将 victim 的分值改为 max_score+1，注入到 proof.scores
      4. 验证篡改后的 proof — 期望 verified=False（检测成功）

    报告检测率（应为 100%）。
    """
    print("=" * 60)
    print("Phase 2 — B3 篡改检测实验")
    print("=" * 60)

    embeddings = np.load(embedding_npy).astype(np.float32)
    N, D = embeddings.shape
    print(f"语料库：N={N}，D={D}，n_queries={n_queries}，n_victims={n_victims}\n")

    rng = np.random.default_rng(7)
    records = []
    detected = 0
    total_victims = 0

    for qi in range(n_queries):
        q_idx = int(rng.integers(0, N))
        query = embeddings[q_idx]
        corpus_vecs = embeddings.tolist()
        q_list = query.tolist()

        # 正常证明
        proof_ok = prove_global_batch(q_list, corpus_vecs, scale=65536)
        vr_before = verify_global_batch(q_list, corpus_vecs, proof_ok, top_k=top_k)
        topk_set = set(vr_before["top_k_indices"])
        non_topk = [i for i in range(N) if i not in topk_set]

        victims = rng.choice(non_topk, size=min(n_victims, len(non_topk)), replace=False).tolist()
        max_score = max(proof_ok["scores"])

        for victim in victims:
            # 篡改：把 victim 的分值改为 max+1
            proof_bad = dict(proof_ok)
            proof_bad["scores"] = list(proof_ok["scores"])
            proof_bad["scores"][victim] = (max_score + 1) % ((1 << 61) - 1)

            vr_after = verify_global_batch(q_list, corpus_vecs, proof_bad, top_k=top_k)
            det = not vr_after["verified"]
            detected += det
            total_victims += 1
            records.append({
                "query_idx": q_idx,
                "victim_idx": victim,
                "detected": det,
            })

        print(f"  query {qi+1}/{n_queries}  q_idx={q_idx}  "
              f"victims={len(victims)}  "
              f"检测率={detected}/{total_victims}={detected/max(total_victims,1)*100:.1f}%")

    detection_rate = detected / max(total_victims, 1)
    print(f"\n总检测率: {detected}/{total_victims} = {detection_rate*100:.2f}%")

    result = {
        "n_queries": n_queries,
        "n_victims": n_victims,
        "top_k": top_k,
        "N": N,
        "total_attacks": total_victims,
        "detected": detected,
        "detection_rate": detection_rate,
        "records": records,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"结果保存至: {output}")
    return result


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Sumcheck 可验证检索"
    )
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--experiment", action="store_true")
    parser.add_argument("--test",       action="store_true", help="运行单元测试（无需语料库）")
    parser.add_argument("--ipa",        action="store_true", help="启用 IPA oracle 模式")
    parser.add_argument("--commitment-path", default=None,
                        help="embedding_commitments.bin 路径（IPA 模式必需）")
    parser.add_argument("--workdir",    default="zkllm-workdir/jina-v4",
                        help="IPA 临时文件目录，需含 embedding-pp.bin")
    parser.add_argument("--embedding-npy",  default="embedding/embedding.npy")
    parser.add_argument("--corpus-jsonl",   default="corpora/image.jsonl")
    parser.add_argument("--k",          type=int, default=5)
    parser.add_argument("--scale",      type=int, default=65536)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    _ROOT = Path(__file__).resolve().parents[1]

    # Resolve IPA paths
    pp_path = str((_ROOT / args.workdir / "embedding-pp.bin").resolve()) \
              if args.ipa else None
    commitment_path = str((_ROOT / args.commitment_path).resolve()) \
                      if args.commitment_path else None
    workdir = str((_ROOT / args.workdir).resolve()) if args.ipa else None

    if args.test:
        ok = run_unit_tests()
        sys.exit(0 if ok else 1)

    if args.demo:
        run_demo(args.embedding_npy, args.corpus_jsonl, args.k, args.scale)

    if args.experiment:
        default_out = "output/phase2/sumcheck_experiment_ipa.json" if args.ipa \
                      else "output/phase2/sumcheck_experiment.json"
        run_experiment(
            args.embedding_npy,
            args.corpus_jsonl,
            output=args.output or default_out,
            scale=args.scale,
            ipa_mode=args.ipa,
            commitment_path=commitment_path,
            workdir=workdir,
            pp_path=pp_path,
        )

    if not (args.demo or args.experiment or args.test):
        parser.print_help()


if __name__ == "__main__":
    main()

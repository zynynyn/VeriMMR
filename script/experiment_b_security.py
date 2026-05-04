"""
实验 B：安全验证（B1-B4 四种攻击场景）

B1 图像替换攻击    → 被 ZAC 成员证明检测（50 个随机样本）
B2 Embedding 替换  → 被 ZAC 跨层绑定检测（50 个随机样本）
B3 排名操控攻击    → 被 Sumcheck 验证检测（10 条 query × 5 个篡改位置）
B4 权重矩阵篡改    → 被 zkLLM Sumcheck 承诺绑定检测
  攻击：篡改 layer-35 的实际权重（int.bin），保持承诺（commitment.bin）不变，
        Prover 用篡改权重计算，Verifier 用原始承诺验证 → 多线性扩展不匹配 → rc≠0

每个攻击场景期望结果：verified=False（检测率 100%）

用法
----
# 全部攻击（non-IPA，需要 GPU 运行 B4）
python script/experiment_b_security.py

# 仅 B3（CPU-only，non-IPA）
python script/experiment_b_security.py --only-b3

# 仅 B3（IPA 模式，CPU-only，无 GPU）
python script/experiment_b_security.py --only-b3 --ipa \\
    --commitment-path embedding/embedding_commitments_cpu.bin \\
    --pp-pkl-path     zkllm-workdir/jina-v4/embedding-pp-cpu.pkl
"""

import argparse
import sys, os, json, shutil, copy, time, subprocess, tempfile
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zac.accumulator import ZACAccumulator
from sumcheck.inner_product import prove_global_batch, verify_global_batch

CORPUS_JSONL  = ROOT / "corpora" / "image.jsonl"
EMBEDDING_NPY = ROOT / "embedding" / "embedding.npy"
PROVER_STATE  = ROOT / "output" / "phase1" / "prover_state.json"
ZKLLM_WORKDIR = ROOT / "zkllm-workdir" / "jina-v4"
ZKLLM_BIN     = ROOT / "src" / "zkllm" / "bin"
ZKLLM_CWD     = ROOT / "src" / "zkllm"
CORPUS_BASE   = CORPUS_JSONL.parent

TOP_K   = 5
N_B1B2  = 50
N_B3_Q  = 10
N_B3_V  = 5

EMBED_DIM    = 2048
HIDDEN_DIM   = 11008
KV_DIM       = 256
NUM_KV_HEADS = 2
SEQ_LEN      = 1024
B4_LAYER     = 35

PASS_STR = "\033[32m✅ 通过\033[0m"
FAIL_STR = "\033[31m❌ 未检测到\033[0m"

# ── CLI 参数 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="实验 B：安全验证（B1-B4）")
parser.add_argument("--only-b3", action="store_true",
                    help="仅运行 B3 排名操控实验（跳过 B1/B2/B4）")
parser.add_argument("--ipa", action="store_true",
                    help="B3 使用 IPA oracle 模式（BLS12-381 Fr 域）")
parser.add_argument("--commitment-path", default=None,
                    help="IPA 模式：embedding_commitments*.bin 路径（GPU 默认 embedding/embedding_commitments.bin）")
parser.add_argument("--pp-pkl-path", default=None,
                    help="CPU IPA 模式：embedding-pp-cpu.pkl 路径（提供则走 CPU 路径）")
parser.add_argument("--pp-path", default=None,
                    help="GPU IPA 模式：embedding-pp.bin 路径（默认 zkllm-workdir/jina-v4/embedding-pp.bin）")
parser.add_argument("--workdir", default=None,
                    help="GPU IPA 模式：open-ipa 临时文件目录（默认 zkllm-workdir/jina-v4）")
parser.add_argument("--output", default=None,
                    help="结果 JSON 输出路径（默认 notes/experiment_b_result.json）")
args = parser.parse_args()

IPA_MODE        = args.ipa
COMMITMENT_PATH = args.commitment_path
PP_PKL_PATH     = args.pp_pkl_path
ONLY_B3         = args.only_b3
OUT_PATH        = Path(args.output) if args.output else ROOT / "notes" / "experiment_b_result.json"

# GPU IPA 默认路径
_PP_PATH  = args.pp_path  or str(ROOT / "zkllm-workdir" / "jina-v4" / "embedding-pp.bin")
_WORKDIR  = args.workdir  or str(ROOT / "zkllm-workdir" / "jina-v4")
if IPA_MODE and COMMITMENT_PATH is None:
    if PP_PKL_PATH is not None:
        COMMITMENT_PATH = str(ROOT / "embedding" / "embedding_commitments_cpu.bin")
    else:
        COMMITMENT_PATH = str(ROOT / "embedding" / "embedding_commitments.bin")

# ── 加载语料库 ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("实验 B：安全验证（B1-B4）")
print(f"{'='*60}")

corpus_entries = [json.loads(l) for l in CORPUS_JSONL.read_text().splitlines()]
embeddings     = np.load(EMBEDDING_NPY).astype(np.float32)
N, D           = embeddings.shape
all_paths      = [str(CORPUS_BASE / e["image_path"]) for e in corpus_entries]

print(f"语料库：N={N}  D={D}")

# ── 初始化 ZAC accumulator（仅 B1/B2/B4 需要）────────────────────────────────
if not ONLY_B3:
    print("\n[准备] 加载 ZAC prover state...")
    t0 = time.time()
    acc = ZACAccumulator.load_prover_state(str(PROVER_STATE))
    print(f"  ZAC Root: {acc.root_hex()[:32]}…  ({time.time()-t0:.1f}s)\n")

    def zac_verify_single(path: str, emb_matrix: np.ndarray, idx: int) -> bool:
        element = ZACAccumulator.image_embedding_hash(path, emb_matrix[idx])
        proof   = acc.prove_membership_batch([element])
        return   acc.verify_membership_batch([element], proof)

# ── 加载 IPA pp_generators（B3 CPU IPA 模式）────────────────────────────────
pp_generators = None
if IPA_MODE:
    if PP_PKL_PATH is not None:
        import pickle
        print(f"\n[IPA-CPU] 加载 pp_generators from {PP_PKL_PATH} ...")
        t0 = time.time()
        with open(PP_PKL_PATH, "rb") as _f:
            pp_generators = pickle.load(_f)
        print(f"  加载 {len(pp_generators)} 个 G1 生成器  ({time.time()-t0:.1f}s)")
    else:
        print(f"\n[IPA-GPU] 使用 open-ipa binary  pp={_PP_PATH}")

results = {}
rng = np.random.default_rng(42)

if not ONLY_B3:
    # ══════════════════════════════════════════════════════════════════════════
    # B1：图像替换攻击（50 个随机样本）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"{'─'*60}")
    print(f"B1 图像替换攻击（{N_B1B2} 个随机样本）")
    print("攻击：替换目标图像文件内容，embedding 不变，ZAC Root 不变")
    print(f"{'─'*60}")

    victim_indices = rng.choice(N, size=N_B1B2, replace=False).tolist()
    b1_detected = 0
    b1_detail   = []
    t_b1 = time.time()

    for i, vic in enumerate(victim_indices):
        donor = (vic + N // 2) % N
        if donor == vic:
            donor = (vic + 1) % N

        vic_path   = all_paths[vic]
        donor_path = all_paths[donor]
        backup     = vic_path + ".b1_bak"

        ok_before = zac_verify_single(vic_path, embeddings, vic)
        shutil.copy2(vic_path, backup)
        try:
            shutil.copy2(donor_path, vic_path)
            ok_after = zac_verify_single(vic_path, embeddings, vic)
        finally:
            shutil.copy2(backup, vic_path)
            os.remove(backup)

        detected = (ok_before is True) and (ok_after is False)
        if detected:
            b1_detected += 1
        b1_detail.append({"victim": vic, "donor": donor,
                          "before": bool(ok_before), "after": bool(ok_after),
                          "detected": detected})
        print(f"  [{i+1:02d}/{N_B1B2}] victim={vic:3d} donor={donor:3d}  "
              f"before={ok_before} after={ok_after}  {'✓' if detected else '✗'}")

    b1_rate = b1_detected / N_B1B2
    b1_pass = (b1_detected == N_B1B2)
    print(f"\n  检测率：{b1_detected}/{N_B1B2} = {b1_rate*100:.1f}%  耗时 {time.time()-t_b1:.1f}s")
    print(f"  结果：{PASS_STR if b1_pass else FAIL_STR}")
    results["B1"] = {"detected": b1_detected, "total": N_B1B2,
                     "detection_rate": b1_rate, "pass": b1_pass, "detail": b1_detail}

    # ══════════════════════════════════════════════════════════════════════════
    # B2：Embedding 替换攻击——ZAC 跨层绑定（50 个随机样本）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"B2 Embedding 替换攻击——ZAC 跨层绑定（{N_B1B2} 个随机样本）")
    print("攻击：保持图像文件不变，将 embedding 替换为随机单位向量")
    print(f"{'─'*60}")

    target_indices = rng.choice(N, size=N_B1B2, replace=False).tolist()
    b2_detected = 0
    b2_detail   = []
    t_b2 = time.time()

    for i, tgt in enumerate(target_indices):
        tgt_path = all_paths[tgt]
        ok_before = zac_verify_single(tgt_path, embeddings, tgt)

        forged_embs      = embeddings.copy()
        fake_vec         = rng.standard_normal(D).astype(np.float32)
        fake_vec        /= np.linalg.norm(fake_vec)
        forged_embs[tgt] = fake_vec

        ok_after = zac_verify_single(tgt_path, forged_embs, tgt)

        detected = (ok_before is True) and (ok_after is False)
        if detected:
            b2_detected += 1
        b2_detail.append({"target": tgt, "before": bool(ok_before),
                          "after": bool(ok_after), "detected": detected})
        print(f"  [{i+1:02d}/{N_B1B2}] target={tgt:3d}  "
              f"before={ok_before} after={ok_after}  {'✓' if detected else '✗'}")

    b2_rate = b2_detected / N_B1B2
    b2_pass = (b2_detected == N_B1B2)
    print(f"\n  检测率：{b2_detected}/{N_B1B2} = {b2_rate*100:.1f}%  耗时 {time.time()-t_b2:.1f}s")
    print(f"  结果：{PASS_STR if b2_pass else FAIL_STR}")
    results["B2"] = {"detected": b2_detected, "total": N_B1B2,
                     "detection_rate": b2_rate, "pass": b2_pass, "detail": b2_detail}

# ══════════════════════════════════════════════════════════════════════════════
# B3：排名操控攻击（10 条 query × 5 个篡改位置）
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"B3 排名操控攻击——Sumcheck（{N_B3_Q} 条 query × {N_B3_V} 个篡改位置）")
mode_tag = ("IPA/CPU" if pp_generators is not None else "IPA/GPU") if IPA_MODE else "non-IPA"
print(f"攻击：生成合法证明后，篡改 scores 列表中低排名条目的声明分值  [{mode_tag}]")
print(f"{'─'*60}")

query_indices = [int(i * N / N_B3_Q) for i in range(N_B3_Q)]
corpus_vecs   = embeddings.tolist()

b3_detected = 0
b3_total    = N_B3_Q * N_B3_V
b3_detail   = []
t_b3 = time.time()

for qi, q_idx in enumerate(query_indices):
    q_vec = embeddings[q_idx].tolist()

    proof_ok  = prove_global_batch(q_vec, corpus_vecs,
                                   ipa_mode=IPA_MODE,
                                   pp_generators=pp_generators,
                                   pp_path=_PP_PATH if IPA_MODE and pp_generators is None else None,
                                   workdir=_WORKDIR if IPA_MODE and pp_generators is None else None)
    vr_before = verify_global_batch(
        q_vec,
        corpus_vecs if not IPA_MODE else None,
        proof_ok, TOP_K,
        commitment_path=COMMITMENT_PATH if IPA_MODE else None,
    )
    ok_before = vr_before["verified"]
    top_k_set = set(vr_before["top_k_indices"])

    non_topk    = [i for i in range(N) if i not in top_k_set]
    victim_pool = rng.choice(non_topk, size=N_B3_V, replace=False).tolist()
    max_score   = max(proof_ok["scores"])

    row_detected = 0
    for vi, vic in enumerate(victim_pool):
        proof_tamper = copy.deepcopy(proof_ok)
        orig_score   = proof_tamper["scores"][vic]
        proof_tamper["scores"][vic] = max_score + 1

        vr_after = verify_global_batch(
            q_vec,
            corpus_vecs if not IPA_MODE else None,
            proof_tamper, TOP_K,
            commitment_path=COMMITMENT_PATH if IPA_MODE else None,
        )
        ok_after = vr_after["verified"]

        detected = (ok_before is True) and (ok_after is False)
        if detected:
            b3_detected += 1
            row_detected += 1
        b3_detail.append({"query_idx": q_idx, "victim": vic,
                          "orig_score": orig_score,
                          "tampered_score": max_score + 1,
                          "before": bool(ok_before), "after": bool(ok_after),
                          "detected": detected})

    print(f"  [query {qi+1:02d}/{N_B3_Q}] q_idx={q_idx:3d}  "
          f"before={ok_before}  篡改检测 {row_detected}/{N_B3_V}  "
          f"victims={victim_pool}")

b3_rate = b3_detected / b3_total
b3_pass = (b3_detected == b3_total)
print(f"\n  检测率：{b3_detected}/{b3_total} = {b3_rate*100:.1f}%  耗时 {time.time()-t_b3:.1f}s")
print(f"  结果：{PASS_STR if b3_pass else FAIL_STR}")
results["B3"] = {"detected": b3_detected, "total": b3_total,
                 "detection_rate": b3_rate, "pass": b3_pass,
                 "ipa_mode": IPA_MODE, "detail": b3_detail}

if not ONLY_B3:
    # ══════════════════════════════════════════════════════════════════════════
    # B4：zkLLM 权重矩阵篡改攻击（承诺绑定）
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"B4 zkLLM 权重矩阵篡改攻击（承诺绑定，layer-{B4_LAYER}）")
    print("攻击：篡改 gate_proj 权重 int.bin，commitment.bin 不变")
    print("期望：Sumcheck 多线性扩展不匹配 → FFN binary rc≠0")
    print(f"{'─'*60}")

    prefix     = f"layer-{B4_LAYER}"
    weight_int = ZKLLM_WORKDIR / f"{prefix}-mlp.gate_proj.weight-int.bin"
    weight_bak = str(weight_int) + ".b4_bak"

    with tempfile.TemporaryDirectory() as tmpdir:
        act_path = Path(tmpdir) / "b4_activation.bin"
        out_path = Path(tmpdir) / "b4_ffn_out.bin"
        rng_np   = np.random.default_rng(123)
        act_data = (rng_np.standard_normal((SEQ_LEN, EMBED_DIM)) * 65536).astype(np.int32)
        act_data.tofile(str(act_path))

        def run_ffn(label: str):
            t = time.time()
            for tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                (Path(str(ZKLLM_CWD)) / tmp).unlink(missing_ok=True)
            r = subprocess.run(
                [str(ZKLLM_BIN / "ffn"),
                 str(act_path), str(SEQ_LEN), str(EMBED_DIM), str(HIDDEN_DIM),
                 str(ZKLLM_WORKDIR), prefix, str(out_path)],
                capture_output=True,
                cwd=str(ZKLLM_CWD),
            )
            return r.returncode, time.time() - t

        print(f"\n  [Step 1] 正常权重基准运行...")
        rc_before, t_before = run_ffn("baseline")
        ok_before_b4 = (rc_before == 0)
        print(f"  returncode={rc_before}  耗时 {t_before:.1f}s  → verified={ok_before_b4}")

        shutil.copy2(str(weight_int), weight_bak)
        try:
            w_orig     = np.fromfile(str(weight_int), dtype=np.int32)
            w_size     = w_orig.size
            delta      = rng_np.integers(-1 << 20, 1 << 20,
                                         size=w_size, dtype=np.int64).astype(np.int32)
            w_tampered = (w_orig.astype(np.int64) + delta.astype(np.int64)).astype(np.int32)
            w_tampered.tofile(str(weight_int))

            print(f"\n  [Step 2] 篡改 {weight_int.name}：")
            print(f"    权重元素数 = {w_size:,}")
            print(f"    篡改比例   = {float(np.mean(delta != 0))*100:.1f}%")
            print(f"    扰动幅度   = ±2^20（量化 scale 的 16×）")

            print(f"\n  [Step 3] 篡改权重后运行 FFN binary...")
            rc_after, t_after = run_ffn("tampered")
            ok_after_b4 = (rc_after == 0)
            print(f"  returncode={rc_after}  耗时 {t_after:.1f}s  → verified={ok_after_b4}")
        finally:
            shutil.copy2(weight_bak, str(weight_int))
            os.remove(weight_bak)
            print(f"\n  [恢复] 原始权重已还原")

    b4_pass = (ok_before_b4 is True) and (ok_after_b4 is False)
    print(f"\n  结果：{PASS_STR if b4_pass else FAIL_STR}  "
          f"(正常 rc={rc_before}, 篡改后 rc={rc_after})")
    print(f"  安全含义：攻击者无法在不持有与承诺匹配的原始权重的前提下通过验证。")
    results["B4"] = {"rc_before": rc_before, "rc_after": rc_after,
                     "before": bool(ok_before_b4), "after": bool(ok_after_b4),
                     "detected": b4_pass, "total": 1,
                     "detection_rate": float(b4_pass)}

# ══════════════════════════════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("实验 B 总结")
print(f"{'='*60}")
print(f"  {'实验':<6} {'攻击类型':<28} {'检测率':<14} {'结论'}")
print(f"  {'─'*58}")

attack_names = {
    "B1": "图像替换（ZAC）",
    "B2": "Embedding 替换（ZAC 跨层绑定）",
    "B3": "排名操控（Sumcheck）",
    "B4": "权重篡改（zkLLM 承诺绑定）",
}
all_pass = True
for name, r in results.items():
    det  = r["detected"] if isinstance(r["detected"], int) else int(r["detected"])
    tot  = r["total"]
    rate = r["detection_rate"]
    ok   = r.get("pass", bool(r.get("detected", False)))
    if not ok:
        all_pass = False
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<6} {attack_names[name]:<30} {det}/{tot} ({rate*100:.0f}%)   {status}")

print(f"\n  总体：{'✅ 全部通过，检索完整性保障有效' if all_pass else '❌ 存在未检测项，需排查'}")
print(f"{'='*60}\n")

# ── 序列化并保存 ──────────────────────────────────────────────────────────────
def _json_safe(obj):
    if isinstance(obj, (np.integer, np.bool_)):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

OUT_PATH.write_text(json.dumps(_json_safe(results), indent=2))
print(f"结果已保存至：{OUT_PATH}")

"""
实验 B：安全验证（B1-B4 四种攻击场景）

B1 图像替换攻击    → 被 ZAC 成员证明检测（50 个随机样本）
B2 Embedding 替换  → 被 ZAC 跨层绑定检测（50 个随机样本）
B3 排名操控攻击    → 被 Sumcheck 验证检测（10 条 query × 5 个篡改位置）
B4 权重矩阵篡改    → 被 zkLLM Sumcheck 承诺绑定检测
  攻击：篡改 layer-35 的实际权重（int.bin），保持承诺（commitment.bin）不变，
        Prover 用篡改权重计算，Verifier 用原始承诺验证 → 多线性扩展不匹配 → rc≠0

每个攻击场景期望结果：verified=False（检测率 100%）
"""

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
N_B1B2  = 50   # B1/B2 随机攻击样本数
N_B3_Q  = 10   # B3 query 数量
N_B3_V  = 5    # B3 每条 query 的篡改位置数

# zkLLM 参数（与 build_corpus_zkllm_proofs.py 保持一致）
EMBED_DIM  = 2048
HIDDEN_DIM = 11008
KV_DIM     = 256
NUM_KV_HEADS = 2
SEQ_LEN    = 1024
B4_LAYER   = 35   # 测试层（最后一层，committed weights 已存在）

PASS = "\033[32m✅ 通过\033[0m"
FAIL = "\033[31m❌ 未检测到\033[0m"

# ── 加载语料库 ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("实验 B：安全验证（B1-B4）")
print(f"{'='*60}")

corpus_entries = [json.loads(l) for l in CORPUS_JSONL.read_text().splitlines()]
embeddings     = np.load(EMBEDDING_NPY).astype(np.float32)
N, D           = embeddings.shape
all_paths      = [str(CORPUS_BASE / e["image_path"]) for e in corpus_entries]

print(f"语料库：N={N}  D={D}")

# ── 初始化 ZAC accumulator ────────────────────────────────────────────────────
print("\n[准备] 加载 ZAC prover state...")
t0 = time.time()
acc = ZACAccumulator.load_prover_state(str(PROVER_STATE))
print(f"  ZAC Root: {acc.root_hex()[:32]}…  ({time.time()-t0:.1f}s)\n")

def zac_verify_single(path: str, emb_matrix: np.ndarray, idx: int) -> bool:
    """对单张图像（path + emb_matrix[idx]）运行 ZAC 成员证明验证。"""
    element = ZACAccumulator.image_embedding_hash(path, emb_matrix[idx])
    proof   = acc.prove_membership_batch([element])
    return   acc.verify_membership_batch([element], proof)

results = {}
rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════════════
# B1：图像替换攻击（50 个随机样本）
# 攻击：将 corpus[i] 的图像文件替换为 corpus[j] 的内容，embedding 不变
# 期望：SHA256(new_bytes ∥ old_emb) ∉ BF → ZAC 拒绝
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'─'*60}")
print(f"B1 图像替换攻击（{N_B1B2} 个随机样本）")
print("攻击：替换目标图像文件内容，embedding 不变，ZAC Root 不变")
print(f"{'─'*60}")

victim_indices = rng.choice(N, size=N_B1B2, replace=False).tolist()
b1_detected = 0
b1_detail   = []
t_b1 = time.time()

for i, vic in enumerate(victim_indices):
    # 选 donor：与 victim 不同，用偏移保证多样性
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
print(f"  结果：{PASS if b1_pass else FAIL}")
results["B1"] = {"detected": b1_detected, "total": N_B1B2,
                 "detection_rate": b1_rate, "pass": b1_pass, "detail": b1_detail}

# ══════════════════════════════════════════════════════════════════════════════
# B2：Embedding 替换攻击——ZAC 跨层绑定（50 个随机样本）
# 攻击：图像文件不变，将 embedding[i] 替换为随机单位向量（模拟排名操控）
# 期望：SHA256(old_bytes ∥ fake_emb) ∉ BF → ZAC 拒绝
# ══════════════════════════════════════════════════════════════════════════════
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
print(f"  结果：{PASS if b2_pass else FAIL}")
results["B2"] = {"detected": b2_detected, "total": N_B1B2,
                 "detection_rate": b2_rate, "pass": b2_pass, "detail": b2_detail}

# ══════════════════════════════════════════════════════════════════════════════
# B3：排名操控攻击（10 条 query × 5 个篡改位置）
# 攻击：生成证明后，将某低排名条目的声明分值改为最高分 +1，试图让其进入 top-k
# 期望：Schwartz-Zippel 随机线性组合导致 batch 分值不一致 → Sumcheck 拒绝
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"B3 排名操控攻击——Sumcheck（{N_B3_Q} 条 query × {N_B3_V} 个篡改位置）")
print("攻击：生成合法证明后，篡改 scores 列表中低排名条目的声明分值")
print(f"{'─'*60}")

query_indices = [int(i * N / N_B3_Q) for i in range(N_B3_Q)]
corpus_vecs   = embeddings.tolist()

b3_detected = 0
b3_total    = N_B3_Q * N_B3_V
b3_detail   = []
t_b3 = time.time()

for qi, q_idx in enumerate(query_indices):
    q_vec = embeddings[q_idx].tolist()

    proof_ok  = prove_global_batch(q_vec, corpus_vecs)
    vr_before = verify_global_batch(q_vec, corpus_vecs, proof_ok, TOP_K)
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

        vr_after = verify_global_batch(q_vec, corpus_vecs, proof_tamper, TOP_K)
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
print(f"  结果：{PASS if b3_pass else FAIL}")
results["B3"] = {"detected": b3_detected, "total": b3_total,
                 "detection_rate": b3_rate, "pass": b3_pass, "detail": b3_detail}

# ══════════════════════════════════════════════════════════════════════════════
# B4：zkLLM 权重矩阵篡改攻击（承诺绑定）
#
# 攻击模型：
#   攻击者将服务器权重矩阵 W_gate（layer-35）替换为 W_gate + ΔW，
#   但公开承诺 commitment.bin 仍反映原始 W_gate。
#   Prover 用 W_tampered 计算 FFN，Verifier 用 commitment(W_original) 验证：
#   多线性扩展 eval(W_tampered, r) ≠ eval(W_original, r) → Sumcheck 拒绝 → rc≠0
#
# 这是 zkLLM 的核心安全属性：KZG 承诺绑定（Binding）。
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
print(f"B4 zkLLM 权重矩阵篡改攻击（承诺绑定，layer-{B4_LAYER}）")
print("攻击：篡改 gate_proj 权重 int.bin，commitment.bin 不变")
print("期望：Sumcheck 多线性扩展不匹配 → FFN binary rc≠0")
print(f"{'─'*60}")

prefix       = f"layer-{B4_LAYER}"
weight_int   = ZKLLM_WORKDIR / f"{prefix}-mlp.gate_proj.weight-int.bin"
weight_bak   = str(weight_int) + ".b4_bak"

# ── 生成实验用随机激活 ──
with tempfile.TemporaryDirectory() as tmpdir:
    act_path = Path(tmpdir) / "b4_activation.bin"
    out_path = Path(tmpdir) / "b4_ffn_out.bin"
    rng_np   = np.random.default_rng(123)
    act_data = (rng_np.standard_normal((SEQ_LEN, EMBED_DIM)) * 65536).astype(np.int32)
    act_data.tofile(str(act_path))

    def run_ffn(label: str):
        """运行 FFN binary，返回 returncode 和耗时。"""
        t = time.time()
        # 清理上次遗留的临时文件
        for tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
            (Path(str(ZKLLM_CWD)) / tmp).unlink(missing_ok=True)
        r = subprocess.run(
            [str(ZKLLM_BIN / "ffn"),
             str(act_path), str(SEQ_LEN), str(EMBED_DIM), str(HIDDEN_DIM),
             str(ZKLLM_WORKDIR), prefix, str(out_path)],
            capture_output=True,
            cwd=str(ZKLLM_CWD),
        )
        elapsed = time.time() - t
        return r.returncode, elapsed

    # ── Step 1：基准运行（正常权重）→ 期望 rc=0 ──
    print(f"\n  [Step 1] 正常权重基准运行...")
    rc_before, t_before = run_ffn("baseline")
    ok_before = (rc_before == 0)
    print(f"  returncode={rc_before}  耗时 {t_before:.1f}s  → verified={ok_before}")

    # ── Step 2：篡改权重 int.bin（大幅扰动 gate_proj 矩阵）──
    # 篡改策略：对全部权重值加 ±(2^20) 的随机扰动（约为量化 scale 2^16 的 16×）
    shutil.copy2(str(weight_int), weight_bak)
    try:
        w_orig  = np.fromfile(str(weight_int), dtype=np.int32)
        w_size  = w_orig.size
        delta   = rng_np.integers(-1 << 20, 1 << 20,
                                   size=w_size, dtype=np.int64).astype(np.int32)
        w_tampered = (w_orig.astype(np.int64) + delta.astype(np.int64)).astype(np.int32)
        w_tampered.tofile(str(weight_int))

        tamper_frac = float(np.mean(delta != 0))
        print(f"\n  [Step 2] 篡改 {weight_int.name}：")
        print(f"    权重元素数 = {w_size:,}")
        print(f"    篡改比例   = {tamper_frac*100:.1f}%")
        print(f"    扰动幅度   = ±2^20（量化 scale 的 16×）")

        # ── Step 3：用篡改权重重新运行 FFN ──
        print(f"\n  [Step 3] 篡改权重后运行 FFN binary...")
        rc_after, t_after = run_ffn("tampered")
        ok_after = (rc_after == 0)
        print(f"  returncode={rc_after}  耗时 {t_after:.1f}s  → verified={ok_after}")

    finally:
        # 无论如何恢复原始权重
        shutil.copy2(weight_bak, str(weight_int))
        os.remove(weight_bak)
        print(f"\n  [恢复] 原始权重已还原")

# 期望：篡改前 rc=0，篡改后 rc≠0
b4_pass = (ok_before is True) and (ok_after is False)
print(f"\n  结果：{PASS if b4_pass else FAIL}  "
      f"(正常 rc={rc_before}, 篡改后 rc={rc_after})")
print(f"  安全含义：攻击者无法在不持有与承诺匹配的原始权重的前提下通过验证。")
results["B4"] = {"rc_before": rc_before, "rc_after": rc_after,
                 "before": bool(ok_before), "after": bool(ok_after),
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

out_path = ROOT / "notes" / "experiment_b_result.json"
out_path.write_text(json.dumps(_json_safe(results), indent=2))
print(f"结果已保存至：{out_path}")

"""
实验 A1：端到端延迟分解

目标：测量完整验证流水线各阶段延迟，与无验证基线对比，量化每个 Phase 的额外开销。

各阶段：
  Stage 0 (baseline): jina-v4 编码 + FAISS 检索
  Stage 1: + Phase 2 Sumcheck（Global Batch，N=303）
  Stage 2: + Phase 1 ZAC 成员证明（k=5）
  Stage 3: + Phase 3C zkLLM corpus proof 读取（预计算，< 1ms）

Phase 3Q（zkLLM query proof）为后台异步，不计入同步延迟，单独报告。

实验设计：
  - 5 条代表性 query（覆盖不同语义方向）
  - 每条 query 重复 10 次，取中位数 ± IQR
  - 报告：各阶段绝对延迟 + 相对 baseline 的开销百分比
"""

import sys, json, time, statistics
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── 导入验证组件 ──────────────────────────────────────────────────────────────
from zac.accumulator import ZACAccumulator
from sumcheck.inner_product import prove_global_batch, verify_global_batch

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH     = "/root/autodl-tmp/models/jina-embeddings-v4"
GEN_MODEL_PATH = "/root/autodl-tmp/models/MiniCPM-V-4"
INDEX_PATH     = str(ROOT / "index" / "index.index")
CORPUS_JSONL   = ROOT / "corpora" / "image.jsonl"
EMBEDDING_NPY  = ROOT / "embedding" / "embedding.npy"
PROVER_STATE   = ROOT / "output" / "phase1" / "prover_state.json"
ZKLLM_WORKDIR  = ROOT / "zkllm-workdir" / "jina-v4"

TOP_K        = 5
N_REPEAT     = 10   # 每条 query 重复次数（同步阶段）
N_REPEAT_GEN = 3    # 生成阶段重复次数（每次 ~20s）

QUERIES = [
    "What is the sensor resolution of the Nikon Z8?",
    "How to set ISO sensitivity in manual exposure mode?",
    "Battery life and USB-C charging specifications",
    "AF tracking performance for 4K 60fps video recording",
    "Differences between Nikon Z8 and Z9 in weight and price",
]

print(f"\n{'='*64}")
print("实验 A1：端到端延迟分解")
print(f"{'='*64}")
print(f"  Queries : {len(QUERIES)}")
print(f"  Repeats : {N_REPEAT} 次/query（取中位数）")
print(f"  Top-K   : {TOP_K}")

# ══════════════════════════════════════════════════════════════════════════════
# 加载所有组件（只计一次，不计入 per-query 延迟）
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[加载] 模型和索引...")

# jina-v4
import torch
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cuda:0")
model.eval()
print(f"  jina-v4 loaded  ({time.time()-t0:.1f}s)")

# FAISS index
import faiss
t0 = time.time()
index = faiss.read_index(INDEX_PATH)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
print(f"  FAISS index loaded  N={index.ntotal}  ({time.time()-t0:.1f}s)")

# embedding matrix for Sumcheck
t0 = time.time()
embeddings   = np.load(EMBEDDING_NPY).astype(np.float32)
N, D         = embeddings.shape
corpus_vecs  = embeddings.tolist()   # Sumcheck 需要 Python list
corpus_paths = [json.loads(l)["image_path"] for l in CORPUS_JSONL.read_text().splitlines()]
print(f"  Embeddings loaded  N={N} D={D}  ({time.time()-t0:.1f}s)")

# ZAC
t0 = time.time()
acc = ZACAccumulator.load_prover_state(str(PROVER_STATE))
print(f"  ZAC prover state loaded  ({time.time()-t0:.1f}s)")

# corpus paths for ZAC
base_dir  = CORPUS_JSONL.parent
all_paths = [str(base_dir / p) for p in corpus_paths]

# MiniCPM-V-4（GPU 1）
from transformers import AutoTokenizer, AutoModel
t0 = time.time()
try:
    gen_model = AutoModel.from_pretrained(
        GEN_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda:1").eval()
    gen_proc = AutoTokenizer.from_pretrained(GEN_MODEL_PATH, trust_remote_code=True)
    print(f"  MiniCPM-V-4 loaded  ({time.time()-t0:.1f}s)")
except Exception as _e:
    gen_model = gen_proc = None
    print(f"  MiniCPM-V-4 加载失败（将跳过生成计时）: {_e}")

print(f"\n[组件就绪] N={N}  D={D}  ZAC Root={acc.root_hex()[:24]}…\n")

# ══════════════════════════════════════════════════════════════════════════════
# 计时辅助
# ══════════════════════════════════════════════════════════════════════════════
def median_iqr(vals):
    """返回 (中位数, IQR) ms"""
    s   = sorted(vals)
    med = statistics.median(s)
    q1  = s[len(s) // 4]
    q3  = s[3 * len(s) // 4]
    return med, q3 - q1

# ══════════════════════════════════════════════════════════════════════════════
# 逐 query 计时
# ══════════════════════════════════════════════════════════════════════════════
all_times = {
    "encode":           [],
    "faiss":            [],
    "sumcheck":         [],
    "zac":              [],
    "zkllm_corpus":     [],
}

# 保存每条 query 首次运行的 q_emb 和检索路径，供后续生成计时使用
q_embs_saved  = []
q_paths_saved = []

print(f"{'─'*64}")
print("逐 query 计时（每条重复 10 次）")
print(f"{'─'*64}")

for qi, query in enumerate(QUERIES):
    t_encode   = []
    t_faiss    = []
    t_sumcheck = []
    t_zac      = []
    t_zkllm    = []

    for rep in range(N_REPEAT):
        # ── Stage: 编码 ──
        t = time.perf_counter()
        with torch.no_grad():
            q_emb = model.encode(
                [query],
                task="retrieval",
                prompt_name="query",
                normalize_embeddings=False,
            )
        t_encode.append((time.perf_counter() - t) * 1000)
        q_vec = q_emb[0].tolist()

        # ── Stage: FAISS 检索 ──
        t = time.perf_counter()
        q_np = np.array([q_emb[0]], dtype=np.float32)
        scores_faiss, indices = index.search(q_np, TOP_K)
        t_faiss.append((time.perf_counter() - t) * 1000)
        top_k_ids    = indices[0].tolist()
        top_k_paths  = [all_paths[i] for i in top_k_ids if i >= 0]

        # 首次运行保存 q_emb 和路径，供生成计时使用
        if rep == 0:
            q_embs_saved.append(q_emb[0].copy())
            q_paths_saved.append(top_k_paths[:])

        # ── Stage: Sumcheck（Global Batch）──
        t = time.perf_counter()
        proof = prove_global_batch(q_vec, corpus_vecs)
        vr    = verify_global_batch(q_vec, corpus_vecs, proof, TOP_K)
        t_sumcheck.append((time.perf_counter() - t) * 1000)

        # ── Stage: ZAC 成员证明（k=5）──
        t = time.perf_counter()
        elements = [
            ZACAccumulator.image_embedding_hash(p, embeddings[i])
            for p, i in zip(top_k_paths, top_k_ids) if i >= 0
        ]
        zac_proof = acc.prove_membership_batch(elements)
        zac_ok    = acc.verify_membership_batch(elements, zac_proof)
        t_zac.append((time.perf_counter() - t) * 1000)

        # ── Stage: zkLLM corpus proof 读取（预计算）──
        t = time.perf_counter()
        corpus_verified = 0
        for path in top_k_paths:
            safe_id  = Path(path).name.replace("/", "_").replace("\\", "_")
            pf_path  = ZKLLM_WORKDIR / f"corpus_proof_{safe_id}.json"
            if pf_path.exists():
                pf = json.loads(pf_path.read_text())
                if pf.get("verified") and all(
                        l.get("verified") for l in pf.get("layers", [])):
                    corpus_verified += 1
        t_zkllm.append((time.perf_counter() - t) * 1000)

    # 汇总本条 query 数据
    all_times["encode"].extend(t_encode)
    all_times["faiss"].extend(t_faiss)
    all_times["sumcheck"].extend(t_sumcheck)
    all_times["zac"].extend(t_zac)
    all_times["zkllm_corpus"].extend(t_zkllm)

    med_enc, _ = median_iqr(t_encode)
    med_fai, _ = median_iqr(t_faiss)
    med_sc,  _ = median_iqr(t_sumcheck)
    med_zac, _ = median_iqr(t_zac)
    med_zkl, _ = median_iqr(t_zkllm)
    total_sync = med_enc + med_fai + med_sc + med_zac + med_zkl
    print(f"  Q{qi+1}: encode={med_enc:.0f}ms  faiss={med_fai:.1f}ms  "
          f"sumcheck={med_sc:.0f}ms  zac={med_zac:.0f}ms  "
          f"zkllm_read={med_zkl:.1f}ms  total={total_sync:.0f}ms")

# ══════════════════════════════════════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("A1 延迟分解汇总（全部 query × 10 次，中位数 ± IQR）")
print(f"{'='*64}")

stage_labels = {
    "encode":       "jina-v4 编码",
    "faiss":        "FAISS 检索",
    "sumcheck":     "Phase 2 Sumcheck",
    "zac":          "Phase 1 ZAC（k=5）",
    "zkllm_corpus": "Phase 3C corpus 读取",
}

medians = {}
print(f"\n  {'阶段':<24} {'中位数':>8} {'IQR':>8}")
print(f"  {'─'*44}")
for key, label in stage_labels.items():
    med, iqr = median_iqr(all_times[key])
    medians[key] = med
    print(f"  {label:<24} {med:>7.1f}ms {iqr:>7.1f}ms")

baseline    = medians["encode"] + medians["faiss"]
verified    = sum(medians.values())
overhead_ms    = verified - baseline
overhead_ratio = overhead_ms / baseline

print(f"\n  {'─'*44}")
print(f"  {'Baseline（编码+检索）':<24} {baseline:>7.1f}ms")
print(f"  {'完整验证流水线':<24} {verified:>7.1f}ms")
print(f"  {'验证额外开销':<24} {overhead_ms:>7.1f}ms  ({overhead_ratio:.1f}× baseline)")

# 各阶段占比
print(f"\n  各阶段占同步总延迟的比例：")
for key, label in stage_labels.items():
    pct = medians[key] / verified * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<24} {pct:>5.1f}%  {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3Q：zkLLM query proof 计时（异步后台，单独实测）
#
# 与 interactive_demo.py 完全一致：
#   K=3（layers 33-35，文本 ablation 结论：coverage 44.3% 集中在 33-35）
#   双 GPU 并行：GPU0=[33]，GPU1=[34,35]，ThreadPoolExecutor(max_workers=2)
#   wall clock ≈ max(1×T_layer, 2×T_layer) ≈ 30s（vs 串行 3×T_layer ≈ 45s）
#   跑 N_REPEAT_Q 次取中位数；同时汇总历史 K=3 数据。
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, tempfile, os, concurrent.futures as _cf
import statistics as _stat

ZKLLM_BIN    = ROOT / "src" / "zkllm" / "bin"
ZKLLM_BASE   = ROOT / "src" / "zkllm"
K_LAYERS_Q   = 3    # 与 interactive_demo.py K_LAYERS_TEXT=3 一致
EMBED_DIM_ZK = 2048
HIDDEN_DIM   = 11008
KV_DIM       = 256
NUM_KV_HEADS = 2
SEQ_LEN_ZK   = 1024
N_REPEAT_Q   = 3    # Phase 3Q 重复次数

def _ensure_worker_cwd(slot: int):
    """创建 src/zkllm/worker{slot}/，symlink swiglu-table.bin，返回 (cwd, env)。"""
    d = ZKLLM_BASE / f"worker{slot}"
    d.mkdir(exist_ok=True)
    link = d / "swiglu-table.bin"
    src  = ZKLLM_BASE / "swiglu-table.bin"
    if src.exists() and not link.exists():
        link.symlink_to(src.resolve())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot)}
    return str(d), env

def _prove_layers_q3(layer_list, cwd, env, act_dir):
    """在一个 GPU worker 上串行证明 layer_list，返回 layer_results 列表。"""
    results = []
    for li in layer_list:
        prefix   = f"layer-{li}"
        attn_inp = act_dir / f"{prefix}-attn-input.bin"
        ffn_inp  = act_dir / f"{prefix}-ffn-input.bin"
        ffn_out  = act_dir / f"{prefix}-ffn-out.bin"
        attn_out = act_dir / f"{prefix}-attn-out.bin"
        attn_sfx = act_dir / f"{prefix}-attn-sfx-out.bin"
        for tmp_f in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
            Path(cwd, tmp_f).unlink(missing_ok=True)
        r_ffn = subprocess.run(
            [str(ZKLLM_BIN / "ffn"), str(ffn_inp), str(SEQ_LEN_ZK),
             str(EMBED_DIM_ZK), str(HIDDEN_DIM),
             str(ZKLLM_WORKDIR), prefix, str(ffn_out)],
            capture_output=True, cwd=cwd, env=env)
        r_lin = subprocess.run(
            [str(ZKLLM_BIN / "self-attn"), "linear", str(attn_inp),
             str(SEQ_LEN_ZK), str(EMBED_DIM_ZK),
             str(ZKLLM_WORKDIR), prefix, str(attn_out), str(KV_DIM)],
            capture_output=True, cwd=cwd, env=env)
        if r_lin.returncode == 0:
            r_sfx = subprocess.run(
                [str(ZKLLM_BIN / "self-attn"), "attn", str(attn_inp),
                 str(SEQ_LEN_ZK), str(EMBED_DIM_ZK),
                 str(ZKLLM_WORKDIR), prefix, str(attn_sfx),
                 str(KV_DIM), str(NUM_KV_HEADS)],
                capture_output=True, cwd=cwd, env=env)
        else:
            class _F:
                returncode = 1; stderr = b""
            r_sfx = _F()
        ok = (r_ffn.returncode == 0 and r_lin.returncode == 0
              and r_sfx.returncode == 0)
        print(f"    layer-{li}: {'✓' if ok else '✗'}  "
              f"ffn={r_ffn.returncode} lin={r_lin.returncode} sfx={r_sfx.returncode}")
        results.append({"layer": li, "verified": ok,
                        "ffn_rc": r_ffn.returncode,
                        "attn_linear_rc": r_lin.returncode,
                        "attn_sfx_rc": r_sfx.returncode})
        for p in [ffn_out, attn_out, attn_sfx]:
            p.unlink(missing_ok=True)
    return results

print(f"\n{'─'*64}")
print(f"Phase 3Q：zkLLM query proof 计时（K={K_LAYERS_Q}，双GPU并行，{N_REPEAT_Q} 次）")
print(f"{'─'*64}")

cwd0, env0 = _ensure_worker_cwd(0)
cwd1, env1 = _ensure_worker_cwd(1)
all_q3_layers = list(range(36 - K_LAYERS_Q, 36))  # [33, 34, 35]
half          = len(all_q3_layers) // 2
gpu0_layers   = all_q3_layers[:half]               # [33]
gpu1_layers   = all_q3_layers[half:]               # [34, 35]
print(f"  GPU0 负责层：{gpu0_layers}  GPU1 负责层：{gpu1_layers}")

q3_elapsed_list      = []
q3_layer_results_all = []

for run_idx in range(N_REPEAT_Q):
    print(f"\n  [Run {run_idx+1}/{N_REPEAT_Q}]")
    with tempfile.TemporaryDirectory() as tmpdir:
        act_dir = Path(tmpdir)
        rng_run = np.random.default_rng(999 + run_idx)
        for li in all_q3_layers:
            (rng_run.standard_normal((SEQ_LEN_ZK, EMBED_DIM_ZK)) * 65536
             ).astype(np.int32).tofile(str(act_dir / f"layer-{li}-attn-input.bin"))
            (rng_run.standard_normal((SEQ_LEN_ZK, EMBED_DIM_ZK)) * 65536
             ).astype(np.int32).tofile(str(act_dir / f"layer-{li}-ffn-input.bin"))

        t_run = time.perf_counter()
        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            fut0 = ex.submit(_prove_layers_q3, gpu0_layers, cwd0, env0, act_dir)
            fut1 = ex.submit(_prove_layers_q3, gpu1_layers, cwd1, env1, act_dir)
            res0, res1 = fut0.result(), fut1.result()
        elapsed = round((time.perf_counter() - t_run) * 1000)

        layer_results_run = sorted(res0 + res1, key=lambda r: r["layer"])
        verified_run = all(r["verified"] for r in layer_results_run)
        q3_elapsed_list.append(elapsed)
        q3_layer_results_all.append(layer_results_run)
        print(f"  → elapsed={elapsed}ms ({elapsed/1000:.1f}s)  verified={verified_run}")

q3_median_ms = round(_stat.median(q3_elapsed_list))
q3_verified  = all(r["verified"] for run in q3_layer_results_all for r in run)
print(f"\n  Phase 3Q 中位数：{q3_median_ms}ms ({q3_median_ms/1000:.1f}s)  "
      f"n={N_REPEAT_Q}  verified={q3_verified}")
print(f"  注：异步后台执行，不阻塞用户响应。")

# 历史 K=3 数据
hist_k3 = []
for pf in sorted(ZKLLM_WORKDIR.glob("zkllm_proof_*.json")):
    d_pf = json.loads(pf.read_text())
    if d_pf.get("k_layers") == K_LAYERS_Q and d_pf.get("elapsed_ms"):
        hist_k3.append(d_pf["elapsed_ms"])
if hist_k3:
    print(f"  历史 K=3 参考（n={len(hist_k3)}）：中位数 {round(_stat.median(hist_k3))}ms  "
          f"范围 {min(hist_k3)}–{max(hist_k3)}ms")

# ══════════════════════════════════════════════════════════════════════════════
# Stage: 大模型生成计时
# 与 interactive_demo.py run_generation 完全一致：最多取前 3 张图像
# N_REPEAT_GEN 次取中位数；因为 Phase 3Q 在等待结束后才调用生成，所以
# 生成时间直接加到 effective_wait 后面即可得到完整端到端延迟。
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*64}")
print(f"Stage 生成：MiniCPM-V-4 推理计时（{N_REPEAT_GEN} 次/query）")
print(f"{'─'*64}")

gen_times_all = []
gen_median_ms = None

if gen_model is not None:
    from PIL import Image as _PIL_Image
    for qi, (query, q_paths) in enumerate(zip(QUERIES, q_paths_saved)):
        imgs = [_PIL_Image.open(p).convert("RGB")
                for p in q_paths[:3] if p and Path(p).exists()]
        prompt_text = (
            "Please answer the following question based on the provided images. "
            "Answer in Chinese.\n\nQuestion: " + query
        )
        msgs = [{"role": "user", "content": imgs + [prompt_text]}]
        t_gen_list = []
        for _ in range(N_REPEAT_GEN):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = gen_model.chat(image=None, msgs=msgs, tokenizer=gen_proc)
            t_gen_list.append((time.perf_counter() - t0) * 1000)
        gen_times_all.extend(t_gen_list)
        med_g, _ = median_iqr(t_gen_list)
        print(f"  Q{qi+1}: 中位数={med_g:.0f}ms  ({med_g/1000:.1f}s)")
    gen_median_ms = round(_stat.median(gen_times_all))
    print(f"\n  生成中位数（全部 query）：{gen_median_ms}ms ({gen_median_ms/1000:.1f}s)")
else:
    gen_median_ms = None
    print("  MiniCPM-V-4 未加载，跳过生成计时")

# ══════════════════════════════════════════════════════════════════════════════
# 建库开销（一次性，离线）
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("建库开销（一次性，离线）")
print(f"{'='*64}")

# Phase 3C：从已有 corpus_proof_*.json 统计
corpus_proof_files = sorted(ZKLLM_WORKDIR.glob("corpus_proof_*.json"))
corpus_elapsed_ms  = []
corpus_k_val       = None
for pf in corpus_proof_files:
    d_cp = json.loads(pf.read_text())
    if d_cp.get("elapsed_ms"):
        corpus_elapsed_ms.append(d_cp["elapsed_ms"])
    if corpus_k_val is None:
        corpus_k_val = d_cp.get("k_layers")
if corpus_elapsed_ms:
    corpus_elapsed_ms.sort()
    c_n      = len(corpus_elapsed_ms)
    c_med    = round(_stat.median(corpus_elapsed_ms))
    c_total  = sum(corpus_elapsed_ms)
    print(f"  Phase 3C corpus zkLLM（K={corpus_k_val}，layers 31-35）")
    print(f"    N={c_n} 张  中位数={c_med}ms/张  总计={c_total/1000:.0f}s"
          f" ({c_total/3600000:.2f}h)")
else:
    c_n, c_med, c_total, corpus_k_val = 0, None, None, None
    print("  Phase 3C corpus proof 文件不存在")

print(f"  Phase 1 ZAC 建库（N=303）：详见 A2 实验")
print(f"  PDF→images + embedding + FAISS 索引：通常 <10min（N=303）")

# ══════════════════════════════════════════════════════════════════════════════
# 完整开销对比汇总
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("完整开销对比汇总")
print(f"{'='*64}")

print(f"\n  ── 查询时开销（per-query，每次查询） ──")
print(f"  {'阶段':<30} {'延迟':>9}  {'类型'}")
print(f"  {'─'*54}")
sync_rows = [
    ("jina-v4 编码",           medians["encode"]),
    ("FAISS 检索",              medians["faiss"]),
    ("Phase 2 Sumcheck",        medians["sumcheck"]),
    ("Phase 1 ZAC（k=5）",     medians["zac"]),
    ("Phase 3C corpus 读取",    medians["zkllm_corpus"]),
]
for label, val in sync_rows:
    print(f"  {label:<30} {val:>7.0f}ms  同步")
print(f"  {'─'*54}")
# 关键时序：Phase 3Q 在 encode 完成后才开始计算；
# 主线程在 Step 6 join() 处等待 Phase 3Q，所以从查询开始到可以生成的时刻为：
#   effective_wait = max(同步总计, encode + Phase3Q_时长)
# 因 encode 已含在同步总计中，等效于：
#   effective_wait = max(verified, medians["encode"] + q3_median_ms)
effective_wait_ms = max(verified, medians["encode"] + q3_median_ms)
phase3q_wait_ms   = effective_wait_ms - verified   # 主线程实际等待 Phase 3Q 的时间

print(f"  {'同步验证完成时刻':<30} {verified:>7.0f}ms")
print(f"  {'Phase 3Q（K=3，双GPU）完成时刻':<30} {medians['encode'] + q3_median_ms:>7.0f}ms  异步")
print(f"  {'─'*54}")
print(f"  {'大模型可开始生成的时刻':<30} {effective_wait_ms:>7.0f}ms  （两者取 max）")
print(f"  {'其中主线程 join 等待':<30} {phase3q_wait_ms:>7.0f}ms  Phase 3Q 阻塞")
if gen_median_ms:
    print(f"  {'大模型生成':<30} {gen_median_ms:>7.0f}ms")
    total_e2e_ms      = effective_wait_ms + gen_median_ms
    baseline_e2e_ms   = baseline + gen_median_ms
    print(f"  {'─'*54}")
    print(f"  {'端到端总计（含生成）':<30} {total_e2e_ms:>7.0f}ms")
    print(f"  {'Baseline 端到端（含生成）':<30} {baseline_e2e_ms:>7.0f}ms")
    e2e_overhead_ms    = total_e2e_ms - baseline_e2e_ms
    e2e_overhead_ratio = e2e_overhead_ms / baseline_e2e_ms
    print(f"  端到端额外开销：{e2e_overhead_ms:.0f}ms = {e2e_overhead_ratio:.1f}× baseline")
else:
    total_e2e_ms = baseline_e2e_ms = e2e_overhead_ms = e2e_overhead_ratio = None
    print(f"  {'─'*54}")
    print(f"  {'大模型可开始生成的时刻':<30} {effective_wait_ms:>7.0f}ms")
    print(f"  {'Baseline（无验证）':<30} {baseline:>7.0f}ms")
    e2e_overhead_ratio_ = (effective_wait_ms - baseline) / baseline
    print(f"  等待生成额外开销：{effective_wait_ms-baseline:.0f}ms = {e2e_overhead_ratio_:.1f}× baseline")

if corpus_elapsed_ms:
    print(f"\n  ── 建库时开销（一次性，离线） ──")
    print(f"  Phase 3C corpus zkLLM（K={corpus_k_val}，N={c_n} 张）：~{c_total/3600000:.2f}h")
    print(f"  Phase 1 ZAC 建库 + embedding + FAISS：<10min（见 A2 实验）")

# ══════════════════════════════════════════════════════════════════════════════
# 保存结果
# ══════════════════════════════════════════════════════════════════════════════
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

output = {
    "config": {
        "n_queries":   len(QUERIES),
        "n_repeat":    N_REPEAT,
        "top_k":       TOP_K,
        "N":           N,
        "D":           D,
        "phase3q_k":   K_LAYERS_Q,
        "phase3q_runs": N_REPEAT_Q,
    },
    "medians_ms": {k: round(v, 2) for k, v in medians.items()},
    "iqr_ms":     {k: round(median_iqr(all_times[k])[1], 2) for k in all_times},
    "raw_ms":     {k: [round(x, 2) for x in v] for k, v in all_times.items()},
    "phase3q": {
        "k_layers":        K_LAYERS_Q,
        "gpu0_layers":     gpu0_layers,
        "gpu1_layers":     gpu1_layers,
        "elapsed_ms_list": q3_elapsed_list,
        "median_ms":       q3_median_ms,
        "verified":        q3_verified,
        "hist_k3_n":       len(hist_k3),
        "hist_k3_median":  round(_stat.median(hist_k3)) if hist_k3 else None,
        "note":            "双GPU并行，异步后台执行，不阻塞用户响应",
    },
    "build_overhead": {
        "phase3c_n_images":    c_n,
        "phase3c_k_layers":    corpus_k_val,
        "phase3c_median_ms":   c_med,
        "phase3c_total_ms":    c_total,
        "phase3c_total_hours": round(c_total / 3600000, 3) if c_total else None,
    },
    "generation": {
        "median_ms":   gen_median_ms,
        "n_repeat":    N_REPEAT_GEN,
        "raw_ms":      [round(x, 1) for x in gen_times_all] if gen_model else [],
    },
    "summary": {
        "baseline_ms":            round(baseline, 2),
        "sync_verified_ms":       round(verified, 2),
        "sync_overhead_ms":       round(overhead_ms, 2),
        "sync_overhead_ratio":    round(overhead_ratio, 3),
        "phase3q_median_ms":      q3_median_ms,
        "effective_wait_ms":      round(effective_wait_ms, 2),
        "phase3q_join_wait_ms":   round(phase3q_wait_ms, 2),
        "gen_median_ms":          gen_median_ms,
        "e2e_total_ms":           round(total_e2e_ms) if total_e2e_ms else None,
        "e2e_baseline_ms":        round(baseline_e2e_ms) if baseline_e2e_ms else None,
        "e2e_overhead_ratio":     round(e2e_overhead_ratio, 3) if e2e_overhead_ratio else None,
        "note": "effective_wait=max(sync_verified, encode+phase3q); e2e=effective_wait+gen",
    },
}

out_path = ROOT / "notes" / "experiment_a1_result.json"
out_path.write_text(json.dumps(_json_safe(output), indent=2))
print(f"\n结果已保存至：{out_path}")

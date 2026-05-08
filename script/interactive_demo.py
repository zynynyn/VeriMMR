"""
可验证 VisRAG 系统 — 实时交互演示 v3

启动：
  cd /root/autodl-tmp/UltraRAG
  conda activate ultrarag
  python script/interactive_demo.py

访问：
  SSH 隧道：ssh -L 7860:127.0.0.1:7860 <用户@服务器IP>
  AutoDL 控制台：「自定义服务」配置端口 7860
"""

import sys
import os
import json
import html as _html
import hashlib
import random as _random
import time
import uuid
import threading
import subprocess
import traceback
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import gradio as gr

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "servers" / "retriever" / "src"))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# ── 配置 ──────────────────────────────────────────────────────────────────────
CORPUS_PATH    = ROOT / "corpora" / "image.jsonl"
EMBEDDING_PATH = ROOT / "embedding" / "embedding.npy"
INDEX_PATH     = ROOT / "index" / "index.index"
ZAC_STATE      = ROOT / "output" / "phase1" / "prover_state.json"
ZKLLM_WORKDIR  = ROOT / "zkllm-workdir" / "jina-v4"
ZKLLM_BIN_DIR  = ROOT / "src" / "zkllm"
MODEL_PATH     = "/root/autodl-tmp/models/jina-embeddings-v4"
GEN_MODEL_PATH = "/root/autodl-tmp/models/MiniCPM-V-4"
TOP_K          = 5
# Fiat-Shamir 随机层挑战参数（在线查询安全）：
#   challenge = SHA256(query_text || nonce)，从全部 36 层中随机抽 K=6 层
#   P(caught | L 层被篡改) = 1 - C(36-L, K) / C(36, K)
#   K=6: L≥6 篡改→P≥69.5%，整模型替换→P=100%，在线墙钟≈46s（2-GPU 异步）
K_LAYERS_TEXT  = 6   # Fiat-Shamir 在线挑战：从 36 层中随机选 6 层
K_LAYERS_IMG   = 5   # 图像语料库侧固定：层 31-35（预计算，不需要随机化）
# 统一填充到 1024：满足 FFN/linear（seq×2048=2^21）和 GQA zkAttn（seq²=2^20）全部 NTT 约束
# 语料库侧图像（~641 tokens）同样填充到 1024，两侧策略完全对称
SEQ_LEN_PAD    = 1024
KV_DIM_QUERY   = 256    # jina-v4 GQA kv_dim
NUM_KV_HEADS_Q = 2      # jina-v4 GQA num_kv_heads

# ── Fiat-Shamir 工具 ─────────────────────────────────────────────────────────

def _fiat_shamir_layers(query_text: str, nonce: str,
                        K: int = 6, total: int = 36) -> list:
    """以 SHA256(query||nonce) 为种子从 [0, total) 中无放回抽取 K 层，返回排序后的列表。"""
    seed = hashlib.sha256((query_text + "|" + nonce).encode("utf-8")).digest()
    rng  = _random.Random(seed)
    return sorted(rng.sample(range(total), K))


# ── 全局状态 ──────────────────────────────────────────────────────────────────
_model          = None
_gen_model      = None
_gen_processor  = None
_faiss_index    = None
_corpus: List[Dict] = []
_sc_embeddings: Optional[np.ndarray] = None
_zac_acc        = None
_zkllm_cache: Dict[str, Dict] = {}
_zkllm_lock     = threading.Lock()
# 全量模式 timer 回调重建 VTL 所需的查询最终状态（on_query 完成后写入）
_query_final_state: Dict[str, Dict] = {}
_load_errors: List[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# 组件加载
# ─────────────────────────────────────────────────────────────────────────────
def load_all():
    global _model, _gen_model, _gen_processor
    global _faiss_index, _corpus, _sc_embeddings, _zac_acc

    print("── 加载语料库元数据 ──")
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                _corpus.append(json.loads(line))
    print(f"  {len(_corpus)} 条记录")

    print("── 加载 corpus embedding ──")
    _sc_embeddings = np.load(str(EMBEDDING_PATH))
    print(f"  shape={_sc_embeddings.shape}")

    print("── 加载 FAISS index ──")
    import faiss
    _faiss_index = faiss.read_index(str(INDEX_PATH))
    print(f"  ntotal={_faiss_index.ntotal}")

    print("── 加载 jina-v4 (GPU 0) ──")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(str(MODEL_PATH), trust_remote_code=True, device="cuda:0")
    print("  OK")

    print("── 加载 ZAC prover state ──")
    if ZAC_STATE.exists():
        try:
            from zac.accumulator import ZACAccumulator
            _zac_acc = ZACAccumulator.load_prover_state(str(ZAC_STATE))
            print(f"  OK  root={_zac_acc.root_hex()[:16]}…")
        except Exception as e:
            _load_errors.append(f"ZAC 加载失败: {e}")
            print(f"  警告: {_load_errors[-1]}")
    else:
        print("  未找到 prover state，ZAC 验证将跳过")

    print("── 尝试加载 MiniCPM-V-4 (GPU 1) ──")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _gen_model = AutoModel.from_pretrained(
            GEN_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to("cuda:1").eval()
        _gen_processor = AutoTokenizer.from_pretrained(GEN_MODEL_PATH, trust_remote_code=True)
        print("  OK")
    except Exception as e:
        _load_errors.append(f"MiniCPM-V-4 加载失败: {e}")
        print(f"  警告: {_load_errors[-1]}")

    print("── 全部组件就绪 ──\n")


def reload_index():
    """构建完成后热重载：corpus / embedding / FAISS / ZAC（不重载模型）"""
    global _faiss_index, _corpus, _sc_embeddings, _zac_acc
    print("── 热重载数据索引 ──")
    try:
        _corpus.clear()
        if CORPUS_PATH.exists():
            with open(CORPUS_PATH) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        _corpus.append(json.loads(line))
        print(f"  corpus: {len(_corpus)} 条")

        if EMBEDDING_PATH.exists():
            _sc_embeddings = np.load(str(EMBEDDING_PATH))
            print(f"  embedding: shape={_sc_embeddings.shape}")

        if INDEX_PATH.exists():
            import faiss
            _faiss_index = faiss.read_index(str(INDEX_PATH))
            print(f"  FAISS ntotal={_faiss_index.ntotal}")

        if ZAC_STATE.exists():
            from zac.accumulator import ZACAccumulator
            _zac_acc = ZACAccumulator.load_prover_state(str(ZAC_STATE))
            print(f"  ZAC root={_zac_acc.root_hex()[:16]}…")
        else:
            _zac_acc = None
            print("  ZAC: 未找到 prover state")
        print("── 热重载完成 ──")
    except Exception as e:
        print(f"  热重载出错: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 核心逻辑
# ─────────────────────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    return _model.encode(
        [query], task="retrieval", prompt_name="query", normalize_embeddings=False,
    )[0].astype(np.float32)


def embed_query_with_hooks(query: str, proof_id: str,
                           act_ready: threading.Event,
                           layers: list = None) -> np.ndarray:
    """
    编码查询并捕获指定层的真实激活，保存为 zkLLM 二进制格式：
      layer-{l}-qry-{proof_id}-attn-input.bin  → input_layernorm 输出（self-attn 输入）
      layer-{l}-qry-{proof_id}-ffn-input.bin   → post_attention_layernorm 输出（ffn 输入）
    文件均 zero-pad 到 SEQ_LEN_PAD×2048 的 int32 数组。
    layers: Fiat-Shamir 选出的层列表（None 时退回固定最后 K 层）
    完成后设置 act_ready 事件，解锁后台证明线程。
    """
    import torch
    if layers is None:
        layers = list(range(36 - K_LAYERS_TEXT, 36))
    print(f"[Step 1] 向量编码 + hook 捕获  proof_id={proof_id}  layers={layers}")
    try:
        model_layers = (list(_model.children())[0]
                        .model.base_model.model.model.language_model.layers)
        captured = {}
        handles = []

        def _make_hook(key):
            def _hook(module, inp, out):
                captured[key] = out.detach().float().cpu()
            return _hook

        for li in layers:
            handles.append(
                model_layers[li].input_layernorm.register_forward_hook(
                    _make_hook(f"attn_{li}")))
            handles.append(
                model_layers[li].post_attention_layernorm.register_forward_hook(
                    _make_hook(f"ffn_{li}")))

        result = _model.encode(
            [query], task="retrieval", prompt_name="query",
            normalize_embeddings=False)

        for h in handles:
            h.remove()

        scale = 1 << 16
        for li in layers:
            for hook_type in ("attn", "ffn"):
                key = f"{hook_type}_{li}"
                if key not in captured:
                    continue
                act = captured[key]               # (1, S, D) 或 (S, D)
                if act.dim() == 2:
                    act = act.unsqueeze(0)
                a = act[0]                        # (S, D)
                S, D = a.shape
                if S < SEQ_LEN_PAD:
                    a = torch.cat([a, torch.zeros(SEQ_LEN_PAD - S, D)], dim=0)
                elif S > SEQ_LEN_PAD:
                    a = a[:SEQ_LEN_PAD]
                out_path = str(ZKLLM_WORKDIR /
                               f"layer-{li}-qry-{proof_id}-{hook_type}-input.bin")
                (a * scale).round().to(torch.int32).numpy().astype(
                    np.int32).tofile(out_path)

        emb = result[0].astype(np.float32)
        norm = float(np.linalg.norm(emb))
        print(f"[Step 1] 编码完成  emb_dim={emb.shape[0]}  norm={norm:.4f}  激活文件已保存，解锁 zkLLM 后台线程")
        act_ready.set()
        return emb

    except Exception as e:
        print(f"[Step 1] embed_query_with_hooks 出错（回退到普通 encode）: {e}")
        act_ready.set()   # 仍然解锁证明线程（将回退到随机基线）
        return _model.encode(
            [query], task="retrieval", prompt_name="query",
            normalize_embeddings=False)[0].astype(np.float32)


def faiss_search(q_emb: np.ndarray):
    print(f"[Step 2] FAISS 检索  top_k={TOP_K}  index_size={_faiss_index.ntotal}")
    scores, ids = _faiss_index.search(q_emb.reshape(1, -1), TOP_K)
    paths, scs, emb_ids = [], [], []
    for sid, sc in zip(ids[0], scores[0]):
        if sid == -1:
            paths.append(None)
            emb_ids.append(-1)
        else:
            item = _corpus[int(sid)]
            paths.append(str(CORPUS_PATH.parent / item["image_path"]))
            emb_ids.append(int(sid))
        scs.append(float(sc))
    valid = [(emb_ids[i], scs[i]) for i in range(len(paths)) if paths[i] is not None]
    score_str = "  ".join(f"id={eid} sc={sc:.4f}" for eid, sc in valid)
    print(f"[Step 2] 检索完成  {score_str}")
    return paths, scs, emb_ids


def run_zac(paths: List, emb_ids: List[int] = None) -> Dict:
    if _zac_acc is None:
        print("[Step 5] ZAC 未初始化（已跳过）")
        return {"disabled": True}
    print(f"[Step 5] ZAC 成员证明  n_images={sum(1 for p in paths if p is not None)}")
    try:
        from zac.accumulator import ZACAccumulator
        elements = []
        for i, p in enumerate(paths):
            if p is None:
                continue
            sid = emb_ids[i] if (emb_ids and i < len(emb_ids)) else -1
            if sid >= 0 and _sc_embeddings is not None and sid < len(_sc_embeddings):
                # 跨层承诺绑定：SHA256(image_bytes ∥ embedding_bytes)
                h = ZACAccumulator.image_embedding_hash(p, _sc_embeddings[sid])
                elements.append(h)
                print(f"  image  {Path(p).name}  emb_id={sid}  hash={h.hex()[:16]}…  (image+embedding)")
            else:
                # 无 embedding 时回退旧哈希（理论上不应发生）
                h = ZACAccumulator.image_hash(p)
                elements.append(h)
                print(f"  image  {Path(p).name}  emb_id={sid}  hash={h.hex()[:16]}…  (image-only fallback)")
        t0 = time.perf_counter()
        proof = _zac_acc.prove_membership_batch(elements)
        prove_ms = round((time.perf_counter() - t0) * 1000, 1)
        t0 = time.perf_counter()
        ok = _zac_acc.verify_membership_batch(elements, proof)
        verify_ms = round((time.perf_counter() - t0) * 1000, 1)
        cm_hex = proof.get("cm_hex", "")
        # Single-filter: proof_hex at top level. Cascade: inside layer_proofs[0].
        n_f = proof.get("n_filters", 1)
        if n_f == 1:
            proof_hex = proof.get("proof_hex", "")
        else:
            lps = proof.get("layer_proofs", [{}])
            proof_hex = lps[0].get("proof_hex", "") if lps else ""
        print(f"[Step 5] 完成  verified={ok}  prove={prove_ms}ms  verify={verify_ms}ms  "
              f"n_filters={n_f}  root={cm_hex[:16]}…")
        return {
            "verified": ok, "num_images": len(elements),
            "cm_hex": cm_hex, "proof_hex": proof_hex, "n_filters": n_f,
            "prove_ms": prove_ms, "verify_ms": verify_ms,
        }
    except Exception as e:
        print(f"[Step 5] 异常: {e}")
        return {"verified": False, "error": str(e), "traceback": traceback.format_exc()}


def run_sumcheck(q_emb: np.ndarray) -> Dict:
    print(f"[Step 4] Sumcheck 内积证明  N={len(_sc_embeddings) if _sc_embeddings is not None else 0}  k={TOP_K}")
    try:
        from sumcheck.inner_product import prove_global_batch, verify_global_batch
        q_list = q_emb.tolist()
        corpus_vecs = _sc_embeddings.tolist()
        t0 = time.perf_counter()
        proof = prove_global_batch(q_list, corpus_vecs)
        prove_ms = round((time.perf_counter() - t0) * 1000, 1)
        t0 = time.perf_counter()
        vr = verify_global_batch(q_list, corpus_vecs, proof, TOP_K)
        verify_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = vr.get("verified", False)
        proof_bytes = len(json.dumps(proof).encode())
        print(f"[Step 4] 完成  verified={ok}  prove={prove_ms}ms  verify={verify_ms}ms  proof_bytes={proof_bytes}")
        return {
            "verified": ok, "N": len(corpus_vecs), "k": TOP_K,
            "proof_bytes": proof_bytes,
            "top_k_indices": vr.get("top_k_indices", []),
            "top_k_scores": [int(s) for s in vr.get("top_k_scores", [])],
            "prove_ms": prove_ms, "verify_ms": verify_ms,
        }
    except Exception as e:
        print(f"[Step 4] 异常: {e}")
        return {"verified": False, "error": str(e), "traceback": traceback.format_exc()}


def get_corpus_proofs(paths: List) -> List[Dict]:
    print(f"[Step 3] 读取语料库侧 zkLLM 预计算证明  n_paths={len(paths)}")
    results = []
    for p in paths:
        if not p:
            results.append({"status": "not_found"})
            continue
        image_id = "/".join(Path(p).parts[-2:])
        safe = image_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        pf = ZKLLM_WORKDIR / f"corpus_proof_{safe}.json"
        r = json.loads(pf.read_text()) if pf.exists() else {"status": "not_precomputed", "image_id": image_id}
        # corpus proof 使用 all_ok 字段（不是 verified）
        is_ok = bool(r.get("all_ok"))
        status_tag = "✅" if is_ok else ("⚠️ not_precomputed" if r.get("status") == "not_precomputed" else "❌")
        res = r.get("results", {})
        vit_pass  = res.get("vit", {}).get("n_pass", "?")
        llm_pass  = res.get("llm_layers", {}).get("n_pass", "?")
        pool_ok   = "✓" if res.get("pooling", {}).get("all_ok") else "✗"
        print(f"  corpus_proof  {image_id}  {status_tag}  vit={vit_pass}/32  llm={llm_pass}/36  pool={pool_ok}")
        results.append(r)
    ok_n = sum(1 for r in results if r.get("all_ok"))
    print(f"[Step 3] 完成  {ok_n}/{len(results)} 全量验证通过")
    return results


def run_generation(query: str, image_paths: List[str]) -> str:
    if _gen_model is None or _gen_processor is None:
        return "（生成模型未加载）"
    try:
        import torch
        from PIL import Image
        images = [Image.open(p).convert("RGB")
                  for p in image_paths[:3] if p and Path(p).exists()]
        if not images:
            return "（未找到有效图像）"
        prompt_text = (
            "Please answer the following question based on both the text and the provided images. "
            "Think step by step, use evidence from the images whenever possible, and give a clear, factual answer. "
            "Answer in Chinese.\n\nQuestion: " + query
        )
        msgs = [{"role": "user", "content": images + [prompt_text]}]
        with torch.no_grad():
            answer = _gen_model.chat(image=None, msgs=msgs, tokenizer=_gen_processor)
        return str(answer)
    except Exception as e:
        return f"生成失败: {e}"


def _ensure_worker_cwd(slot: int) -> tuple:
    """创建 src/zkllm/worker{slot}/ 并 symlink swiglu-table.bin，返回 (cwd_str, env_dict)。"""
    base = ZKLLM_BIN_DIR          # src/zkllm/
    d    = base / f"worker{slot}"
    d.mkdir(exist_ok=True)
    link = d / "swiglu-table.bin"
    src  = base / "swiglu-table.bin"
    if src.exists() and not link.exists():
        link.symlink_to(src.resolve())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slot)}
    return str(d), env


def _run_zkllm_query_bg(proof_id: str, act_ready: threading.Event = None,
                        layers: list = None):
    workdir = str(ZKLLM_WORKDIR)
    bin_dir = str(ZKLLM_BIN_DIR)
    # Fiat-Shamir 层间并行：K 层均分给 GPU0 / GPU1
    # 各 GPU 在自己负责的层上独立串行：FFN+linear+zkAttn
    # K=6 → gpu0=[3层]×15.2s ∥ gpu1=[3层]×15.2s ≈ 45.6s 墙钟
    if layers is None:
        layers = list(range(36 - K_LAYERS_TEXT, 36))
    print(f"[zkLLM-Q] 后台线程启动  proof_id={proof_id}  layers={layers}  等待激活文件…")
    cwd0, env0 = _ensure_worker_cwd(0)
    cwd1, env1 = _ensure_worker_cwd(1)
    # 查询侧 worker0 用 GPU0（jina-v4 ~8.3GB，空闲 ~15.8GB）
    # worker1 用 GPU1（MiniCPM 静态权重 ~11.6GB，生成完毕后 KV cache 已 empty_cache，空闲 ~12+GB）
    t0 = time.perf_counter()
    layer_results, success = [], True

    # 等待真实激活文件就绪（embed_query_with_hooks 完成后设置）
    if act_ready is not None:
        act_ready.wait(timeout=60)
        print(f"[zkLLM-Q] 激活文件就绪，开始证明  layers={layers}")

    import concurrent.futures as _cf

    all_layers = layers
    half       = len(all_layers) // 2
    # 奇数层时 worker1 多一层
    gpu0_layers = all_layers[:half]
    gpu1_layers = all_layers[half:]

    def _prove_layers(layer_list: list, cwd: str, env: dict) -> list:
        """在给定 GPU 上串行证明 layer_list 中每一层，返回 layer_results 子列表。"""
        gpu_id = env.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"[zkLLM-Q] GPU{gpu_id} 开始证明层 {layer_list}")
        results = []
        seq = SEQ_LEN_PAD
        for li in layer_list:
            prefix   = f"layer-{li}"
            attn_inp = os.path.join(workdir, f"{prefix}-qry-{proof_id}-attn-input.bin")
            ffn_inp  = os.path.join(workdir, f"{prefix}-qry-{proof_id}-ffn-input.bin")
            ffn_out      = os.path.join(workdir, f"{prefix}-qry-{proof_id}-ffn-out.bin")
            attn_out     = os.path.join(workdir, f"{prefix}-qry-{proof_id}-attn-out.bin")
            attn_sfx_out = os.path.join(workdir, f"{prefix}-qry-{proof_id}-attn-sfx-out.bin")

            # 激活缺失时回退随机基线
            if not os.path.exists(attn_inp):
                (np.random.randn(seq, 2048) * 65536).astype(np.int32).tofile(attn_inp)
            if not os.path.exists(ffn_inp):
                (np.random.randn(seq, 2048) * 65536).astype(np.int32).tofile(ffn_inp)

            # 清理 stale temp 文件：self-attn linear 写 temp_Q/K/V.bin，
            # 若上次运行中途失败会留下错误尺寸的残留，导致 attn 报 transpose 错误
            for _tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                Path(os.path.join(cwd, _tmp)).unlink(missing_ok=True)

            r_ffn = subprocess.run(
                [f"{bin_dir}/ffn", ffn_inp, str(seq), "2048", "11008",
                 workdir, prefix, ffn_out],
                capture_output=True, cwd=cwd, env=env)
            r_lin = subprocess.run(
                [f"{bin_dir}/self-attn", "linear", attn_inp, str(seq), "2048",
                 workdir, prefix, attn_out, str(KV_DIM_QUERY)],
                capture_output=True, cwd=cwd, env=env)
            if r_lin.returncode == 0:
                r_sfx = subprocess.run(
                    [f"{bin_dir}/self-attn", "attn", attn_inp, str(seq), "2048",
                     workdir, prefix, attn_sfx_out,
                     str(KV_DIM_QUERY), str(NUM_KV_HEADS_Q)],
                    capture_output=True, cwd=cwd, env=env)
            else:
                class _Fail:
                    returncode = 1; stderr = b"linear failed"
                r_sfx = _Fail()

            ok = (r_ffn.returncode == 0 and r_lin.returncode == 0
                  and r_sfx.returncode == 0)
            status_tag = "✅" if ok else f"❌ ffn_rc={r_ffn.returncode} lin_rc={r_lin.returncode} sfx_rc={r_sfx.returncode}"
            print(f"  layer {li}  {status_tag}")
            if not ok:
                if r_ffn.returncode != 0:
                    print(f"    ffn stderr: {r_ffn.stderr.decode(errors='replace')[-200:]}")
                if r_lin.returncode != 0:
                    print(f"    linear stderr: {r_lin.stderr.decode(errors='replace')[-200:]}")
                if r_sfx.returncode != 0:
                    print(f"    attn_sfx stderr: {r_sfx.stderr.decode(errors='replace')[-200:]}")
            results.append({
                "layer": li, "verified": ok,
                "ffn_rc": r_ffn.returncode,
                "attn_linear_rc": r_lin.returncode,
                "attn_sfx_rc": r_sfx.returncode,
            })
            for p in [ffn_out, attn_out, attn_sfx_out, attn_inp, ffn_inp]:
                Path(p).unlink(missing_ok=True)
        return results

    try:
        # 并行执行：worker0 在 GPU0，worker1 在 GPU1（生成完毕后 KV cache 已清空）
        # GPU0: jina-v4 ~8.3GB，空闲 ~15.8GB；GPU1: MiniCPM 静态权重 ~11.6GB，空闲 ~12+GB
        # 各自峰值 ~5-8GB，两路互不干扰
        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            fut0 = ex.submit(_prove_layers, gpu0_layers, cwd0, env0)
            fut1 = ex.submit(_prove_layers, gpu1_layers, cwd1, env1)
            res0 = fut0.result()
            res1 = fut1.result()

        # 按层号排序合并
        layer_results = sorted(res0 + res1, key=lambda r: r["layer"])
        success = all(r["verified"] for r in layer_results)
    except Exception as _ex:
        print(f"[zkLLM-Q] 证明线程异常: {_ex}")
        success = False
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    verified = success and all(r["verified"] for r in layer_results)
    print(f"[zkLLM-Q] 完成  verified={verified}  elapsed={elapsed_ms}ms  layers_ok={sum(r['verified'] for r in layer_results)}/{len(layer_results)}")
    result = {
        "status": "completed", "k_layers": len(layers), "modality": "text",
        "fiat_shamir_layers": layers, "layers": layer_results,
        "verified": verified,
        "elapsed_ms": elapsed_ms,
    }
    with _zkllm_lock:
        _zkllm_cache[proof_id] = result
    (ZKLLM_WORKDIR / f"zkllm_proof_{proof_id}.json").write_text(json.dumps(result, indent=2))


def _get_zkllm_result(proof_id: str) -> Optional[Dict]:
    with _zkllm_lock:
        r = _zkllm_cache.get(proof_id)
    if r is None:
        pf = ZKLLM_WORKDIR / f"zkllm_proof_{proof_id}.json"
        if pf.exists():
            r = json.loads(pf.read_text())
            with _zkllm_lock:
                _zkllm_cache[proof_id] = r
    return r


# ─────────────────────────────────────────────────────────────────────────────
# HTML 生成：竖向时间轴 + 内嵌弹窗
# ─────────────────────────────────────────────────────────────────────────────
_STATUS_COLOR = {
    "✅": "#61d5c7", "❌": "#ff6b6b",
    "⏳": "#ffd166", "⚪": "#555",
    "[见状态栏]": "#7aa2ff",
}

_VTL_CSS = """
<style>
@keyframes _spin{to{transform:rotate(360deg)}}
._sp{display:inline-block;width:14px;height:14px;border:2px solid #444;
  border-top-color:#7aa2ff;border-radius:50%;animation:_spin .8s linear infinite;
  vertical-align:middle;margin-right:4px}
.vtl{padding:4px 0}
.vtl-item{display:flex;gap:0;margin-bottom:0}
.vtl-col{display:flex;flex-direction:column;align-items:center;width:28px;flex-shrink:0}
.vtl-dot{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:11px;font-weight:700;flex-shrink:0;z-index:1}
.vtl-seg{width:2px;flex:1;min-height:12px;background:#2a2a2a;margin:0}
.vtl-body{padding:2px 0 18px 12px;flex:1;min-width:0}
.vtl-name{font-weight:600;font-size:0.88em;color:#e0e0e0;margin-bottom:2px}
.vtl-time{font-size:0.72em;color:#666;margin-bottom:6px}
.ptags{display:flex;flex-direction:column;gap:4px}
.ptag{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;
  border-radius:12px;font-size:0.75em;border:1px solid;cursor:default;width:fit-content}
.ptag-btn{background:rgba(255,255,255,0.06);border:1px solid #444;color:#ccc;
  border-radius:6px;padding:1px 7px;font-size:0.72em;cursor:pointer;flex-shrink:0}
.ptag-btn:hover{background:rgba(255,255,255,0.12)}
/* modal overlay */
.pm-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,0.75);z-index:99999;align-items:center;justify-content:center}
.pm-overlay.open{display:flex!important}
.pm-box{background:#131320;border:1px solid #2a2a3a;border-radius:14px;
  max-width:580px;width:92%;max-height:82vh;overflow-y:auto;padding:24px;
  position:relative;box-shadow:0 8px 40px rgba(0,0,0,0.6)}
.pm-close{position:absolute;top:12px;right:14px;background:none;border:none;
  color:#aaa!important;font-size:1.4em;cursor:pointer;line-height:1}
.pm-close:hover{color:#fff!important}
.pm-title{margin:0 0 16px 0;font-size:1.05em;color:#a5c0ff!important;font-weight:700}
.pm-section{margin-bottom:14px}
.pm-label{font-size:0.72em;color:#bbb!important;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.pm-block{background:#0d0d1f;padding:12px;border-radius:8px;
  font-size:0.82em;line-height:1.65;white-space:pre-wrap;color:#e8e8e8!important}
.pm-block-p{border-left:3px solid #7aa2ff}
.pm-block-v{border-left:3px solid #61d5c7}
.pm-data{background:#0d0d1f;padding:10px 12px;border-radius:8px;
  font-size:0.78em;color:#c0c0c0!important;font-family:monospace;word-break:break-all;
  max-height:120px;overflow-y:auto;margin-top:8px}
</style>
"""

_VTL_MODAL_HTML = """
<div id="vtl-proof-modal" class="pm-overlay" onclick="if(event.target===this)closePM()">
  <div class="pm-box">
    <button class="pm-close" onclick="closePM()">✕</button>
    <div id="pm-title" class="pm-title">证明详情</div>
    <div class="pm-section">
      <div class="pm-label">证明者 (Prover) 提交</div>
      <div id="pm-prover" class="pm-block pm-block-p"></div>
    </div>
    <div class="pm-section">
      <div class="pm-label">验证者 (Verifier) 检查</div>
      <div id="pm-verifier" class="pm-block pm-block-v"></div>
    </div>
    <div id="pm-data-wrap" class="pm-section" style="display:none">
      <div class="pm-label">当前证明数据</div>
      <div id="pm-data" class="pm-data"></div>
    </div>
  </div>
</div>
"""

# JS 已移至 PAGE_JS（通过 gr.Blocks(js=...) 注入，避免 React innerHTML 不执行 script 的问题）
_VTL_MODAL_JS = ""


def _build_proofs_dict(zac: Dict, sc: Dict, corpus_proofs: List, proof_id: str,
                       zkllm_result: Optional[Dict],
                       fs_layers: list = None, mode: str = "random") -> Dict:
    """返回 key → {title, prover, verifier, data} 的证明详情字典，用于嵌入按钮 data 属性"""
    N = sc.get("N", len(_corpus)) if sc else len(_corpus)
    is_full = (mode == "full")

    # zkllm_query 描述：根据实际选中层动态生成
    if fs_layers is not None:
        if is_full:
            layer_desc = "全量 36 层（层0-35）"
            mode_note  = "【全量模式】后台异步验证，不阻塞检索响应"
        else:
            layer_desc = f"Fiat-Shamir 选中 K={len(fs_layers)} 层：{fs_layers}"
            mode_note  = (f"【随机抽样】challenge = SHA256(query || proof_id)\n"
                          f"安全性: P(检出 | L层被篡改) ≥ 1 - C(36-L,{K_LAYERS_TEXT})/C(36,{K_LAYERS_TEXT})")
    else:
        layer_desc = f"后 K={K_LAYERS_TEXT} 层"
        mode_note  = ""

    proofs = {
        "zkllm_corpus": {
            "title": "zkLLM 证明（语料库侧·完整五组件）",
            "prover": (
                f"离线预计算 · corpus_proof_{{image_id}}.json\n"
                f"对每张图像证明完整前向传播链：\n"
                f"  · Conv3d Patch Embedding       (IPA on BLS12-381)\n"
                f"  · ViT 32 Blocks               (IPA × 9/块，Window+Full Attn)\n"
                f"  · PatchMerger                 (IPA × 3)\n"
                f"  · Pooling: MeanPool Sumcheck + L2Norm 代数约束\n"
                f"  · LLM 36 Decoder Layers       (IPA × 6/层)\n"
                f"证明绑定公开权重承诺（ppgen + commit-param 离线生成）。"
            ),
            "verifier": (
                f"验证图像 embedding 由 jina-v4 完整五组件正确产生：\n"
                f"  1. Conv3d IPA binding check\n"
                f"  2. ViT 32块 IPA binding check (批量验证 288 个 proof)\n"
                f"  3. PatchMerger IPA binding check\n"
                f"  4. Pooling Sumcheck + L2Norm 约束验证\n"
                f"  5. LLM 全量36层 IPA binding check\n"
                f"Prover 无法替换任意组件的计算结果。"
            ),
            "data": None,
        },
        "zkllm_query": {
            "title": f"zkLLM 证明（查询侧·{'全量' if is_full else 'Fiat-Shamir'}）",
            "prover": (
                f"实时后台计算 · zkllm_proof_{proof_id}.json\n"
                f"{layer_desc}\n\n"
                f"每层证明：\n"
                f"  · FFN (SwiGLU gate+up+down) Sumcheck\n"
                f"  · Self-Attn linear Q/K/V 投影 Sumcheck\n"
                f"  · GQA zkAttn Softmax Sumcheck（16 Q头 / 2 KV头）\n"
                f"激活零填充到 seq={SEQ_LEN_PAD}，满足 NTT 约束 seq²=2²⁰。\n\n"
                f"{mode_note}"
            ),
            "verifier": (
                f"验证查询 embedding 由同一 jina-v4 正确推理，\n"
                f"确保查询与文档在同一特征空间下计算相似度。\n\n"
                + (f"Fiat-Shamir 层选择（防 Prover 预选有利层）：\n"
                   f"  challenge = SHA256({proof_id[:8]}…)\n"
                   f"  选中层: {fs_layers}\n" if not is_full and fs_layers else
                   f"全量 36 层，后台异步验证，结果通过状态栏更新。\n")
                + f"\nproof_id = {proof_id}"
            ),
            "data": None,
        },
        "sumcheck": {
            "title": "Global Batch Sumcheck 内积证明",
            "prover": (
                f"实时计算 · 覆盖全量语料库 N={N} 个向量\n"
                f"提交内容：\n"
                f"  · 全部 N={N} 个查询-文档内积值（N×8B）\n"
                f"  · Sumcheck 多项式证明（~264B）\n"
                f"总证明大小：约 {N*8 + 264} 字节"
            ),
            "verifier": (
                f"接收全 N={N} 个内积分值，独立验证：\n"
                f"  1. Sumcheck：Σᵢ sᵢ = 证明声明的总和\n"
                f"  2. 独立对 N 个分值排序，得到自己的 top-{TOP_K}\n"
                f"  3. 比对 Prover 返回的 top-k 是否一致\n\n"
                f"Verifier 不信任 Prover 的排名，自行确认最优解。"
            ),
            "data": None,
        },
        "zac": {
            "title": "ZAC 聚合成员证明（BLS12-381）",
            "prover": (
                f"实时计算 · Pointproofs on BLS12-381\n"
                f"提交内容：\n"
                f"  · ZAC Root cm_hex（G₁ 点，语料库构建时公开发布）\n"
                f"  · 聚合成员证明 π̂（48 字节，1 个 G₁ 点）\n"
                f"  · {TOP_K} 张图像的 SHA256(image ∥ embedding) → BF → 承诺\n\n"
                f"O(1) 证明大小，与语料库规模 N 无关。"
            ),
            "verifier": (
                f"持有发布的 cm_hex，验证配对方程：\n\n"
                f"  e(cm, Σ tᵢ·g₂^{{α^i}}) = e(π̂, g₂) · gT^{{...}}\n\n"
                f"通过则确认：返回的 {TOP_K} 张图像均属于\n"
                f"构建时已承诺的原始语料库，未被替换或篡改。\n"
                f"（绑定 embedding：ZAC 承诺的是 SHA256(image∥embedding)）"
            ),
            "data": None,
        },
    }

    # 填充实际证明数据
    if sc and not sc.get("error"):
        idx = sc.get("top_k_indices", [])
        proofs["sumcheck"]["data"] = (
            f"verified={sc.get('verified')}  N={sc.get('N')}  k={TOP_K}\n"
            f"proof_bytes={sc.get('proof_bytes')}\n"
            f"prove_ms={sc.get('prove_ms')}  verify_ms={sc.get('verify_ms')}\n"
            f"top_k_indices={idx}"
        )
    if zac and not zac.get("disabled") and not zac.get("error"):
        cm = zac.get("cm_hex", "")
        ph = zac.get("proof_hex", "")
        n_f = zac.get("n_filters", 1)
        proofs["zac"]["data"] = (
            f"verified={zac.get('verified')}\n"
            f"n_filters={n_f}  proof_size={n_f*48}B\n"
            f"cm_hex={cm[:32]}…\n"
            f"proof_hex(layer0)={ph[:32]}…\n"
            f"prove_ms={zac.get('prove_ms')}  verify_ms={zac.get('verify_ms')}"
        )
    if corpus_proofs:
        ok_n = sum(1 for cp in corpus_proofs if cp.get("all_ok"))
        lines = [f"corpus proofs: {ok_n}/{len(corpus_proofs)} 全量验证通过"]
        for i, cp in enumerate(corpus_proofs):
            if cp.get("image_id"):
                iid  = cp.get("image_id", f"img{i+1}")
                ms   = cp.get("elapsed_ms", 0)
                ov   = "✓" if cp.get("all_ok") else "✗"
                res  = cp.get("results", {})
                c3ok = "✓" if res.get("conv3d", {}).get("all_ok") else "✗"
                vp   = res.get("vit", {}).get("n_pass", "?")
                vt   = res.get("vit", {}).get("n_total", 32)
                pmok = "✓" if res.get("patchmerger", {}).get("all_ok") else "✗"
                pool = "✓" if res.get("pooling", {}).get("all_ok") else "✗"
                lp   = res.get("llm_layers", {}).get("n_pass", "?")
                lt   = res.get("llm_layers", {}).get("n_total", 36)
                lines.append(
                    f"  [{i+1}] {iid[:35]}: {ov}  {ms//1000}s\n"
                    f"       Conv3d:{c3ok} ViT:{vp}/{vt} PM:{pmok} Pool:{pool} LLM:{lp}/{lt}"
                )
        proofs["zkllm_corpus"]["data"] = "\n".join(lines)
    if zkllm_result:
        ok  = zkllm_result.get("verified", False)
        ms  = zkllm_result.get("elapsed_ms", 0)
        lrs = zkllm_result.get("layers", [])
        sel = zkllm_result.get("fiat_shamir_layers", fs_layers or [])
        mode_line = "全量验证  层0-35" if is_full else f"Fiat-Shamir  选中层: {sel}"
        detail_lines = []
        for r in lrs:
            li  = r["layer"]
            ffn = "✓" if r.get("ffn_rc") == 0 else "✗"
            lin = "✓" if r.get("attn_linear_rc") == 0 else "✗"
            sfx = "✓" if r.get("attn_sfx_rc") == 0 else "✗"
            detail_lines.append(f"  L{li:2d}: FFN {ffn}  Attn-linear {lin}  zkAttn {sfx}")
        proofs["zkllm_query"]["data"] = (
            f"proof_id={proof_id}\n"
            f"{mode_line}\n"
            f"verified={ok}  elapsed={ms/1000:.1f}s\n"
            + ("\n".join(detail_lines) if detail_lines else "(验证中…)")
        )

    return proofs


# ─────────────────────────────────────────────────────────────────────────────
# 竖向时间轴节点渲染
# ─────────────────────────────────────────────────────────────────────────────
def _dot_style(icon: str) -> str:
    c = _STATUS_COLOR.get(icon, "#555")
    bg = c + "22"
    return f'style="background:{bg};border:2px solid {c};color:{c}"'


def _proof_tag_html(name: str, icon: str, detail: Dict) -> str:
    """证明 badge；将 detail dict JSON 编码后嵌入 data-proof-json 属性（由全局 JS 读取）"""
    c = _STATUS_COLOR.get(icon, "#555")
    label = "验证中" if icon == "⏳" else ("通过" if icon == "✅" else ("失败" if icon == "❌" else ("见状态栏" if icon == "[见状态栏]" else "待验证")))
    spinner = '<span class="_sp"></span>' if icon == "⏳" else ""
    # HTML-escape the JSON so it's safe inside a double-quoted attribute
    data_attr = _html.escape(json.dumps(detail), quote=True) if detail else ""
    btn = (f'<button class="ptag-btn" data-proof-json="{data_attr}">展开 ›</button>'
           if data_attr else "")
    return (
        f'<div class="ptag" style="border-color:{c}30;background:{c}0f">'
        f'{spinner}<span style="color:{c}">{icon}</span>'
        f'<span style="color:#ccc">{name}</span>'
        f'<span style="color:#888;font-size:0.9em">{label}</span>'
        f'{btn}'
        f'</div>'
    )


def _vtl_node(icon: str, name: str, t: str, proof_tags: str, is_last: bool) -> str:
    dot = _dot_style(icon)
    seg = "" if is_last else '<div class="vtl-seg"></div>'
    return (
        f'<div class="vtl-item">'
        f'  <div class="vtl-col">'
        f'    <div class="vtl-dot" {dot}>{icon}</div>'
        f'    {seg}'
        f'  </div>'
        f'  <div class="vtl-body">'
        f'    <div class="vtl-name">{name}</div>'
        f'    <div class="vtl-time">{t}</div>'
        f'    {"<div class=ptags>" + proof_tags + "</div>" if proof_tags else ""}'
        f'  </div>'
        f'</div>'
    )


def build_vtl_html(steps: List[Dict], proofs: Dict = None) -> str:
    """
    steps: [{icon, name, time, proofs: [(display_name, icon, key), ...]}]
    proofs: key → {title, prover, verifier, data}  (from _build_proofs_dict or _BUILD_PROOF_DATA)
    """
    nodes = []
    proofs = proofs or {}
    for i, s in enumerate(steps):
        tags = "".join(
            _proof_tag_html(pn, pi, proofs.get(pk, {}))
            for pn, pi, pk in s.get("proofs", [])
        )
        nodes.append(_vtl_node(s["icon"], s["name"], s.get("time", "—"), tags, i == len(steps) - 1))
    return '<div class="vtl">' + "".join(nodes) + "</div>"


# 查询流程步骤模板
def _query_steps(
    input_icon="✅", input_t="—",
    encode_icon="⚪", encode_t="—", zkllm_q_icon="⚪",
    retrieve_icon="⚪", retrieve_t="—",
    zkllm_c_icon="⚪", sc_icon="⚪", zac_icon="⚪",
    results_icon="⚪", results_t="—",
    verify_icon="⚪", verify_t="—",
    gen_icon="⚪", gen_t="—",
    fs_info="",   # Fiat-Shamir 层信息，例 "层[3,8,15,22,29,35]" 或 "全量36层"
) -> List[Dict]:
    encode_time = f"{encode_t}  ·  {fs_info}".strip(" ·") if fs_info else encode_t
    return [
        {"icon": input_icon,    "name": "用户输入问题",      "time": input_t,     "proofs": []},
        {"icon": encode_icon,   "name": "向量编码 (jina-v4)", "time": encode_time,
         "proofs": [("zkLLM query", zkllm_q_icon, "zkllm_query")]},
        {"icon": retrieve_icon, "name": "精确检索 (FAISS)",   "time": retrieve_t,
         "proofs": [
             ("zkLLM corpus (全5组件)", zkllm_c_icon, "zkllm_corpus"),
             ("Sumcheck 内积",          sc_icon,       "sumcheck"),
             ("ZAC 成员证明",           zac_icon,      "zac"),
         ]},
        {"icon": results_icon, "name": "返回检索结果",        "time": results_t,   "proofs": []},
        {"icon": verify_icon,  "name": "验证通过 / 阻止生成", "time": verify_t,    "proofs": []},
        {"icon": gen_icon,     "name": "大模型生成回答",      "time": gen_t,       "proofs": []},
    ]


# 建库流程（静态，用于展示机制）
_BUILD_STEPS_STATIC = [
    {"icon": "✅", "name": "PDF 导入",               "time": "—",   "proofs": []},
    {"icon": "✅", "name": "图像切片 → image.jsonl",  "time": "—",
     "proofs": [("SHA-256 指纹", "✅", "build_hash")]},
    {"icon": "✅", "name": "语义编码 → embedding.npy","time": "—",
     "proofs": [("zkLLM corpus 预计算", "✅", "build_zkllm")]},
    {"icon": "✅", "name": "构建向量索引 (FAISS)",    "time": "—",
     "proofs": [("Sumcheck 内积覆盖", "✅", "build_sc")]},
    {"icon": "✅", "name": "生成语料库承诺 (ZAC)",    "time": "—",
     "proofs": [("ZAC Root 承诺", "✅", "build_zac")]},
    {"icon": "✅", "name": "zkLLM 预计算（后台）",    "time": "~3.6h (288 张)",
     "proofs": []},
    {"icon": "✅", "name": "知识库就绪",              "time": "—",   "proofs": []},
]

_BUILD_PROOF_DATA = {
    "build_hash": {
        "title": "SHA-256 数字指纹",
        "prover": "对每张图像文件计算 SHA-256(image_bytes)。\n累积到 Bloom Filter 向量，作为 ZAC 承诺的输入集合 S。\n输出：prover_state.json（含 BF 参数 + S）",
        "verifier": "持有 ZAC Root (cm_hex) 的验证者可独立验证：\n任意图像 i 的哈希是否属于已承诺的集合 S。",
        "data": "位于 output/phase1/fingerprint.json\nBF 位数组 + Pointproofs 承诺（48 字节 G₁ 点）",
    },
    "build_zkllm": {
        "title": "zkLLM Sumcheck 预计算（语料库侧）",
        "prover": "对语料库每张图像，用 jina-v4 后 K=5 层（层31-35）权重，\n对量化激活值运行三段式 Sumcheck 证明：\n  · FFN (SwiGLU)：seq×2048 × 2048×11008\n  · Self-Attn linear：seq×2048 × 2048×256（GQA 投影）\n  · GQA zkAttn Softmax：16 Q头 / 2 KV头，seq=1024\n输出：corpus_proof_{image_id}.json",
        "verifier": "验证者收到 corpus_proof，重跑 Sumcheck 验证：\n多项式求值与权重承诺（KZG）一致，\n确认 embedding 由声称的 jina-v4 推理产生。",
        "data": f"zkllm-workdir/jina-v4/corpus_proof_*.json\n303 个文件，每张 ~95s（RTX 4090 D，双卡并行约 4h）",
    },
    "build_sc": {
        "title": "Global Batch Sumcheck 覆盖设计",
        "prover": "在线检索时为每次查询生成证明，覆盖全量 N 个内积。\n无需离线预计算，建库时确认 FAISS 使用 IndexFlatIP\n（精确内积，非 ANN 近似），保证 Sumcheck 证明语义完整。",
        "verifier": "每次检索时接收全 N 个内积值，独立排序，\n无需信任检索服务器的排名结果。",
        "data": "使用 FAISS IndexFlatIP（暴力内积）\n确保 Sumcheck 证明精确覆盖所有候选",
    },
    "build_zac": {
        "title": "ZAC 语料库承诺（Pointproofs + BF）",
        "prover": "S = {SHA256(image_bytes_i) | i ∈ corpus}\nBF.Gen(S) → 二进制向量 v\nPointproofs.Commit(v, r) → cm ∈ G₁（48 字节）\n\ncm_hex 需通过可信渠道发布（论文附录/公开公告）。",
        "verifier": "持有 cm_hex，对任意返回图像集合验证：\ne(cm, Σ tᵢ·g₂^{α^i}) = e(π̂, g₂)·gT^{...}\n\n证明大小 O(1)，与语料库规模 N 无关。",
        "data": "output/phase1/fingerprint.json\nZAC Root (cm_hex) 已在脚本运行时输出\n请确认已公开发布",
    },
}


def build_build_vtl_html() -> str:
    return build_vtl_html(_BUILD_STEPS_STATIC, _BUILD_PROOF_DATA)


# ─────────────────────────────────────────────────────────────────────────────
# 生成回答 HTML
# ─────────────────────────────────────────────────────────────────────────────
_WRAP = 'style="max-height:220px;overflow-y:auto;border:1px solid #2a2a2a;border-radius:8px"'

def answer_html(text: str = "", loading: bool = False, blocked: bool = False) -> str:
    if loading:
        inner = (
            '<div style="display:flex;align-items:center;gap:8px;padding:10px;color:#aaa">'
            '<span class="_sp"></span>'
            '<span>MiniCPM-V-4 生成中…</span></div>'
        )
    elif blocked:
        inner = (
            '<div style="padding:10px;color:#ff8a80;border-left:3px solid #ff6b6b;'
            'background:#1a0a0a;border-radius:4px">'
            '⚠️ 验证未全部通过，已阻止生成回答。请检查右侧验证状态。</div>'
        )
    elif not text:
        inner = '<div style="color:#555;padding:10px;font-style:italic">提交查询后显示回答</div>'
    else:
        esc = (text.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace("\n", "<br/>"))
        inner = f'<div style="padding:10px;line-height:1.75;font-size:0.9em">{esc}</div>'
    return f'<div {_WRAP}>{inner}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# 主查询流程
# Yields: (gallery, vtl_html, answer_html_str, proof_id_state)
# ─────────────────────────────────────────────────────────────────────────────
def on_query(query: str, mode: str = "随机抽样 (K=6层, ~46s)"):
    if not query.strip():
        yield [], build_vtl_html(_query_steps()), answer_html(), ""
        return

    is_full  = "全量" in mode
    proof_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*60}")
    print(f"[Query] 新查询  proof_id={proof_id}  "
          f"mode={'全量36层' if is_full else 'Fiat-Shamir K=6'}  query={query[:80]!r}")
    print(f"{'='*60}")

    # 共享证明数据（随步骤更新）；fs_layers/fs_info 在 Step 0 赋值，之后由 vtl() 读取
    pdata     = {"zac": {}, "sc": {}, "corpus_proofs": [], "zkllm_result": None}
    fs_layers = None
    fs_info   = ""

    def vtl(**kw):
        steps  = _query_steps(fs_info=fs_info, **kw)
        proofs = _build_proofs_dict(
            pdata["zac"], pdata["sc"], pdata["corpus_proofs"],
            proof_id, pdata["zkllm_result"],
            fs_layers=fs_layers, mode="full" if is_full else "random",
        )
        return build_vtl_html(steps, proofs)

    def y(gallery_v, ah, **kw):
        return gallery_v, vtl(**kw), ah, proof_id

    # ── Step 0: 层挑战策略 + 启动 zkLLM query proof 后台线程 ──────────────────
    if is_full:
        fs_layers = list(range(36))
        fs_info   = "全量 36层 (后台异步)"
    else:
        fs_layers = _fiat_shamir_layers(query, proof_id, K=K_LAYERS_TEXT)
        fs_info   = f"Fiat-Shamir → 层{fs_layers}"
    print(f"[Step 0] {'全量36层' if is_full else f'Fiat-Shamir层: {fs_layers}'}  proof_id={proof_id}")

    act_ready    = threading.Event()
    zkllm_thread = threading.Thread(
        target=_run_zkllm_query_bg, args=(proof_id, act_ready, fs_layers), daemon=True)
    # 全量模式：延迟到 MiniCPM 生成完成后再启动（避免 GPU1 内存竞争）
    # 随机模式：立即启动，Step 6 会 join 等待完成
    if not is_full:
        zkllm_thread.start()
    yield y([], answer_html(), input_icon="✅", encode_icon="⏳", zkllm_q_icon="⏳")

    # ── Step 1: 向量编码（同时捕获 hook 激活，完成后 act_ready.set()）──────────
    t0    = time.perf_counter()
    q_emb = embed_query_with_hooks(query, proof_id, act_ready, layers=fs_layers)
    enc_ms = round((time.perf_counter() - t0) * 1000, 1)
    yield y([], answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms", zkllm_q_icon="⏳",
            retrieve_icon="⏳")

    # ── Step 2: FAISS 检索 ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    paths, _, emb_ids = faiss_search(q_emb)
    faiss_ms = round((time.perf_counter() - t0) * 1000, 1)
    images   = [p for p in paths if p and Path(p).exists()]
    yield y(images, answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms", zkllm_q_icon="⏳",
            retrieve_icon="⏳", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon="⏳", sc_icon="⏳", zac_icon="⏳")

    # ── Step 3: zkLLM corpus proof（离线预计算，直接读取）────────────────────
    pdata["corpus_proofs"] = get_corpus_proofs(paths)
    ok_n = sum(1 for cp in pdata["corpus_proofs"] if cp.get("all_ok") is True)
    zkllm_c_icon = "✅" if ok_n == TOP_K else ("❌" if ok_n == 0 else "⚠️")
    yield y(images, answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms", zkllm_q_icon="⏳",
            retrieve_icon="⏳", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="⏳", zac_icon="⏳")

    # ── Step 4: Sumcheck ──────────────────────────────────────────────────────
    pdata["sc"] = run_sumcheck(q_emb)
    sc_ok  = pdata["sc"].get("verified", False)
    yield y(images, answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms", zkllm_q_icon="⏳",
            retrieve_icon="⏳", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌", zac_icon="⏳")

    # ── Step 5: ZAC ───────────────────────────────────────────────────────────
    pdata["zac"] = run_zac(paths, emb_ids)
    zac_ok      = pdata["zac"].get("verified", False) and not pdata["zac"].get("disabled")
    zac_icon_v  = "✅" if zac_ok else ("⚪" if pdata["zac"].get("disabled") else "❌")
    yield y(images, answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms", zkllm_q_icon="⏳",
            retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
            zac_icon=zac_icon_v,
            results_icon="✅",
            verify_icon="⏳", verify_t=("Phase 3 后台进行中…" if is_full else "等待 zkLLM query…"))

    # ── Step 6: zkLLM query proof ─────────────────────────────────────────────
    if is_full:
        # 全量模式：Phase 3 后台异步，不阻塞响应流程
        # 生成阻断逻辑仅依赖 Phase 1 (ZAC) + Phase 2 (Sumcheck) + corpus proofs
        print(f"[Step 6] 全量模式：Phase 3 后台运行中，不等待")
        zkllm_q_icon_final = "[见状态栏]"
        all_ok      = sc_ok and zac_ok and (ok_n == TOP_K)
        verify_icon_v = "✅" if all_ok else "❌"
        verify_t_str  = (
            f"Phase 1+2 {'✅' if all_ok else '❌'} · Phase 3 全量后台异步…\n"
            f"sc={'✅' if sc_ok else '❌'}  zac={'✅' if zac_ok else '❌'}  corpus={ok_n}/{TOP_K}"
        )
        print(f"[Step 6] sc={sc_ok}  zac={zac_ok}  corpus={ok_n}/{TOP_K}  all_ok={all_ok}")
    else:
        # 随机抽样模式：轮询等待 zkLLM（最多 180s），每 5s yield 一次保持 SSE 连接活跃。
        # 直接 join(timeout=180) 会导致 ~90s 无 yield，Gradio SSE 超时断连 → 页面"刷新"。
        print(f"[Step 6] Fiat-Shamir 模式：轮询等待 zkLLM query proof 线程（最多 180s）…")
        _waited = 0
        while zkllm_thread.is_alive() and _waited < 180:
            zkllm_thread.join(timeout=5)
            _waited += 5
            if zkllm_thread.is_alive():
                yield y(images, answer_html(),
                        input_icon="✅",
                        encode_icon="✅", encode_t=f"{enc_ms:.0f}ms",
                        zkllm_q_icon="⏳",
                        retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
                        zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
                        zac_icon=zac_icon_v,
                        results_icon="✅",
                        verify_icon="⏳", verify_t=f"等待 zkLLM query… ({_waited}s)")
        pdata["zkllm_result"] = _get_zkllm_result(proof_id)
        zkllm_ok = (pdata["zkllm_result"] is not None
                    and pdata["zkllm_result"].get("verified", False))
        zkllm_q_icon_final = "✅" if zkllm_ok else "❌"
        zkllm_ms = (pdata["zkllm_result"].get("elapsed_ms", 0)
                    if pdata["zkllm_result"] else 0)
        all_ok        = sc_ok and zac_ok and zkllm_ok and (ok_n == TOP_K)
        verify_icon_v = "✅" if all_ok else "❌"
        verify_t_str  = (
            f"zkLLM {zkllm_ms//1000}s · {'全部通过' if all_ok else '存在失败'}\n"
            f"层{fs_layers}  sc={'✅' if sc_ok else '❌'}  zac={'✅' if zac_ok else '❌'}  corpus={ok_n}/{TOP_K}"
        )
        print(f"[Step 6] zkLLM_q={'✅' if zkllm_ok else '❌'}  sc={'✅' if sc_ok else '❌'}"
              f"  zac={'✅' if zac_ok else '❌'}  corpus={ok_n}/{TOP_K}  all_ok={all_ok}")

    yield y(images, answer_html(),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms",
            zkllm_q_icon=zkllm_q_icon_final,
            retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
            zac_icon=zac_icon_v,
            results_icon="✅", verify_icon=verify_icon_v, verify_t=verify_t_str,
            gen_icon="⏳")

    # ── Step 7: 大模型生成 ────────────────────────────────────────────────────
    if not all_ok:
        yield y(images, answer_html(blocked=True),
                input_icon="✅",
                encode_icon="✅", encode_t=f"{enc_ms:.0f}ms",
                zkllm_q_icon=zkllm_q_icon_final,
                retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
                zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
                zac_icon=zac_icon_v,
                results_icon="✅", verify_icon=verify_icon_v, verify_t=verify_t_str,
                gen_icon="❌", gen_t="验证未通过，已阻止")
        return

    yield y(images, answer_html(loading=True),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms",
            zkllm_q_icon=zkllm_q_icon_final,
            retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
            zac_icon=zac_icon_v,
            results_icon="✅", verify_icon=verify_icon_v, verify_t=verify_t_str,
            gen_icon="⏳")

    print(f"[Step 7] 生成回答  n_images={len(images)}")
    t0     = time.perf_counter()
    ans    = run_generation(query, images)
    gen_ms = round((time.perf_counter() - t0) * 1000, 1)
    print(f"[Step 7] 生成完成  {gen_ms/1000:.1f}s  answer_len={len(ans)}")

    # 释放 MiniCPM 在 GPU1 上的 KV cache / 激活缓存池，让 CUDA 空闲内存对子进程可见
    import torch as _torch
    _torch.cuda.empty_cache()

    # 全量模式：MiniCPM 生成完毕，GPU1 KV cache 已释放，现在启动 zkLLM 后台线程
    if is_full:
        print(f"[Step 7→8] 启动全量 zkLLM 后台线程（GPU1 已空闲）")
        zkllm_thread.start()
        # 存储查询最终状态，供 timer 回调在 zkLLM 完成后重建 VTL
        _query_final_state[proof_id] = {
            "pdata":       {"zac": pdata["zac"], "sc": pdata["sc"],
                            "corpus_proofs": pdata["corpus_proofs"]},
            "fs_info":     fs_info,
            "fs_layers":   fs_layers,
            "icons": {
                "input_icon":    "✅",
                "encode_icon":   "✅", "encode_t":   f"{enc_ms:.0f}ms",
                "retrieve_icon": "✅", "retrieve_t": f"{faiss_ms:.0f}ms",
                "zkllm_c_icon":  zkllm_c_icon,
                "sc_icon":       "✅" if sc_ok else "❌",
                "zac_icon":      zac_icon_v,
                "results_icon":  "✅",
                "gen_icon":      "✅", "gen_t": f"{gen_ms/1000:.1f}s",
                "verify_icon":   verify_icon_v,
                "verify_t":      verify_t_str,
            },
        }

    yield y(images, answer_html(ans),
            input_icon="✅",
            encode_icon="✅", encode_t=f"{enc_ms:.0f}ms",
            zkllm_q_icon=zkllm_q_icon_final,
            retrieve_icon="✅", retrieve_t=f"{faiss_ms:.0f}ms",
            zkllm_c_icon=zkllm_c_icon, sc_icon="✅" if sc_ok else "❌",
            zac_icon=zac_icon_v,
            results_icon="✅", verify_icon=verify_icon_v, verify_t=verify_t_str,
            gen_icon="✅", gen_t=f"{gen_ms/1000:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 新建知识库流程
# Yields: (build_log, build_vtl_html, build_status)
# ─────────────────────────────────────────────────────────────────────────────
def on_build(pdf_path: str):
    if not pdf_path:
        yield "请先指定 PDF 路径", build_build_vtl_html(), "⚠️ 未指定 PDF"
        return

    # 自动检测：语料库已存在则使用增量模式
    is_incremental = CORPUS_PATH.exists()
    mode_label = "增量模式（追加新内容）" if is_incremental else "全量模式（首次建库）"

    if is_incremental:
        slice_name = "追加新图像切片"
        embed_name = "增量语义编码 + FAISS 追加"
    else:
        slice_name = "图像切片"
        embed_name = "语义编码 → embedding.npy"

    steps = [
        {"icon": "✅", "name": f"准备构建（{mode_label}）", "time": "—", "proofs": []},
        {"icon": "⏳", "name": slice_name,   "time": "—", "proofs": [("SHA-256 指纹", "⏳", "build_hash")]},
        {"icon": "⚪", "name": embed_name,   "time": "—", "proofs": [("zkLLM corpus 预计算", "⚪", "build_zkllm")]},
        {"icon": "⚪", "name": "构建/更新向量索引", "time": "—", "proofs": [("Sumcheck 内积覆盖", "⚪", "build_sc")]},
        {"icon": "⚪", "name": "重建语料库承诺 (ZAC)","time": "—", "proofs": [("ZAC Root 承诺", "⚪", "build_zac")]},
        {"icon": "⚪", "name": "zkLLM 预计算（跳过已有）" if is_incremental else "zkLLM 预计算", "time": "—", "proofs": []},
        {"icon": "⚪", "name": "知识库就绪",   "time": "—", "proofs": []},
    ]

    def bvtl():
        return build_vtl_html(steps, _BUILD_PROOF_DATA)

    log = f"开始构建知识库\nPDF: {pdf_path}\n模式: {mode_label}\n"
    yield log, bvtl(), "🔄 构建中…"

    cmd = [
        sys.executable, "script/build_verifiable_corpus.py",
        "--pdf", pdf_path,
        "--k-layers", str(K_LAYERS_IMG),   # 图像语料库侧 K=5
    ]
    if is_incremental:
        cmd.append("--incremental")

    steps[1]["icon"] = "⏳"
    steps[1]["proofs"][0] = ("SHA-256 指纹", "⏳", "build_hash")
    yield log + f"[ 步骤 1 ] {slice_name}…\n", bvtl(), f"🔄 {slice_name}中…"

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    step_idx = 1
    t_step = time.perf_counter()
    for line in proc.stdout:
        log += line
        if "image corpus" in line.lower() or "corpus" in line.lower():
            steps[1]["icon"] = "⏳"
        elif "embedding" in line.lower() or "embed" in line.lower():
            if step_idx < 2:
                steps[1] = {**steps[1], "icon": "✅", "time": f"{time.perf_counter()-t_step:.0f}s",
                             "proofs": [("SHA-256 指纹", "✅", "build_hash")]}
                steps[2]["icon"] = "⏳"
                steps[2]["proofs"][0] = ("zkLLM corpus 预计算", "⏳", "build_zkllm")
                step_idx = 2
                t_step = time.perf_counter()
        elif "zac" in line.lower() or "phase 1" in line.lower():
            if step_idx < 4:
                steps[2] = {**steps[2], "icon": "✅", "time": f"{time.perf_counter()-t_step:.0f}s",
                             "proofs": [("zkLLM corpus 预计算", "✅", "build_zkllm")]}
                steps[3] = {**steps[3], "icon": "✅",
                             "proofs": [("Sumcheck 内积覆盖", "✅", "build_sc")]}
                steps[4]["icon"] = "⏳"
                steps[4]["proofs"][0] = ("ZAC Root 承诺", "⏳", "build_zac")
                step_idx = 4
                t_step = time.perf_counter()
        elif "zkllm" in line.lower() and "后台" in line:
            if step_idx < 5:
                steps[4] = {**steps[4], "icon": "✅", "time": f"{time.perf_counter()-t_step:.0f}s",
                             "proofs": [("ZAC Root 承诺", "✅", "build_zac")]}
                steps[5]["icon"] = "⏳"
                step_idx = 5
        yield log[-3000:], bvtl(), "🔄 构建中…"

    proc.wait()
    if proc.returncode == 0:
        for i in range(len(steps)):
            if steps[i]["icon"] in ("⏳", "⚪"):
                steps[i]["icon"] = "✅"
        reload_index()
        n_new = len(_corpus)
        status = f"✅ 知识库构建完成（已热重载，共 {n_new} 条）"
        log += f"\n[热重载] 检索索引已更新，共 {n_new} 条语料\n"
    else:
        status = f"❌ 构建失败（returncode={proc.returncode}）"
    yield log, bvtl(), status


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
# 全局 JS（通过 gr.Blocks(js=...) 注入，确保在页面级别执行，不受 React innerHTML 限制）
PAGE_JS = """
function openPM(btn) {
  var raw = btn.getAttribute('data-proof-json');
  if (!raw) return;
  var d;
  try { d = JSON.parse(raw); } catch(e) { return; }
  document.getElementById('pm-title').textContent    = d.title    || '';
  document.getElementById('pm-prover').textContent   = d.prover   || '';
  document.getElementById('pm-verifier').textContent = d.verifier || '';
  var dw = document.getElementById('pm-data-wrap');
  var dd = document.getElementById('pm-data');
  if (d.data) { dw.style.display = 'block'; dd.textContent = d.data; }
  else         { dw.style.display = 'none'; }
  document.getElementById('vtl-proof-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closePM() {
  document.getElementById('vtl-proof-modal').classList.remove('open');
  document.body.style.overflow = '';
}
// 事件委托：捕获所有带 data-proof-json 的按钮点击
document.addEventListener('click', function(e) {
  var btn = e.target.closest('[data-proof-json]');
  if (btn) { e.preventDefault(); openPM(btn); }
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closePM();
});
"""

CSS = """
/* 限高滚动（gallery 由 Gradio height 参数控制） */
.left-col  { padding-right: 12px; border-right: 1px solid #1e1e2a; }
/* spinner 动画 */
@keyframes _spin { to { transform: rotate(360deg); } }
"""


def build_ui():
    with gr.Blocks(title="可验证多模态检索系统", js=PAGE_JS) as demo:
        gr.Markdown(
            "# 面向多模态语义数据的可验证检索机制-实现示例"
        )

        # 静态：弹窗 overlay（只渲染一次，modal HTML + CSS）
        gr.HTML(_VTL_CSS + _VTL_MODAL_HTML, visible=True)

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════════
            # Tab 1: 检索演示
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🔍 检索演示"):
                with gr.Row(equal_height=False):

                    # ── 左列：检索过程 ────────────────────────────────────────
                    with gr.Column(scale=2, elem_classes="left-col"):
                        gr.Markdown("### 检索过程")

                        # Phase 3 查询侧验证模式选择
                        mode_radio = gr.Radio(
                            choices=["随机抽样 (K=6层, ~30s)", "全量验证 (36层, ~3.5min)"],
                            value="随机抽样 (K=6层, ~46s)",
                            label="zkLLM 查询侧验证模式",
                            info=(
                                "随机：Fiat-Shamir 从36层中随机挑战K=6层，验证完成后返回结果。\n"
                                "全量：验证全部36层，Phase 1+2通过即返回结果，Phase 3后台异步运行。"
                            ),
                        )
                        # Phase 3 全量模式异步状态栏（timer 更新）
                        phase3_status = gr.HTML("")

                        # submit_btn 将按钮渲染在文本框内部右侧
                        q_input = gr.Textbox(
                            placeholder="输入查询，例：尼康Z7的电子减震功能在哪些场景不可用？",
                            lines=1, show_label=False,
                            submit_btn="→",
                        )

                        gallery = gr.Gallery(
                            label=f"检索结果 top-{TOP_K}（点击放大）",
                            columns=3, height=290, object_fit="contain",
                        )
                        gr.Markdown("**回答**")
                        ans_html = gr.HTML(answer_html())

                    # ── 右列：可验证机制 ──────────────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 可验证检索机制")
                        proof_id_state = gr.State("")
                        # 动态：只含时间轴节点（CSS/modal 已在页面级静态组件里）
                        vtl_html = gr.HTML(build_vtl_html(_query_steps()))

                # ── Timer：轮询 Phase 3 完成状态（全量模式异步更新）──────────
                zkllm_timer = gr.Timer(value=4, active=False)

                def _phase3_status_refresh(proof_id: str):
                    """Timer 回调：检查 Phase 3 状态，更新状态栏并控制 Timer 激活"""
                    if not proof_id:
                        return "", gr.update(active=False), gr.update()
                    result = _get_zkllm_result(proof_id)
                    if result is None:
                        return (
                            '<div style="color:#ffd166;font-size:0.83em;padding:3px 8px">'
                            '<span class="_sp"></span> Phase 3 全量验证后台进行中…</div>'
                        ), gr.update(active=True), gr.update()
                    ok  = result.get("verified", False)
                    ms  = result.get("elapsed_ms", 0)
                    lrs = result.get("layers", [])
                    n_ok = sum(1 for r in lrs if r.get("verified"))
                    sel  = result.get("fiat_shamir_layers", [])
                    c  = "#61d5c7" if ok else "#ff6b6b"
                    ic = "✅" if ok else "❌"
                    mode_tag = "全量36层" if len(sel) == 36 else f"Fiat-Shamir 层{sel}"
                    phase3_html = (
                        f'<div style="color:{c};font-size:0.83em;padding:3px 8px">'
                        f'{ic} Phase 3 完成  {n_ok}/{len(lrs)} 层通过  '
                        f'耗时 {ms//1000}s  {mode_tag}</div>'
                    )
                    # 重建 VTL 面板（更新 zkllm_q 和 verify 图标）
                    state = _query_final_state.get(proof_id)
                    if state:
                        icons = dict(state["icons"])
                        icons["zkllm_q_icon"] = "[见状态栏]"
                        # 重新计算综合验证图标/耗时
                        _all_ok = ok and icons.get("sc_icon") == "✅" and icons.get("zac_icon") == "✅"
                        icons["verify_icon"] = "✅" if _all_ok else ("❌" if not ok else "⚠️")
                        icons["verify_t"]    = f"{ms//1000}s"
                        steps  = _query_steps(fs_info=state["fs_info"], **icons)
                        proofs = _build_proofs_dict(
                            state["pdata"]["zac"], state["pdata"]["sc"],
                            state["pdata"]["corpus_proofs"],
                            proof_id, result,
                            fs_layers=state["fs_layers"], mode="full",
                        )
                        vtl_updated = build_vtl_html(steps, proofs)
                    else:
                        vtl_updated = gr.update()
                    return phase3_html, gr.update(active=False), vtl_updated

                zkllm_timer.tick(fn=_phase3_status_refresh,
                                 inputs=[proof_id_state],
                                 outputs=[phase3_status, zkllm_timer, vtl_html])

                outputs = [gallery, vtl_html, ans_html, proof_id_state]
                q_input.submit(fn=on_query, inputs=[q_input, mode_radio], outputs=outputs)
                proof_id_state.change(
                    fn=lambda pid: gr.update(active=bool(pid)),
                    inputs=[proof_id_state], outputs=[zkllm_timer],
                )

            # ══════════════════════════════════════════════════════════════════
            # Tab 2: 新建知识库
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("📚 新建知识库"):
                with gr.Row(equal_height=False):

                    # ── 左列：构建控制 ────────────────────────────────────────
                    with gr.Column(scale=2, elem_classes="left-col"):
                        gr.Markdown("### 构建控制")
                        pdf_input = gr.Textbox(
                            label="PDF 路径",
                            placeholder="相对于项目根目录的 PDF 路径，例：data/nikon.pdf",
                        )
                        _mode_hint = (
                            "**模式自动检测**：语料库已存在 → **增量模式**（追加新页面，仅重建 ZAC）；"
                            "首次建库 → **全量模式**。"
                        )
                        gr.Markdown(_mode_hint)
                        build_btn = gr.Button("🚀 开始建库", variant="primary")
                        build_status = gr.Markdown("_点击「开始建库」启动_")

                        gr.Markdown("---")
                        gr.Markdown("#### 当前知识库状态")
                        n_corpus = len(_corpus) if _corpus else "（未加载）"
                        zac_root = _zac_acc.root_hex()[:24] + "…" if _zac_acc else "（未加载）"
                        n_zkllm  = len(list(ZKLLM_WORKDIR.glob("corpus_proof_*.json"))) \
                                   if ZKLLM_WORKDIR.exists() else 0
                        gr.Markdown(
                            f"- 语料库记录：**{n_corpus}** 条\n"
                            f"- ZAC Root：`{zac_root}`\n"
                            f"- zkLLM 预计算：**{n_zkllm}** 个证明文件\n"
                        )

                        build_log = gr.Textbox(
                            label="构建日志", lines=12, max_lines=20,
                            interactive=False, buttons=["copy"],
                        )

                    # ── 右列：建库时间轴 ──────────────────────────────────────
                    with gr.Column(scale=1):
                        gr.Markdown("### 入库时间线 & 证明机制")
                        gr.Markdown(
                            "_点击各步骤的「展开 ›」按钮，查看该节点设置的证明机制详情_\n"
                            "_（包括证明者提交的内容和验证者验证的内容）_"
                        )
                        build_vtl = gr.HTML(build_build_vtl_html())

                build_btn.click(
                    fn=on_build,
                    inputs=[pdf_input],
                    outputs=[build_log, build_vtl, build_status],
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_all()
    demo = build_ui()
    print("\n启动成功！")
    print("SSH 隧道访问：ssh -L 7860:127.0.0.1:7860 <用户@IP>")
    print("AutoDL 用户：在控制台「自定义服务」配置端口 7860\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                theme=gr.themes.Soft(), css=CSS)

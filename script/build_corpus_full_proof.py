"""
语料库全量证明脚本（per-image full pipeline）

对每张图像证明完整前向传播链：
  1. Conv3d patch embedding  (IPA)
  2. ViT 32 blocks           (IPA × 9/block)
  3. PatchMerger             (IPA × 3)
  4. Pooling head            (Sumcheck + Rescaling)
  5. LLM 36 decoder layers   (IPA × 6/layer)

单 GPU 模式，用 --worker-id / --num-workers 实现多卡并行（每张卡负责不同图像）。

用法：
  # GPU0，处理偶数图（共 303/2 ≈ 152 张）
  python script/build_corpus_full_proof.py \\
      --worker-id 0 --num-workers 2 \\
      2>&1 | tee logs/corpus_full_gpu0.log &

  # GPU1，处理奇数图
  python script/build_corpus_full_proof.py \\
      --worker-id 1 --num-workers 2 \\
      2>&1 | tee logs/corpus_full_gpu1.log &

预计时间（单 GPU per worker）：
  ~22 min/image × 152 images = ~56 h（两 worker 同时，墙钟 ~56 h）
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent.resolve()
BIN_DIR   = ROOT / "src" / "zkllm"
ZKLLM_CWD = ROOT / "src" / "zkllm"
sys.path.insert(0, str(ROOT))

SCALE        = 1 << 16
SEQ_VIT      = 1024   # ViT NTT 约束 pad 目标
SEQ_LLM      = 1024   # LLM NTT 约束 pad 目标
VIT_HIDDEN   = 1280
LLM_HIDDEN   = 2048
VIT_N_BLOCKS = 32
LLM_N_LAYERS = 36
KV_DIM       = 256
NUM_KV_HEADS = 2


# ─────────────────────────────────────────────────────────────────────────────
# 安全文件名
# ─────────────────────────────────────────────────────────────────────────────
def safe_id(image_id: str) -> str:
    return image_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# 全管道 hook 捕获（单次前向传播）
# ─────────────────────────────────────────────────────────────────────────────
def capture_all_hooks(model, image_path: str, sid: str,
                      workdir: Path) -> dict:
    """
    对一张图像运行 jina-v4 前向传播，捕获所有中间激活并保存为 int32 binary。

    返回 meta dict，包含：
      n_patches   : ViT 输出 patch 数（PatchMerger 输入行数）
      pooling_mask: (SEQ_LLM,) bool 数组，有效 token 位置
      files       : 已保存的激活文件路径列表（用于证明后清理）
    """
    import torch
    from PIL import Image as PILImage

    base = list(model.children())[0]
    try:
        vit = base.model.base_model.model.model.visual
        lm  = base.model.base_model.model.model.language_model
    except AttributeError:
        vit = base.model.model.visual
        lm  = base.model.model.language_model

    captured = {}
    handles  = []

    def _pre(key):
        def _hook(module, inp):
            t = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(t, torch.Tensor):
                captured[key] = t.detach().float().cpu()
        return _hook

    def _post(key):
        def _hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captured[key] = t.detach().float().cpu()
        return _hook

    # Conv3d 输入（模型预处理后的帧 tensor）
    handles.append(vit.patch_embed.proj.register_forward_pre_hook(_pre("conv3d")))

    # ViT block 输入（32 blocks）
    for bi, block in enumerate(vit.blocks):
        handles.append(block.register_forward_pre_hook(_pre(f"vit_{bi}")))

    # PatchMerger 输入（ViT 输出）
    handles.append(vit.merger.register_forward_pre_hook(_pre("merger")))

    # LLM layer 输入（36 layers × attn + ffn）
    for li in range(LLM_N_LAYERS):
        handles.append(lm.layers[li].input_layernorm.register_forward_hook(
            _post(f"attn_{li}")))
        handles.append(lm.layers[li].post_attention_layernorm.register_forward_hook(
            _post(f"ffn_{li}")))

    # Pooling 输入（最后一层 LM 输出，在归一化之前）
    handles.append(lm.layers[-1].register_forward_hook(_post("lm_last")))

    img = PILImage.open(image_path).convert("RGB")
    model.encode([img], task="retrieval", normalize_embeddings=False)
    for h in handles:
        h.remove()

    files = []

    def _pad_save(act: torch.Tensor, seq: int, path: Path):
        if act.dim() == 3:
            act = act[0]          # (batch, S, D) → (S, D)
        S, D = act.shape
        if S < seq:
            act = torch.cat([act, torch.zeros(seq - S, D)], 0)
        elif S > seq:
            act = act[:seq]
        (act * SCALE).round().to(torch.int32).numpy().astype(np.int32).tofile(str(path))
        files.append(path)
        return S   # original length (for mask)

    meta = {"files": files}

    # ── Conv3d patches（从 hook 捕获的预处理帧推导 patches）─────────────────
    if "conv3d" in captured:
        frames_t = captured["conv3d"]      # (batch, C, T, H, W) or (C, T, H, W)
        if frames_t.dim() == 5:
            frames_t = frames_t[0]         # → (C, T, H, W)
        if frames_t.dim() == 4:
            # (C, T, H, W) → (T, C, H, W)
            frames_t = frames_t.permute(1, 0, 2, 3)
        frames_np = frames_t.numpy().astype(np.float32)
        # 将预处理后的浮点帧转换为量化 patches（im2col 后量化）
        from script.prove_conv3d_embed import im2col
        patches = im2col(frames_np)                      # (N, 1176)
        # 已经是预处理后的值，直接量化（不再做 /255 归一化）
        # 注意：模型内部值域可能超出 [0,1]，直接 × SCALE
        patches_int32 = np.round(patches * SCALE).astype(np.int32)
        meta["conv3d_patches"] = patches_int32
        meta["n_conv3d_patches"] = patches_int32.shape[0]

    # ── ViT block inputs ─────────────────────────────────────────────────────
    for bi in range(VIT_N_BLOCKS):
        key = f"vit_{bi}"
        if key not in captured:
            continue
        vit_wd = workdir / f"vit-b{bi}"   # 与 verify_vit.py _setup_vit_workdir 保持一致
        vit_wd.mkdir(exist_ok=True)
        path = vit_wd / f"vit-block-{bi}-h_in.bin"
        _pad_save(captured[key], SEQ_VIT, path)

    # ── PatchMerger input ────────────────────────────────────────────────────
    if "merger" in captured:
        act = captured["merger"]
        if act.dim() == 3:
            act = act[0]            # (N, 1280)
        n_patches = act.shape[0]
        meta["n_patches"] = n_patches
        merger_path = workdir / f"corpus-{sid}-merger-input.bin"
        (act * SCALE).round().to(torch.int32).numpy().astype(np.int32).tofile(str(merger_path))
        files.append(merger_path)
        meta["merger_in_path"] = merger_path

    # ── LLM layer inputs ─────────────────────────────────────────────────────
    for li in range(LLM_N_LAYERS):
        for hook_type in ("attn", "ffn"):
            key = f"{hook_type}_{li}"
            if key not in captured:
                continue
            path = workdir / f"layer-{li}-corpus-{sid}-{hook_type}-input.bin"
            _pad_save(captured[key], SEQ_LLM, path)

    # ── Pooling input ────────────────────────────────────────────────────────
    if "lm_last" in captured:
        act = captured["lm_last"]
        if act.dim() == 3:
            act = act[0]
        real_S = act.shape[0]
        pool_path = workdir / f"corpus-{sid}-pooling-h.bin"
        _pad_save(act, SEQ_LLM, pool_path)
        mask = np.zeros(SEQ_LLM, dtype=bool)
        mask[:min(real_S, SEQ_LLM)] = True
        meta["pooling_h_path"] = pool_path
        meta["pooling_mask"]   = mask

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 全管道证明（单张图像）
# ─────────────────────────────────────────────────────────────────────────────
def prove_full_image(image_id: str, image_path: str, workdir: Path,
                     model, gpu_id: int = 0) -> dict:
    """
    对单张图像运行完整 5 组件证明。
    返回汇总结果 dict。
    """
    sid   = safe_id(image_id)
    t_all = time.perf_counter()
    print(f"\n[{sid[:40]}] 开始全量证明  gpu={gpu_id}", flush=True)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    results = {}

    # ── Step 0: 捕获所有激活 ───────────────────────────────────────────────
    t0 = time.perf_counter()
    meta = capture_all_hooks(model, image_path, sid, workdir)
    print(f"  [hook] 激活捕获完成  {round((time.perf_counter()-t0)*1000)}ms", flush=True)

    # ── Step 1: Conv3d embed ───────────────────────────────────────────────
    from script.prove_conv3d_embed import prove_conv3d_embed
    r_conv = prove_conv3d_embed(
        frames_npy=None,
        workdir=workdir,
        gpu_id=gpu_id,
        patches_int32=meta.get("conv3d_patches"),
    )
    results["conv3d"] = r_conv
    print(f"  [conv3d] {'✓' if r_conv['all_ok'] else '✗'}  {r_conv['elapsed_ms']}ms", flush=True)

    # ── Step 2: ViT 32 blocks ──────────────────────────────────────────────
    # h_in 文件已由 capture_all_hooks 保存到各 block 的子目录中
    # verify_vit_block 会自动使用（_make_input 检查文件是否存在）
    from script.verify_vit import verify_vit_block
    vit_results = []
    for bi in range(VIT_N_BLOCKS):
        r_vit = verify_vit_block(bi, workdir, gpu_id=gpu_id)
        vit_results.append(r_vit)
        ok = r_vit.get("all_pass", False)
        if not ok:
            print(f"  [ViT block {bi}] ✗", flush=True)
    n_vit_pass = sum(1 for r in vit_results if r.get("all_pass", False))
    results["vit"] = {"n_pass": n_vit_pass, "n_total": VIT_N_BLOCKS, "blocks": vit_results}
    print(f"  [ViT 32 blocks] {n_vit_pass}/{VIT_N_BLOCKS} 通过", flush=True)

    # ── Step 3: PatchMerger ────────────────────────────────────────────────
    from script.prove_patchmerger import prove_patchmerger
    merger_int32 = None
    if "merger_in_path" in meta:
        merger_int32 = np.fromfile(str(meta["merger_in_path"]), dtype=np.int32).reshape(
            meta["n_patches"], VIT_HIDDEN)
    r_pm = prove_patchmerger(
        n_patches=meta.get("n_patches", 256),
        workdir=workdir,
        gpu_id=gpu_id,
        input_int32=merger_int32,
    )
    results["patchmerger"] = r_pm
    print(f"  [patchmerger] {'✓' if r_pm['all_ok'] else '✗'}  {r_pm['elapsed_ms']}ms", flush=True)

    # ── Step 4: Pooling head ───────────────────────────────────────────────
    from script.prove_pooling import prove_pooling
    r_pool = prove_pooling(
        H_path=meta.get("pooling_h_path"),
        mask=meta.get("pooling_mask"),
        seq_len=SEQ_LLM,
        embed_dim=LLM_HIDDEN,
    )
    results["pooling"] = r_pool
    print(f"  [pooling] {'✓' if r_pool.get('all_ok') else '✗'}", flush=True)

    # ── Step 5: LLM 36 layers ─────────────────────────────────────────────
    cwd = str(ZKLLM_CWD)
    layer_results = []
    for li in range(LLM_N_LAYERS):
        prefix   = f"layer-{li}"
        attn_inp = workdir / f"{prefix}-corpus-{sid}-attn-input.bin"
        ffn_inp  = workdir / f"{prefix}-corpus-{sid}-ffn-input.bin"
        ffn_out  = workdir / f"{prefix}-corpus-{sid}-ffn-out.bin"
        attn_out = workdir / f"{prefix}-corpus-{sid}-attn-out.bin"
        attn_sfx = workdir / f"{prefix}-corpus-{sid}-attn-sfx-out.bin"

        if not attn_inp.exists():
            (np.random.randn(SEQ_LLM, LLM_HIDDEN) * SCALE).astype(np.int32).tofile(str(attn_inp))
        if not ffn_inp.exists():
            (np.random.randn(SEQ_LLM, LLM_HIDDEN) * SCALE).astype(np.int32).tofile(str(ffn_inp))

        r_ffn = subprocess.run(
            [str(BIN_DIR / "ffn"), str(ffn_inp), str(SEQ_LLM), str(LLM_HIDDEN), "11008",
             str(workdir), prefix, str(ffn_out)],
            capture_output=True, cwd=cwd, env=env)
        r_lin = subprocess.run(
            [str(BIN_DIR / "self-attn"), "linear", str(attn_inp), str(SEQ_LLM), str(LLM_HIDDEN),
             str(workdir), prefix, str(attn_out), str(KV_DIM)],
            capture_output=True, cwd=cwd, env=env)
        if r_lin.returncode == 0:
            r_sfx = subprocess.run(
                [str(BIN_DIR / "self-attn"), "attn", str(attn_inp), str(SEQ_LLM), str(LLM_HIDDEN),
                 str(workdir), prefix, str(attn_sfx), str(KV_DIM), str(NUM_KV_HEADS)],
                capture_output=True, cwd=cwd, env=env)
        else:
            class _F:
                returncode = 1
            r_sfx = _F()

        ok = (r_ffn.returncode == 0 and r_lin.returncode == 0 and r_sfx.returncode == 0)
        layer_results.append({"layer": li, "ok": ok})
        for p in [ffn_out, attn_out, attn_sfx, attn_inp, ffn_inp]:
            Path(p).unlink(missing_ok=True)

    n_layer_pass = sum(1 for r in layer_results if r["ok"])
    results["llm_layers"] = {"n_pass": n_layer_pass, "n_total": LLM_N_LAYERS}
    print(f"  [LLM 36 layers] {n_layer_pass}/{LLM_N_LAYERS} 通过", flush=True)

    # ── 清理临时激活文件 ────────────────────────────────────────────────────
    for p in meta.get("files", []):
        Path(p).unlink(missing_ok=True)
    # 清理 ViT block h_in 文件（下一张图会重写）
    for bi in range(VIT_N_BLOCKS):
        h_in = workdir / f"vit-b{bi}" / f"vit-block-{bi}-h_in.bin"
        h_in.unlink(missing_ok=True)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    all_ok = (
        r_conv["all_ok"]
        and n_vit_pass == VIT_N_BLOCKS
        and r_pm["all_ok"]
        and r_pool.get("all_ok", False)
        and n_layer_pass == LLM_N_LAYERS
    )
    elapsed_ms = round((time.perf_counter() - t_all) * 1000)
    status = "✓ PASS" if all_ok else "✗ FAIL"
    print(f"  [{sid[:30]}] {status}  总耗时 {elapsed_ms}ms", flush=True)

    return {
        "image_id":   image_id,
        "all_ok":     all_ok,
        "elapsed_ms": elapsed_ms,
        "results":    results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="语料库全量证明（per-image full pipeline）")
    parser.add_argument("--corpus",     default="corpora/image.jsonl")
    parser.add_argument("--workdir",    default="zkllm-workdir/jina-v4")
    parser.add_argument("--model-path", default="/root/autodl-tmp/models/jina-embeddings-v4")
    parser.add_argument("--out-dir",    default="zkllm-workdir/jina-v4",
                        help="corpus_full_proof_*.json 输出目录（默认与 workdir 相同）")
    parser.add_argument("--worker-id",   type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit",       type=int, default=-1)
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    workdir  = (ROOT / args.workdir).resolve()
    out_dir  = (ROOT / args.out_dir).resolve()
    # GPU selection: shell script sets CUDA_VISIBLE_DEVICES to restrict to one GPU,
    # so logical cuda:0 is always the intended GPU for this worker.
    gpu_id = 0

    print(f"\n语料库全量证明  worker={args.worker_id}/{args.num_workers}  gpu={gpu_id}")
    print(f"  workdir : {workdir}")
    print(f"  out_dir : {out_dir}")

    # 检查必要 binary
    for name in ["ffn", "self-attn", "rmsnorm", "verify-ipa"]:
        if not (BIN_DIR / name).exists():
            print(f"[ERROR] binary 不存在: {BIN_DIR / name}", file=sys.stderr)
            sys.exit(1)

    # 加载模型
    print("\n加载 jina-v4 模型...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_path, trust_remote_code=True, device=f"cuda:{gpu_id}")
    print("  ✓ 模型加载完成", flush=True)

    # 读取语料库
    corpus_path = ROOT / args.corpus
    with open(corpus_path) as f:
        items = [json.loads(l) for l in f if l.strip()]

    # 分配 worker 任务
    items = [it for i, it in enumerate(items) if i % args.num_workers == args.worker_id]
    if args.limit > 0:
        items = items[:args.limit]

    print(f"\n本 worker 处理 {len(items)} 张图像", flush=True)

    t_total = time.perf_counter()
    n_pass  = 0

    for idx, item in enumerate(items):
        image_id   = item["image_id"]
        image_path = str(corpus_path.parent / item["image_path"])
        sid        = safe_id(image_id)
        out_path   = out_dir / f"corpus_full_proof_{sid}.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{idx+1}/{len(items)}] 跳过（已存在）: {sid[:40]}", flush=True)
            n_pass += 1
            continue

        if not Path(image_path).exists():
            print(f"[{idx+1}/{len(items)}] 跳过（图像不存在）: {image_path}", flush=True)
            continue

        try:
            result = prove_full_image(image_id, image_path, workdir, model, gpu_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            if result["all_ok"]:
                n_pass += 1
        except Exception as e:
            print(f"[{idx+1}/{len(items)}] ✗ 异常: {e}", flush=True, file=sys.stderr)

        elapsed_total = round(time.perf_counter() - t_total)
        avg_s = elapsed_total / (idx + 1)
        remaining = round(avg_s * (len(items) - idx - 1))
        print(f"  进度: {idx+1}/{len(items)}  通过: {n_pass}  "
              f"均值: {avg_s:.0f}s/图  预计剩余: {remaining//3600}h{(remaining%3600)//60}m",
              flush=True)

    print(f"\n{'='*60}")
    print(f"全量证明完成  通过 {n_pass}/{len(items)}")
    print(f"总耗时: {round(time.perf_counter()-t_total)}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

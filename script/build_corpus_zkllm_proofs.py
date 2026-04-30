"""
语料库侧 zkLLM 证明预计算脚本 (Phase 3 - corpus side)

对 corpora/image.jsonl 中的每张图像，用 zkLLM 二进制对其 embedding 表示
的最后 K 层推理做 Sumcheck 证明，证明 embedding 计算过程可验证。

证明结果存储在：
  zkllm-workdir/jina-v4/corpus_proof_{safe_image_id}.json

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/build_corpus_zkllm_proofs.py \\
      --corpus corpora/image.jsonl \\
      --workdir zkllm-workdir/jina-v4 \\
      --k_layers 6 \\
      --overwrite

参数说明：
  --corpus   语料 jsonl 路径（每行含 image_id 字段）
  --workdir  zkLLM workdir（含已提交权重的目录）
  --k_layers 证明最后 K 层，默认 5（层31-35，图像 corpus 消融实验最优值）
  --limit    只处理前 N 张（调试用），默认全量
  --overwrite 覆盖已存在的证明文件
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── 项目根目录（该脚本在 UltraRAG/script/ 下） ────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
BIN_DIR = ROOT / "src" / "zkllm"
ZKLLM_CWD = ROOT / "src" / "zkllm"  # swiglu-table.bin 在这里

EMBED_DIM  = 2048
HIDDEN_DIM = 11008
KV_DIM      = 256
NUM_KV_HEADS = 2            # jina-v4 GQA：2 个 KV head（16 Q-head，group_size=8）
SCALE      = 1 << 16        # 量化系数，与实验 3.H.1 结论一致
SEQ_LEN_IMG = 1024          # 图像 seq_len=641，pad 到 1024（满足 NTT 约束）


def safe_id(image_id: str) -> str:
    """将 image_id 转换为合法文件名（替换路径分隔符和空格）"""
    return image_id.replace("/", "_").replace("\\", "_").replace(" ", "_")


def capture_image_hooks(model, image_path: str, sid: str,
                        workdir: Path, k_layers: int) -> bool:
    """
    对单张图像运行 jina-v4 前向推理，捕获最后 k_layers 层的激活并保存：
      layer-{l}-corpus-{sid}-attn-input.bin  ← input_layernorm 输出（self-attn 输入）
      layer-{l}-corpus-{sid}-ffn-input.bin   ← post_attention_layernorm 输出（ffn 输入）
    均 zero-pad 到 SEQ_LEN_IMG×2048 的 int32 数组。
    返回 True 表示成功，False 表示出错（调用方将回退随机基线）。
    """
    import torch
    from PIL import Image as PILImage

    try:
        layers = (list(model.children())[0]
                  .model.base_model.model.model.language_model.layers)
        captured = {}
        handles = []

        def _make_hook(key):
            def _hook(module, inp, out):
                captured[key] = out.detach().float().cpu()
            return _hook

        for li in range(36 - k_layers, 36):
            handles.append(
                layers[li].input_layernorm.register_forward_hook(
                    _make_hook(f"attn_{li}")))
            handles.append(
                layers[li].post_attention_layernorm.register_forward_hook(
                    _make_hook(f"ffn_{li}")))

        img = PILImage.open(image_path).convert("RGB")
        model.encode([img], task="retrieval", normalize_embeddings=False)

        for h in handles:
            h.remove()

        for li in range(36 - k_layers, 36):
            for hook_type in ("attn", "ffn"):
                key = f"{hook_type}_{li}"
                if key not in captured:
                    continue
                act = captured[key]
                if act.dim() == 2:
                    act = act.unsqueeze(0)
                a = act[0]                        # (S, D)
                S, D = a.shape
                if S < SEQ_LEN_IMG:
                    a = torch.cat([a, torch.zeros(SEQ_LEN_IMG - S, D)], dim=0)
                elif S > SEQ_LEN_IMG:
                    a = a[:SEQ_LEN_IMG]
                out_path = workdir / f"layer-{li}-corpus-{sid}-{hook_type}-input.bin"
                (a * SCALE).round().to(torch.int32).numpy().astype(
                    np.int32).tofile(str(out_path))
        return True

    except Exception as e:
        print(f"  !! capture_image_hooks 失败 ({image_path}): {e}", file=sys.stderr)
        return False


def prove_one(image_id: str, image_path: str, workdir: Path,
              k_layers: int, model=None, zkllm_cwd: Path = None,
              proc_env: dict = None,
              ffn_cwd: Path = None, ffn_env: dict = None) -> dict:
    """
    对单张图像的 embedding 运行 zkLLM 最后 K 层证明。
    若 model 不为 None，则使用真实 hook 激活；否则回退随机基线。

    双卡并行模式（ffn_cwd/ffn_env 非 None）：
      FFN 在 ffn_cwd/ffn_env 指定的 GPU 上运行，
      Attn-linear + zkAttn 在 zkllm_cwd/proc_env 指定的 GPU 上运行。
      两者并发执行，zkAttn 等待 Attn-linear 完成后再启动。
      每层时间从 FFN+linear+zkAttn 串行 → max(FFN, linear+zkAttn) 并行。
    """
    cwd = str(zkllm_cwd) if zkllm_cwd else str(ZKLLM_CWD)
    env = proc_env  # None = 继承当前环境（单卡模式）
    # 双卡模式：FFN 用独立的 GPU/cwd
    dual = ffn_cwd is not None
    f_cwd = str(ffn_cwd) if dual else cwd
    f_env = ffn_env if dual else env

    sid = safe_id(image_id)
    t_start = time.perf_counter()
    layer_results = []
    failed = False

    # 捕获真实激活
    use_real = False
    if model is not None and image_path and Path(image_path).exists():
        use_real = capture_image_hooks(model, image_path, sid, workdir, k_layers)

    # 随机基线也必须用 SEQ_LEN_IMG=1024：attn 模式的 NTT 约束要求 seq²=2²⁰
    seq = SEQ_LEN_IMG

    for layer_idx in range(36 - k_layers, 36):
        prefix = f"layer-{layer_idx}"

        if use_real:
            attn_inp = workdir / f"{prefix}-corpus-{sid}-attn-input.bin"
            ffn_inp  = workdir / f"{prefix}-corpus-{sid}-ffn-input.bin"
            # 若某层激活缺失，单独回退随机
            if not attn_inp.exists():
                np.random.seed(abs(hash(str(attn_inp))) % (2**31))
                (np.random.randn(seq, EMBED_DIM) * 65536).astype(np.int32).tofile(str(attn_inp))
            if not ffn_inp.exists():
                np.random.seed(abs(hash(str(ffn_inp))) % (2**31))
                (np.random.randn(seq, EMBED_DIM) * 65536).astype(np.int32).tofile(str(ffn_inp))
        else:
            # 随机基线：两个 binary 共用同一输入（与旧逻辑等价）
            attn_inp = workdir / f"{prefix}-corpus-{sid}-input.bin"
            ffn_inp  = attn_inp
            if not attn_inp.exists():
                np.random.seed(abs(hash(str(attn_inp))) % (2**31))
                (np.random.randn(seq, EMBED_DIM) * 65536).astype(np.int32).tofile(str(attn_inp))

        ffn_out  = workdir / f"{prefix}-corpus-{sid}-ffn-out.bin"
        attn_out = workdir / f"{prefix}-corpus-{sid}-attn-out.bin"
        attn_sfx_out = workdir / f"{prefix}-corpus-{sid}-attn-sfx-out.bin"

        if dual:
            # ── 双卡并行：FFN(GPU0) ∥ [Attn-linear → zkAttn](GPU1) ──
            # 时序：GPU0: FFN≈6.4s；GPU1: linear≈2.6s → zkAttn≈6.2s（共8.8s）
            # 墙钟时间 = max(6.4s, 8.8s) = 8.8s/层，比单卡串行15.2s快约1.7×
            def _run_ffn():
                return subprocess.run(
                    [str(BIN_DIR / "ffn"),
                     str(ffn_inp), str(seq), str(EMBED_DIM), str(HIDDEN_DIM),
                     str(workdir), prefix, str(ffn_out)],
                    capture_output=True, cwd=f_cwd, env=f_env)

            def _run_linear_then_attn():
                # 清理 stale temp 文件，防止上次中断遗留的错误尺寸数据
                for _tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                    (Path(cwd) / _tmp).unlink(missing_ok=True)
                r_lin = subprocess.run(
                    [str(BIN_DIR / "self-attn"), "linear",
                     str(attn_inp), str(seq), str(EMBED_DIM),
                     str(workdir), prefix, str(attn_out), str(KV_DIM)],
                    capture_output=True, cwd=cwd, env=env)
                if r_lin.returncode != 0:
                    return r_lin, None
                r_sfx = subprocess.run(
                    [str(BIN_DIR / "self-attn"), "attn",
                     str(attn_inp), str(seq), str(EMBED_DIM),
                     str(workdir), prefix, str(attn_sfx_out),
                     str(KV_DIM), str(NUM_KV_HEADS)],
                    capture_output=True, cwd=cwd, env=env)
                return r_lin, r_sfx

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                fut_ffn    = ex.submit(_run_ffn)
                fut_gpu1   = ex.submit(_run_linear_then_attn)
                r_ffn          = fut_ffn.result()
                r_attn, r_attn_sfx = fut_gpu1.result()

            # 若 linear 失败，r_attn_sfx 为 None，补一个假 returncode
            if r_attn_sfx is None:
                class _Fail:
                    returncode = 1
                    stderr = b"linear failed, attn skipped"
                r_attn_sfx = _Fail()
        else:
            # ── 单卡串行（原有逻辑）──
            r_ffn = subprocess.run(
                [str(BIN_DIR / "ffn"),
                 str(ffn_inp), str(seq), str(EMBED_DIM), str(HIDDEN_DIM),
                 str(workdir), prefix, str(ffn_out)],
                capture_output=True, cwd=cwd, env=env)

            # 清理 stale temp 文件，防止上次中断遗留的错误尺寸数据
            for _tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                (Path(cwd) / _tmp).unlink(missing_ok=True)
            r_attn = subprocess.run(
                [str(BIN_DIR / "self-attn"), "linear",
                 str(attn_inp), str(seq), str(EMBED_DIM),
                 str(workdir), prefix, str(attn_out), str(KV_DIM)],
                capture_output=True, cwd=cwd, env=env)

            r_attn_sfx = subprocess.run(
                [str(BIN_DIR / "self-attn"), "attn",
                 str(attn_inp), str(seq), str(EMBED_DIM),
                 str(workdir), prefix, str(attn_sfx_out),
                 str(KV_DIM), str(NUM_KV_HEADS)],
                capture_output=True, cwd=cwd, env=env)

        ok = (r_ffn.returncode == 0 and r_attn.returncode == 0
              and r_attn_sfx.returncode == 0)
        layer_results.append({
            "layer": layer_idx,
            "verified": ok,
            "ffn_rc": r_ffn.returncode,
            "attn_linear_rc": r_attn.returncode,
            "attn_sfx_rc": r_attn_sfx.returncode,
        })
        if not ok:
            failed = True
            stderr_ffn      = r_ffn.stderr.decode(errors="replace")[-200:]
            stderr_attn     = r_attn.stderr.decode(errors="replace")[-200:]
            stderr_attn_sfx = r_attn_sfx.stderr.decode(errors="replace")[-200:]
            print(f"  !! Layer {layer_idx} failed — "
                  f"ffn: {stderr_ffn} | attn_linear: {stderr_attn} | attn_sfx: {stderr_attn_sfx}",
                  file=sys.stderr)

        # 清理中间文件
        cleanup = [ffn_out, attn_out, attn_sfx_out]
        if use_real:
            cleanup += [attn_inp, ffn_inp]
        for p in cleanup:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    elapsed_ms = round((time.perf_counter() - t_start) * 1000)
    return {
        "image_id": image_id,
        "status": "failed" if failed else "completed",
        "k_layers": k_layers,
        "layers": layer_results,
        "verified": not failed,
        "elapsed_ms": elapsed_ms,
        "use_real_hooks": use_real,
    }


def main():
    parser = argparse.ArgumentParser(description="预计算语料库侧 zkLLM 证明")
    parser.add_argument("--corpus",   default="corpora/image.jsonl")
    parser.add_argument("--workdir",  default="zkllm-workdir/jina-v4")
    parser.add_argument("--k_layers", type=int, default=36,
                        help="证明最后 K 层（全量=36；原消融实验推荐 K=5）")
    parser.add_argument("--limit",    type=int, default=-1,
                        help="只处理前 N 张（-1 = 全量）")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-model", action="store_true",
                        help="跳过模型加载，使用随机基线（调试用）")
    parser.add_argument("--worker-id",   type=int, default=0,
                        help="当前 worker 编号（0-based），与 --num-workers 配合实现多卡并行")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="总 worker 数，每张卡一个 worker；每个 worker 处理 items[worker_id::num_workers]")
    parser.add_argument("--dual-gpu", action="store_true",
                        help="每张图像内部双卡并行：FFN→GPU0，Attn-linear+zkAttn→GPU1\n"
                             "与 --num-workers 2 互斥（两者都设时忽略 --dual-gpu）")
    args = parser.parse_args()

    # ── 关键：在 torch 初始化之前设置 CUDA_VISIBLE_DEVICES ──
    # 多 worker 时每个进程只暴露自己负责的 GPU，避免跨设备张量冲突。
    # 设置后该进程内 cuda:0 = 物理 GPU{worker_id}，子进程继承相同映射。
    if args.num_workers > 1 or args.dual_gpu:
        target_gpu = str(args.worker_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
        print(f"[GPU] CUDA_VISIBLE_DEVICES={target_gpu} (worker {args.worker_id})")

    corpus_path = ROOT / args.corpus
    workdir     = ROOT / args.workdir

    if not corpus_path.exists():
        print(f"[Error] corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)
    if not workdir.exists():
        print(f"[Error] workdir not found: {workdir}", file=sys.stderr)
        sys.exit(1)

    # 检查 K 层权重是否已提交
    first_layer = 36 - args.k_layers
    probe = workdir / f"layer-{first_layer}-self_attn.q_proj.weight-commitment.bin"
    if not probe.exists():
        print(f"[Error] K={args.k_layers} 层权重未提交，请先运行 load_jina_weights.py",
              file=sys.stderr)
        sys.exit(1)

    # 读取语料，按 worker 分片（round-robin：worker i 处理索引 i, i+N, i+2N, ...）
    items = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit > 0:
        items = items[:args.limit]
    if args.num_workers > 1:
        items = items[args.worker_id::args.num_workers]

    # ── cwd 和 GPU 分配 ──────────────────────────────────────────────────────
    def _make_worker_cwd(slot: int) -> Path:
        """创建 worker{slot}/ 子目录，symlink swiglu-table.bin。"""
        d = ZKLLM_CWD / f"worker{slot}"
        d.mkdir(exist_ok=True)
        link = d / "swiglu-table.bin"
        src  = ZKLLM_CWD / "swiglu-table.bin"
        if src.exists() and not link.exists():
            link.symlink_to(src.resolve())
        return d

    use_dual = args.dual_gpu and args.num_workers == 1  # 多 worker 时忽略 --dual-gpu

    if args.num_workers > 1:
        # 模式 A：语料分片，每个 worker 独占一张卡（顺序执行，互不干扰）
        # CUDA_VISIBLE_DEVICES 已在进程级设好，子进程继承即可（不再重复设）
        worker_cwd = _make_worker_cwd(args.worker_id)
        proc_env   = dict(os.environ)   # 已含正确的 CUDA_VISIBLE_DEVICES
        ffn_cwd_arg, ffn_env_arg = None, None
    elif use_dual:
        # 模式 B：每张图像内部双卡并行，FFN→GPU0，Attn-linear+zkAttn→GPU1
        worker_cwd   = _make_worker_cwd(1)   # Attn/zkAttn 用 GPU1
        proc_env     = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}
        ffn_cwd_arg  = _make_worker_cwd(0)   # FFN 用 GPU0
        ffn_env_arg  = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    else:
        # 模式 C：单卡顺序（默认）
        worker_cwd  = ZKLLM_CWD
        proc_env    = None
        ffn_cwd_arg = None
        ffn_env_arg = None

    # 加载 jina-v4 用于 hook 捕获（--no-model 时跳过，回退随机基线）
    model = None
    if not getattr(args, "no_model", False):
        model_path = "/root/autodl-tmp/models/jina-embeddings-v4"
        # CUDA_VISIBLE_DEVICES 已设好，该进程的唯一可见 GPU 就是 cuda:0
        model_device = "cuda:0"
        try:
            from sentence_transformers import SentenceTransformer
            print(f"── 加载 jina-v4（device={model_device}）──")
            model = SentenceTransformer(model_path, trust_remote_code=True, device=model_device)
            print("  OK")
        except Exception as e:
            print(f"  警告：模型加载失败，将使用随机基线 ({e})", file=sys.stderr)

    print(f"\n语料库侧 zkLLM 证明预计算")
    print(f"  corpus      : {corpus_path} ({len(items)} 张图像)")
    print(f"  workdir     : {workdir}")
    print(f"  K 层        : {args.k_layers} (层 {first_layer}–35)")
    print(f"  overwrite   : {args.overwrite}")
    mode_str = ("分片多卡" if args.num_workers > 1
                else "双卡并行" if use_dual else "单卡顺序")
    print(f"  worker      : {args.worker_id}/{args.num_workers}  模式={mode_str}")
    print(f"  hook 激活   : {'真实 (jina-v4)' if model else '随机基线'}")
    print()

    ok_count = skip_count = fail_count = 0

    for i, item in enumerate(items):
        image_id = item.get("image_id", str(item.get("id", i)))
        sid      = safe_id(image_id)
        out_path = workdir / f"corpus_proof_{sid}.json"

        if out_path.exists() and not args.overwrite:
            skip_count += 1
            if i % 20 == 0:
                print(f"[{i+1}/{len(items)}] {image_id} — skip (already exists)")
            continue

        # 解析图像绝对路径（corpus JSONL 中 image_path 相对于 corpus 父目录）
        image_path = None
        if "image_path" in item:
            image_path = str(corpus_path.parent / item["image_path"])

        print(f"[{i+1}/{len(items)}] {image_id} ...", end=" ", flush=True)
        proof = prove_one(image_id, image_path, workdir, args.k_layers, model,
                          zkllm_cwd=worker_cwd, proc_env=proc_env,
                          ffn_cwd=ffn_cwd_arg, ffn_env=ffn_env_arg)

        with open(out_path, "w") as f:
            json.dump(proof, f, indent=2)

        hook_tag = "(real)" if proof.get("use_real_hooks") else "(rand)"
        status   = "✓" if proof["verified"] else "✗"
        print(f"{status} {hook_tag} {proof['elapsed_ms']}ms")

        if proof["verified"]:
            ok_count += 1
        else:
            fail_count += 1

    print()
    print(f"完成：{ok_count} 成功 / {fail_count} 失败 / {skip_count} 跳过")
    print(f"证明文件保存在: {workdir}/corpus_proof_*.json")


if __name__ == "__main__":
    main()

"""
层敏感性消融实验 (Layer Sensitivity Ablation)

实验方案（2026-03-31）：
  实验 A：BI Score（Block Influence，ShortGPT arXiv:2403.03853）
    对每层注册 pre/post hook，捕获输入/输出 hidden state，
    计算 BI_l = 1 - cosine(x_in, x_out)（逐 token 后取均值）。
    不扰动模型输出，无需多次前向传播，开销最低。
    文本 + 图像分别计算，对比两种模态的规律是否一致。

  实验 B：单层残差置零（已实现）
    对每层将残差置零（令 x_{l+1} = x_l），测 embedding cos 下降。
    文本 + 图像各一组。

  实验 C 旧版：噪声注入（自适应高斯噪声，有级联传播问题）
    对每层注入 N(0, (scale*std(x_out))²) 噪声，测 embedding cos 下降。
    注意：噪声会经后续层级联传播，早期层天然显得更敏感（传播距离更长）。

  实验 C 完整版：因果追踪（Causal Tracing）
    步骤1：干净前向，记录 clean_acts[l]（各层干净输出）
    步骤2：腐化前向（第 0 层注入大噪声），得到 E_corrupt
    步骤3：对每层 l，腐化前向 + 强制将层 l 输出替换为 clean_acts[l]，得 E_restore_l
    指标：recovery_l = cos(E_restore_l, E_base) − cos(E_corrupt, E_base)
    解读：recovery 高 → 该层是正确 embedding 的关键节点 → 值得被 prove
    参考：Meng et al. ROME NeurIPS 2022；Elhage et al. Transformer Circuits 2021

输出：
  notes/ablation_bi_score.json/png        — 实验 A 结果
  notes/ablation_layer_sensitivity.json   — 实验 B 结果
  notes/ablation_layer_sensitivity.png    — 实验 B 图表
  notes/ablation_layer_sensitivity.md     — 综合结论

用法：
  python script/ablation_layer_sensitivity.py --mode bi       # 只跑实验A
  python script/ablation_layer_sensitivity.py --mode zero     # 只跑实验B
  python script/ablation_layer_sensitivity.py --mode noise    # 只跑实验C
  python script/ablation_layer_sensitivity.py --mode all      # 全部
  python script/ablation_layer_sensitivity.py --mode noise --noise-scale 0.5  # 半强度噪声
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

MODEL_PATH  = "/root/autodl-tmp/models/jina-embeddings-v4"
OUT_DIR     = ROOT / "notes"
CORPUS_PATH = ROOT / "corpora" / "image.jsonl"

# 文本样本（检索 query 风格）
SAMPLE_TEXTS = [
    "What is the main topic of this document?",
    "Explain the key technical contributions.",
    "How does the proposed method compare to baselines?",
    "What are the limitations of this approach?",
    "Describe the experimental setup and datasets used.",
    "What metrics are used for evaluation?",
    "Summarize the introduction section.",
    "What future work is suggested by the authors?",
    "How is the model trained and fine-tuned?",
    "What is the computational complexity of the method?",
    "What datasets are used in the experiments?",
    "Describe the network architecture in detail.",
    "What are the main results reported in the paper?",
    "How does attention mechanism work in this model?",
    "What is the training objective or loss function?",
    "Compare the inference speed with other methods.",
    "What preprocessing steps are applied to the data?",
    "Describe any ablation studies conducted.",
    "What hardware was used for experiments?",
    "What is the memory footprint of the model?",
]


def load_image_samples(n: int = 20) -> list:
    """从 corpora/image.jsonl 随机采样 n 张图像，返回 PIL.Image 列表。"""
    import json as _json
    from PIL import Image as PILImage
    import random

    if not CORPUS_PATH.exists():
        print(f"  [警告] corpus 不存在：{CORPUS_PATH}，跳过图像样本")
        return []

    items = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(_json.loads(line))

    random.seed(42)
    sampled = random.sample(items, min(n, len(items)))

    images = []
    for item in sampled:
        img_path = CORPUS_PATH.parent / item.get("image_path", "")
        if img_path.exists():
            try:
                images.append(PILImage.open(img_path).convert("RGB"))
            except Exception:
                pass
    print(f"  加载图像样本：{len(images)} 张（从 {len(sampled)} 条记录）")
    return images


def load_model(device="cuda:0"):
    from sentence_transformers import SentenceTransformer
    print(f"加载 jina-v4 ({device})...")
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device=device)
    print("  OK")
    return model


def get_lm_layers(model):
    """返回 jina-v4 语言模型的 36 个 transformer 层列表。"""
    return (list(model.children())[0]
            .model.base_model.model.model.language_model.layers)


def run_bi_score(model, texts=None, images=None, device="cuda:0") -> list:
    """
    实验 A：Block Influence (BI) Score
    参考：ShortGPT (arXiv:2403.03853, ACL 2025 Findings)

    对每层注册 pre-hook 捕获输入 hidden state，post-hook 捕获输出 hidden state，
    计算 BI_l = 1 - mean_over_tokens( cosine(x_in_t, x_out_t) )

    BI 高 → 输入/输出差异大 → 这层变换量大 → 不冗余
    BI 低 → 输入≈输出 → 这层冗余 → ShortGPT 优先删除

    文本和图像分别计算，返回两组结果的均值（若两者都提供）。
    """
    import torch

    layers = get_lm_layers(model)
    n_layers = len(layers)

    # 每层存 (x_in, x_out) 列表，每个元素是 (T, D) tensor
    layer_inputs  = [[] for _ in range(n_layers)]
    layer_outputs = [[] for _ in range(n_layers)]

    def _make_pre(idx):
        def _hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is not None:
                layer_inputs[idx].append(h.detach().float())
        return _hook

    def _make_post(idx):
        def _hook(module, args, kwargs, out):
            h = out[0] if isinstance(out, tuple) else out
            if h is not None:
                layer_outputs[idx].append(h.detach().float())
        return _hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(_make_pre(i), with_kwargs=True))
        handles.append(layer.register_forward_hook(_make_post(i), with_kwargs=True))

    try:
        # 文本前向（用 SentenceTransformer.encode，hooks 已挂在内部层上）
        if texts:
            print(f"  BI Score：前向传播文本样本（{len(texts)} 个）...")
            with torch.no_grad():
                model.encode(texts, task="retrieval",
                             normalize_embeddings=False,
                             batch_size=4, show_progress_bar=False)
        # 图像前向
        if images:
            print(f"  BI Score：前向传播图像样本（{len(images)} 张）...")
            with torch.no_grad():
                model.encode(images, task="retrieval",
                             normalize_embeddings=False,
                             batch_size=2, show_progress_bar=False)
    finally:
        for h in handles:
            h.remove()

    results = []
    for i in range(n_layers):
        if not layer_inputs[i] or not layer_outputs[i]:
            results.append({"layer": i, "bi_score": None})
            continue

        # 拼接所有 batch 的 token hidden states：(total_tokens, D)
        x_in  = torch.cat([t.reshape(-1, t.shape[-1]) for t in layer_inputs[i]],  dim=0)
        x_out = torch.cat([t.reshape(-1, t.shape[-1]) for t in layer_outputs[i]], dim=0)

        # 逐 token 余弦相似度，再取均值
        cos = torch.nn.functional.cosine_similarity(x_in, x_out, dim=-1)
        bi  = float(1.0 - cos.mean().item())
        results.append({
            "layer":    i,
            "bi_score": round(bi, 6),
        })
        print(f"  Layer {i:2d}: BI={bi:.4f}  ({'高变换' if bi > 0.1 else '中' if bi > 0.01 else '冗余'})")

    return results


def encode_texts(model, texts, task="retrieval"):
    """编码文本列表，返回 numpy (N, D) embedding 矩阵。"""
    embs = model.encode(texts, task=task, normalize_embeddings=True,
                        batch_size=4, show_progress_bar=False)
    return np.array(embs)


def encode_images(model, images, task="retrieval"):
    """编码 PIL Image 列表，返回 numpy (N, D) embedding 矩阵。"""
    embs = model.encode(images, task=task, normalize_embeddings=True,
                        batch_size=2, show_progress_bar=False)
    return np.array(embs)


def cosine_sim_mean(a: np.ndarray, b: np.ndarray) -> float:
    """逐行余弦相似度均值（两矩阵已归一化时等于点积均值）。"""
    return float(np.sum(a * b, axis=1).mean())


def run_ablation_texts(model, texts, device="cuda:0"):
    """
    对每层做消融：通过 hook 将该层的残差贡献置零，
    计算 embedding 与 baseline 的余弦相似度下降量。
    返回 list[dict]，每元素含 layer、cos_sim、sensitivity。
    """
    layers = get_lm_layers(model)
    n_layers = len(layers)
    print(f"  共 {n_layers} 层，消融文本样本数={len(texts)}")

    # baseline
    baseline = encode_texts(model, texts)

    results = []
    for l in range(n_layers):
        # hook：将 layer l 的输出替换为其输入（即令残差贡献=0）
        # transformer 层输入是 (hidden_states, ...) 元组，输出同样是元组
        handle = None

        def _zero_residual(module, args, kwargs, out):
            # jina-v4 decoder layer 用 kwargs 传 hidden_states，args 为空
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return out
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        handle = layers[l].register_forward_hook(_zero_residual, with_kwargs=True)
        try:
            ablated = encode_texts(model, texts)
        finally:
            handle.remove()

        cos = cosine_sim_mean(baseline, ablated)
        sensitivity = 1.0 - cos
        results.append({
            "layer": l,
            "cos_sim": round(cos, 6),
            "sensitivity": round(sensitivity, 6),
        })
        print(f"  Layer {l:2d}: cos={cos:.4f}  sensitivity={sensitivity:.4f}")

    return results


def run_ablation_images(model, images, device="cuda:0"):
    """
    对每层做消融（图像模态）：将该层残差置零，
    计算 embedding 与 baseline 的余弦相似度下降量。
    """
    layers = get_lm_layers(model)
    n_layers = len(layers)
    print(f"  共 {n_layers} 层，消融图像样本数={len(images)}")

    baseline = encode_images(model, images)

    results = []
    for l in range(n_layers):
        def _zero_residual(module, args, kwargs, out):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return out
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        handle = layers[l].register_forward_hook(_zero_residual, with_kwargs=True)
        try:
            ablated = encode_images(model, images)
        finally:
            handle.remove()

        cos = cosine_sim_mean(baseline, ablated)
        sensitivity = 1.0 - cos
        results.append({
            "layer": l,
            "cos_sim": round(cos, 6),
            "sensitivity": round(sensitivity, 6),
        })
        print(f"  Layer {l:2d}: cos={cos:.4f}  sensitivity={sensitivity:.4f}")

    return results


def run_noise_injection(model, samples, encode_fn, noise_scale=1.0, label=""):
    """
    实验 C：对每层注入自适应高斯噪声。
    noise ~ N(0, (noise_scale * std(x_out))^2)，与该层输出同量级。
    相比残差置零（彻底清除该层变换），噪声注入是更温和的扰动，
    能区分层是否在做有意义的结构化变换。
    返回 list[dict]，每元素含 layer、cos_sim、sensitivity、noise_std。
    """
    layers = get_lm_layers(model)
    n_layers = len(layers)
    print(f"  共 {n_layers} 层，噪声注入{label}，样本数={len(samples)}, scale={noise_scale}")

    baseline = encode_fn(model, samples)

    results = []
    for l in range(n_layers):
        noise_std_record = []

        def _inject_noise(module, args, kwargs, out, _rec=noise_std_record, _scale=noise_scale):
            h = out[0] if isinstance(out, tuple) else out
            if h is None:
                return out
            std = float(h.detach().float().std())
            _rec.append(std)
            noise = torch.randn_like(h) * (std * _scale)
            noisy = h + noise
            if isinstance(out, tuple):
                return (noisy,) + out[1:]
            return noisy

        handle = layers[l].register_forward_hook(_inject_noise, with_kwargs=True)
        try:
            ablated = encode_fn(model, samples)
        finally:
            handle.remove()

        cos = cosine_sim_mean(baseline, ablated)
        sensitivity = 1.0 - cos
        mean_std = float(np.mean(noise_std_record)) if noise_std_record else 0.0
        results.append({
            "layer": l,
            "cos_sim": round(cos, 6),
            "sensitivity": round(sensitivity, 6),
            "noise_std": round(mean_std, 6),
        })
        print(f"  Layer {l:2d}: cos={cos:.4f}  sensitivity={sensitivity:.4f}  σ={mean_std:.4f}")

    return results


def run_causal_tracing(model, samples, encode_fn, corrupt_scale=3.0, label=""):
    """
    实验 C 完整版：因果追踪（Causal Tracing）。

    方法（Meng et al. ROME, NeurIPS 2022 思路，适配 embedding 任务）：
      步骤1  干净前向：记录所有层的干净输出 clean_acts[l]，得到 E_base
      步骤2  腐化前向：在第 0 层注入大幅噪声（scale=corrupt_scale × σ），
             让腐化信号级联传播过所有层，得到 E_corrupt
      步骤3  对每层 l：腐化前向（第 0 层仍注入噪声）+ 在第 l 层将输出
             强制替换为 clean_acts[l]，使 l+1..35 在干净激活上运行，
             得到 E_restore_l
      指标   recovery_l = cos(E_restore_l, E_base) - cos(E_corrupt, E_base)
             recovery_l → 0：恢复层 l 对 embedding 无帮助（层不关键）
             recovery_l → 1：仅恢复层 l 即可完全修复 embedding（层关键）

    相比实验 C 原版（cascade 传播），causal tracing 隔离了单层效应：
    "仅有层 l 的计算是正确的，其余均腐化" → 直接衡量该层的不可替代性。

    参考文献：
      Meng et al., "Locating and Editing Factual Associations in GPT", NeurIPS 2022
      Elhage et al., "A Mathematical Framework for Transformer Circuits", Anthropic 2021
    """
    layers = get_lm_layers(model)
    n_layers = len(layers)
    print(f"  共 {n_layers} 层，因果追踪{label}，样本数={len(samples)}, corrupt_scale={corrupt_scale}")

    # 强制单 batch：所有样本一次 forward，确保每次 hook 触发时 seq_len 完全一致，
    # 避免不同 mini-batch padding 长度不同导致 clean_acts[l] 与恢复运行形状不匹配。
    def _encode(model, samps):
        return np.array(model.encode(
            samps, task="retrieval", normalize_embeddings=True,
            batch_size=len(samps), show_progress_bar=False,
        ))

    # ── 步骤 1：干净前向，记录每层干净输出 ────────────────────────────────────
    clean_acts = [None] * n_layers

    def _make_capture(idx):
        def _hook(module, args, kwargs, out):
            h = out[0] if isinstance(out, tuple) else out
            if h is not None:
                clean_acts[idx] = h.detach().clone()
            return out
        return _hook

    cap_hooks = [layers[l].register_forward_hook(_make_capture(l), with_kwargs=True)
                 for l in range(n_layers)]
    baseline = _encode(model, samples)
    for h in cap_hooks:
        h.remove()

    # ── 步骤 2：腐化前向（第 0 层注入大噪声） ───────────────────────────────
    corrupt_std_ref = []

    def _corrupt_layer0(module, args, kwargs, out):
        h = out[0] if isinstance(out, tuple) else out
        if h is None:
            return out
        std = float(h.detach().float().std())
        corrupt_std_ref.append(std)
        noise = torch.randn_like(h) * (std * corrupt_scale)
        noisy = h + noise
        return (noisy,) + out[1:] if isinstance(out, tuple) else noisy

    hc = layers[0].register_forward_hook(_corrupt_layer0, with_kwargs=True)
    corrupted_emb = _encode(model, samples)
    hc.remove()
    cos_corrupt = cosine_sim_mean(baseline, corrupted_emb)
    print(f"  腐化 embedding cos={cos_corrupt:.4f}（σ_layer0≈{np.mean(corrupt_std_ref):.2f}, scale={corrupt_scale}）")

    # ── 步骤 3：逐层恢复运行 ─────────────────────────────────────────────────
    results = []
    for l in range(n_layers):
        # hook 注册顺序：先 corrupt（layer0），再 restore（layer l）
        # 若 l==0，corrupt+restore 同在 layer0，restore 后于 corrupt 执行，净效果 = 干净输出
        h_corrupt = layers[0].register_forward_hook(_corrupt_layer0, with_kwargs=True)

        def _restore(module, args, kwargs, out, _l=l):
            ca = clean_acts[_l]
            if ca is None:
                return out
            if isinstance(out, tuple):
                return (ca,) + out[1:]
            return ca

        h_restore = layers[l].register_forward_hook(_restore, with_kwargs=True)

        try:
            restored_emb = _encode(model, samples)
        finally:
            h_corrupt.remove()
            h_restore.remove()

        cos_restore = cosine_sim_mean(baseline, restored_emb)
        recovery = cos_restore - cos_corrupt          # 0 = 无帮助，~1 = 完全恢复
        results.append({
            "layer":         l,
            "cos_corrupt":   round(cos_corrupt,  6),
            "cos_restored":  round(cos_restore,  6),
            "recovery":      round(recovery,     6),
        })
        print(f"  Layer {l:2d}: recovery={recovery:.4f}"
              f"  (corrupt={cos_corrupt:.4f} → restore={cos_restore:.4f})")

    return results


def run_lastk_ablation(model, texts, device="cuda:0"):
    """
    最后K层消融（K递减曲线）：
    对每个 K（从 n_layers 递减到 1），将前 (n_layers-K) 层全部置零，
    只保留最后 K 层的计算，测量 embedding 相对完整模型的余弦相似度。

    K=n_layers → cos=1.0（完整模型）
    K=6        → 只有最后6层（前30层置零）的质量
    K=1        → 只有最后1层的质量

    用途：绘制「K vs 质量」曲线，直观展示：
      - K 太小时质量断崖式下跌（不够）
      - K 超过拐点后边际收益趋近于零（多了无用）
    """
    layers = get_lm_layers(model)
    n_layers = len(layers)
    print(f"  共 {n_layers} 层，K 递减消融（{n_layers}→1），样本数={len(texts)}")

    baseline = encode_texts(model, texts)
    results = []

    for K in range(n_layers, 0, -1):
        first_proven = n_layers - K   # 第一个保留层的索引
        handles = []
        for li in range(0, first_proven):   # 置零前 (n_layers-K) 层
            def _zero(module, args, kwargs, out, _li=li):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    return out
                return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
            handles.append(layers[li].register_forward_hook(_zero, with_kwargs=True))
        try:
            ablated = encode_texts(model, texts)
        finally:
            for h in handles:
                h.remove()

        cos = cosine_sim_mean(baseline, ablated)
        results.append({
            "K": K,
            "first_layer": first_proven,
            "cos_sim": round(cos, 6),
            "quality_retention": round(cos, 6),   # cos vs full model
        })
        print(f"  K={K:2d} (层{first_proven:2d}-35 保留): cos={cos:.4f}")

    # 按 K 升序排列，方便绘图
    results.sort(key=lambda r: r["K"])
    return results


def plot_bi_score(text_results, image_results, out_path: Path):
    """绘制 BI Score 双模态柱状图（文本蓝色 + 图像橙色）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in text_results]
    bi_text  = [r["bi_score"] or 0 for r in text_results]
    bi_image = [r["bi_score"] or 0 for r in (image_results if image_results else text_results)]

    x = np.arange(len(layers))
    w = 0.4

    fig, ax = plt.subplots(figsize=(16, 5))
    bars_t = ax.bar(x - w/2, bi_text,  w, label="Text",  color="#1f77b4", alpha=0.85)
    bars_i = ax.bar(x + w/2, bi_image, w, label="Image", color="#ff7f0e", alpha=0.85)

    # 标注 Top-6 高 BI 层（文本）
    top6_idx = sorted(range(len(bi_text)), key=lambda i: bi_text[i], reverse=True)[:6]
    for idx in top6_idx:
        ax.text(idx - w/2, bi_text[idx] + max(bi_text) * 0.01,
                str(idx), ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1f77b4")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("BI Score  (1 − cosine(input, output))", fontsize=12)
    ax.set_title("jina-v4 Block Influence (BI) Score per Layer\n"
                 "(ShortGPT metric: higher = more transformation = less redundant)",
                 fontsize=13)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(layers[::2])
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  BI Score 图表已保存：{out_path}")


def plot_causal_tracing(text_results, image_results, out_path: Path):
    """绘制因果追踪双模态 recovery 柱状图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_x = [r["layer"] for r in text_results]
    r_text  = [r["recovery"] for r in text_results]
    r_image = [r["recovery"] for r in image_results] if image_results else []

    x = np.arange(len(layers_x))
    w = 0.4 if r_image else 0.6

    fig, ax = plt.subplots(figsize=(16, 5))
    if r_image:
        ax.bar(x - w/2, r_text,  w, label="Text",  color="#1f77b4", alpha=0.85)
        ax.bar(x + w/2, r_image, w, label="Image", color="#ff7f0e", alpha=0.85)
        top6_i = sorted(range(len(r_image)), key=lambda i: r_image[i], reverse=True)[:6]
        for idx in top6_i:
            ax.text(idx + w/2, r_image[idx] + max(r_image) * 0.01, str(idx),
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color="#ff7f0e")
    else:
        ax.bar(x, r_text, w, label="Text", color="#1f77b4", alpha=0.85)

    top6_t = sorted(range(len(r_text)), key=lambda i: r_text[i], reverse=True)[:6]
    for idx in top6_t:
        ax.text(idx - w/2 if r_image else idx,
                r_text[idx] + max(r_text) * 0.01, str(idx),
                ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1f77b4")

    # 标记最后 6 层区域
    ax.axvspan(29.5, 35.5, color="lightgreen", alpha=0.15, label="Last 6 layers (30-35)")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Recovery  (cos_restored − cos_corrupt)", fontsize=12)
    ax.set_title("jina-v4 Causal Tracing — Recovery per Layer\n"
                 "(higher = restoring this layer most fixes corrupted embedding)", fontsize=13)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(layers_x[::2])
    ax.legend(fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  因果追踪图表已保存：{out_path}")


def plot_results(results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers       = [r["layer"]       for r in results]
    sensitivity  = [r["sensitivity"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(layers, sensitivity, color=[
        "#d62728" if s > 0.1 else "#ff7f0e" if s > 0.01 else "#1f77b4"
        for s in sensitivity
    ], edgecolor="white", linewidth=0.5)

    top6 = sorted(results, key=lambda r: r["sensitivity"], reverse=True)[:6]
    top6_layers = {r["layer"] for r in top6}
    for bar, l in zip(bars, layers):
        if l in top6_layers:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.5)

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Sensitivity  (1 − cosine similarity)", fontsize=12)
    ax.set_title("jina-v4 Single-Layer Residual Ablation\n"
                 "(each layer zeroed independently; measures per-layer increment)", fontsize=13)
    ax.set_xticks(range(0, len(layers), 2))
    ax.axhline(0, color="gray", linewidth=0.5)
    for r in top6:
        ax.text(r["layer"], r["sensitivity"] + max(sensitivity) * 0.01,
                str(r["layer"]), ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  图表已保存：{out_path}")


def plot_suffix_results(suffix_results, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cutoffs     = [r["cutoff_layer"]  for r in suffix_results]
    sensitivity = [r["sensitivity"]   for r in suffix_results]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cutoffs, sensitivity, "o-", color="#1f77b4", markersize=4)

    # 标注 K=6 截断点（层30）
    k6 = next((r for r in suffix_results if r["cutoff_layer"] == 30), None)
    if k6:
        ax.axvline(30, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"K=6 truncation (layer 30): Δcos={k6['sensitivity']:.3f}")
        ax.scatter([30], [k6["sensitivity"]], color="#d62728", zorder=5, s=60)

    ax.set_xlabel("Cutoff Layer (layers from cutoff to 35 are removed)", fontsize=12)
    ax.set_ylabel("Sensitivity  (1 − cosine similarity)", fontsize=12)
    ax.set_title("jina-v4 Suffix Truncation Ablation\n"
                 "(x=l means layers l..35 removed; measures collective value of last (36-l) layers)",
                 fontsize=13)
    ax.set_xticks(range(0, len(cutoffs), 2))
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  截断消融图表已保存：{out_path}")


def write_markdown(single_results, suffix_results, out_path: Path):
    top_n = sorted(single_results, key=lambda r: r["sensitivity"], reverse=True)

    lines = [
        "# jina-v4 层敏感性消融实验结果",
        "",
        "## 实验一：单层残差置零（per-layer increment）",
        "",
        "**方法**：对每层独立置零残差，测 cos 下降量。",
        "**解读**：衡量该层的「单步增量」，早期层增量天然更大（从零开始建表示）。",
        "",
        "| 排名 | 层 | Sensitivity (1−cos) | 说明 |",
        "|------|-----|---------------------|------|",
    ]
    for i, r in enumerate(top_n[:10]):
        s = r["sensitivity"]
        tag = "⚠️ 高" if s > 0.1 else "中" if s > 0.01 else "低"
        lines.append(f"| {i+1} | {r['layer']} | {s:.6f} | {tag} |")

    last6_single = [r for r in single_results if r["layer"] >= 30]
    other_single = [r for r in single_results if r["layer"] < 30]
    last6_mean = np.mean([r["sensitivity"] for r in last6_single])
    other_mean = np.mean([r["sensitivity"] for r in other_single])

    lines += [
        "",
        f"> 最后 6 层单层平均敏感性 {last6_mean:.4f}，前 30 层均值 {other_mean:.4f}。",
        "> 早期层单层增量更大，但这不等于「早期层对最终 embedding 更重要」。",
        "",
    ]

    if suffix_results:
        lines += [
            "## 实验二：后缀截断消融（collective value of last-K layers）",
            "",
            "**方法**：将层 l\~35 全部置零（等价于在第 l-1 层截断模型），测 cos 下降量。",
            "**解读**：衡量「最后 (36-l) 层的集体价值」，直接支撑 K 层选择决策。",
            "",
            "| 截断点 l | 移除层数 | Sensitivity | 含义 |",
            "|---------|---------|------------|------|",
        ]
        for r in suffix_results:
            l = r["cutoff_layer"]
            if l in [30, 31, 32, 33, 34, 35]:
                k = 36 - l
                lines.append(f"| {l} | {k} 层（层{l}-35）| {r['sensitivity']:.6f} | "
                              f"K={k} 方案的总损失 |")

        k6 = next((r for r in suffix_results if r["cutoff_layer"] == 30), None)
        k1 = next((r for r in suffix_results if r["cutoff_layer"] == 35), None)
        if k6 and k1:
            lines += [
                "",
                f"> **K=6 方案（层30-35）截断损失：{k6['sensitivity']:.4f}**",
                f"> K=1 方案（仅层35）截断损失：{k1['sensitivity']:.4f}",
                f"> 最后 6 层集体贡献是最后 1 层的 {k6['sensitivity']/max(k1['sensitivity'],1e-9):.1f}×",
            ]

    lines += [
        "",
        "## 对 zkLLM 层选择的指导意义",
        "",
        "1. **单层实验**揭示了残差流累积规律：早期层单步增量大，但这是正常的深度网络特性，",
        "   不代表应该证明早期层。",
        "2. **截断实验**是更相关的指标：显示最后 K 层集体对 embedding 质量有不可忽视的贡献。",
        "3. **连续最后 K 层**仍是最优策略，理由：",
        "   - 单一信任边界（vs 分散层的多信任边界）",
        "   - 直接证明产生最终 embedding 的计算",
        "   - 截断实验定量支撑 K=6 的贡献大小",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown 摘要已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser(description="jina-v4 层敏感性消融实验")
    parser.add_argument("--samples", type=int, default=20,
                        help="文本样本数（默认 20）")
    parser.add_argument("--img-samples", type=int, default=20,
                        help="图像样本数（默认 20，需要 corpora/image.jsonl）")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", default="bi",
                        choices=["bi", "zero", "noise", "causal", "all"],
                        help="bi=实验A  zero=实验B  noise=实验C旧版  causal=实验C完整版  all=全部")
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="噪声强度倍率（实验C旧版，默认1.0）")
    parser.add_argument("--corrupt-scale", type=float, default=3.0,
                        help="因果追踪腐化强度（默认3.0，建议≥2使腐化充分）")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    run_bi     = args.mode in ("bi",     "all")
    run_zero   = args.mode in ("zero",   "all")
    run_noise  = args.mode in ("noise",  "all")
    run_causal = args.mode in ("causal", "all")

    model  = load_model(args.device)
    texts  = (SAMPLE_TEXTS * ((args.samples // len(SAMPLE_TEXTS)) + 1))[:args.samples]
    images = load_image_samples(args.img_samples) if (run_bi or run_zero or run_noise or run_causal) else []

    step = 1

    # ── 实验 A：BI Score ──────────────────────────────────────────────────────
    if run_bi:
        print(f"\n[{step}] 实验 A：BI Score（文本 {len(texts)} 个 + 图像 {len(images)} 张）...")
        step += 1

        print("  → 文本 BI Score")
        bi_text = run_bi_score(model, texts=texts, images=None, device=args.device)

        bi_image = []
        if images:
            print("  → 图像 BI Score")
            bi_image = run_bi_score(model, texts=None, images=images, device=args.device)

        # 保存 JSON
        bi_json = OUT_DIR / "ablation_bi_score.json"
        with open(bi_json, "w") as f:
            json.dump({
                "model": "jina-embeddings-v4",
                "method": "BI Score (ShortGPT, arXiv:2403.03853)",
                "n_text_samples":  len(texts),
                "n_image_samples": len(images),
                "text_bi":  bi_text,
                "image_bi": bi_image,
            }, f, indent=2)
        print(f"  JSON 已保存：{bi_json}")

        plot_bi_score(bi_text, bi_image if bi_image else None,
                      OUT_DIR / "ablation_bi_score.png")

        # 打印摘要
        top6_bi = sorted(bi_text, key=lambda r: r["bi_score"] or 0, reverse=True)[:6]
        print("\n== 实验 A Top-6 高 BI 层（文本，变换量最大）==")
        for r in top6_bi:
            print(f"  Layer {r['layer']:2d}  BI={r['bi_score']:.4f}")
        if bi_image:
            top6_img = sorted(bi_image, key=lambda r: r["bi_score"] or 0, reverse=True)[:6]
            print("\n== 实验 A Top-6 高 BI 层（图像）==")
            for r in top6_img:
                print(f"  Layer {r['layer']:2d}  BI={r['bi_score']:.4f}")

    # ── 实验 B：单层残差置零 ──────────────────────────────────────────────────
    if run_zero:
        print(f"\n[{step}] 实验 B：单层残差置零（文本 {len(texts)} 个 + 图像 {len(images)} 张）...")
        step += 1

        print("  → 文本消融")
        text_zero_results = run_ablation_texts(model, texts, args.device)

        image_zero_results = []
        if images:
            print("  → 图像消融")
            image_zero_results = run_ablation_images(model, images, args.device)

        json_path = OUT_DIR / "ablation_layer_sensitivity.json"
        with open(json_path, "w") as f:
            json.dump({
                "model": "jina-embeddings-v4",
                "method": "single-layer residual zeroing",
                "n_text_samples": len(texts),
                "n_image_samples": len(images),
                "text_results": text_zero_results,
                "image_results": image_zero_results,
            }, f, indent=2)
        print(f"  JSON 已保存：{json_path}")

        # 文本图表
        plot_results(text_zero_results, OUT_DIR / "ablation_layer_sensitivity_text.png")

        # 图像图表
        if image_zero_results:
            plot_results(image_zero_results, OUT_DIR / "ablation_layer_sensitivity_image.png")

        # 双模态对比图（复用 plot_bi_score 布局但显示 sensitivity）
        if image_zero_results:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            layers_x = [r["layer"] for r in text_zero_results]
            s_text  = [r["sensitivity"] for r in text_zero_results]
            s_image = [r["sensitivity"] for r in image_zero_results]
            x = np.arange(len(layers_x))
            w = 0.4
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.bar(x - w/2, s_text,  w, label="Text",  color="#1f77b4", alpha=0.85)
            ax.bar(x + w/2, s_image, w, label="Image", color="#ff7f0e", alpha=0.85)
            top6_t = sorted(range(len(s_text)),  key=lambda i: s_text[i],  reverse=True)[:6]
            top6_i = sorted(range(len(s_image)), key=lambda i: s_image[i], reverse=True)[:6]
            for idx in top6_t:
                ax.text(idx - w/2, s_text[idx]  + max(s_text)  * 0.01, str(idx),
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1f77b4")
            for idx in top6_i:
                ax.text(idx + w/2, s_image[idx] + max(s_image) * 0.01, str(idx),
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color="#ff7f0e")
            ax.set_xlabel("Layer Index", fontsize=12)
            ax.set_ylabel("Sensitivity  (1 − cosine similarity)", fontsize=12)
            ax.set_title("jina-v4 Single-Layer Residual Ablation — Text vs Image\n"
                         "(each layer zeroed independently)", fontsize=13)
            ax.set_xticks(x[::2])
            ax.set_xticklabels(layers_x[::2])
            ax.legend(fontsize=11)
            ax.axhline(0, color="gray", linewidth=0.5)
            plt.tight_layout()
            dual_path = OUT_DIR / "ablation_layer_sensitivity_dual.png"
            plt.savefig(dual_path, dpi=150)
            plt.close()
            print(f"  双模态对比图已保存：{dual_path}")

        top6 = sorted(text_zero_results, key=lambda r: r["sensitivity"], reverse=True)[:6]
        print("\n== 实验 B Top-6 最敏感层（文本）==")
        for r in top6:
            print(f"  Layer {r['layer']:2d}  sensitivity={r['sensitivity']:.4f}")

        if image_zero_results:
            top6_img = sorted(image_zero_results, key=lambda r: r["sensitivity"], reverse=True)[:6]
            print("\n== 实验 B Top-6 最敏感层（图像）==")
            for r in top6_img:
                print(f"  Layer {r['layer']:2d}  sensitivity={r['sensitivity']:.4f}")


    # ── 实验 C：噪声注入 ──────────────────────────────────────────────────────
    if run_noise:
        scale = args.noise_scale
        print(f"\n[{step}] 实验 C：噪声注入（scale={scale}，文本 {len(texts)} 个 + 图像 {len(images)} 张）...")
        step += 1

        print("  → 文本噪声注入")
        noise_text = run_noise_injection(model, texts, encode_texts,
                                         noise_scale=scale, label="（文本）")

        noise_image = []
        if images:
            print("  → 图像噪声注入")
            noise_image = run_noise_injection(model, images, encode_images,
                                              noise_scale=scale, label="（图像）")

        json_path = OUT_DIR / "ablation_noise_injection.json"
        with open(json_path, "w") as f:
            json.dump({
                "model": "jina-embeddings-v4",
                "method": "adaptive Gaussian noise injection",
                "noise_scale": scale,
                "n_text_samples": len(texts),
                "n_image_samples": len(images),
                "text_results": noise_text,
                "image_results": noise_image,
            }, f, indent=2)
        print(f"  JSON 已保存：{json_path}")

        # 双模态对比图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        layers_x = [r["layer"] for r in noise_text]
        s_text  = [r["sensitivity"] for r in noise_text]
        s_image = [r["sensitivity"] for r in noise_image] if noise_image else []
        x = np.arange(len(layers_x))
        w = 0.4 if s_image else 0.6
        fig, ax = plt.subplots(figsize=(16, 5))
        if s_image:
            ax.bar(x - w/2, s_text,  w, label="Text",  color="#1f77b4", alpha=0.85)
            ax.bar(x + w/2, s_image, w, label="Image", color="#ff7f0e", alpha=0.85)
            top6_i = sorted(range(len(s_image)), key=lambda i: s_image[i], reverse=True)[:6]
            for idx in top6_i:
                ax.text(idx + w/2, s_image[idx] + max(s_image) * 0.01, str(idx),
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color="#ff7f0e")
        else:
            ax.bar(x, s_text, w, label="Text", color="#1f77b4", alpha=0.85)
        top6_t = sorted(range(len(s_text)), key=lambda i: s_text[i], reverse=True)[:6]
        for idx in top6_t:
            ax.text(idx - w/2 if s_image else idx,
                    s_text[idx] + max(s_text) * 0.01, str(idx),
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1f77b4")
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Sensitivity  (1 − cosine similarity)", fontsize=12)
        ax.set_title(f"jina-v4 Noise Injection Ablation — Text vs Image  (scale={scale})\n"
                     "(Gaussian noise σ=scale×std(layer output) injected per layer)", fontsize=13)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(layers_x[::2])
        ax.legend(fontsize=11)
        ax.axhline(0, color="gray", linewidth=0.5)
        plt.tight_layout()
        noise_path = OUT_DIR / "ablation_noise_injection.png"
        plt.savefig(noise_path, dpi=150)
        plt.close()
        print(f"  图表已保存：{noise_path}")

        top6 = sorted(noise_text, key=lambda r: r["sensitivity"], reverse=True)[:6]
        print(f"\n== 实验 C Top-6 最敏感层（文本，scale={scale}）==")
        for r in top6:
            print(f"  Layer {r['layer']:2d}  sensitivity={r['sensitivity']:.4f}  σ={r['noise_std']:.4f}")
        if noise_image:
            top6_img = sorted(noise_image, key=lambda r: r["sensitivity"], reverse=True)[:6]
            print(f"\n== 实验 C Top-6 最敏感层（图像，scale={scale}）==")
            for r in top6_img:
                print(f"  Layer {r['layer']:2d}  sensitivity={r['sensitivity']:.4f}  σ={r['noise_std']:.4f}")


    # ── 实验 C 完整版：因果追踪 ──────────────────────────────────────────────
    if run_causal:
        scale = args.corrupt_scale
        print(f"\n[{step}] 实验 C 完整版：因果追踪（corrupt_scale={scale}，"
              f"文本 {len(texts)} 个 + 图像 {len(images)} 张）...")
        step += 1

        print("  → 文本因果追踪")
        causal_text = run_causal_tracing(model, texts, encode_texts,
                                          corrupt_scale=scale, label="（文本）")

        causal_image = []
        if images:
            print("  → 图像因果追踪")
            causal_image = run_causal_tracing(model, images, encode_images,
                                               corrupt_scale=scale, label="（图像）")

        json_path = OUT_DIR / "ablation_causal_tracing.json"
        with open(json_path, "w") as f:
            json.dump({
                "model": "jina-embeddings-v4",
                "method": "causal tracing (Meng et al. ROME NeurIPS 2022)",
                "corrupt_scale": scale,
                "n_text_samples": len(texts),
                "n_image_samples": len(images),
                "text_results":  causal_text,
                "image_results": causal_image,
            }, f, indent=2)
        print(f"  JSON 已保存：{json_path}")

        plot_causal_tracing(causal_text, causal_image,
                            OUT_DIR / "ablation_causal_tracing.png")

        top6_t = sorted(causal_text, key=lambda r: r["recovery"], reverse=True)[:6]
        print(f"\n== 实验 C(完整) Top-6 关键层（文本，recovery 最高）==")
        for r in top6_t:
            print(f"  Layer {r['layer']:2d}  recovery={r['recovery']:.4f}"
                  f"  (corrupt={r['cos_corrupt']:.4f} → restore={r['cos_restored']:.4f})")

        if causal_image:
            top6_i = sorted(causal_image, key=lambda r: r["recovery"], reverse=True)[:6]
            print(f"\n== 实验 C(完整) Top-6 关键层（图像，recovery 最高）==")
            for r in top6_i:
                print(f"  Layer {r['layer']:2d}  recovery={r['recovery']:.4f}"
                      f"  (corrupt={r['cos_corrupt']:.4f} → restore={r['cos_restored']:.4f})")

        # 打印最后 6 层的 recovery 统计（直接论据）
        last6_t = [r for r in causal_text  if r["layer"] >= 30]
        last6_i = [r for r in causal_image if r["layer"] >= 30] if causal_image else []
        print(f"\n  最后 6 层（30-35）平均 recovery — 文本: "
              f"{np.mean([r['recovery'] for r in last6_t]):.4f}"
              + (f"  图像: {np.mean([r['recovery'] for r in last6_i]):.4f}" if last6_i else ""))


if __name__ == "__main__":
    main()

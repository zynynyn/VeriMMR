"""
Step 2：加载 jina-embeddings-v4 真实权重并验证量化精度

功能：
1. 从 safetensors 加载 base 权重 + LoRA (retrieval task)
2. 合并有效权重：W_eff = W_base + lora_B @ lora_A
3. 量化：W_int = round(W_eff.T * scale).int32  (转置为 in×out 格式)
4. 精度验证：比较 bfloat16 参考输出 vs 反量化输出
5. 以 zkLLM 格式保存指定层权重 + 生成 ppgen + 提交承诺

jina-v4 LM 关键维度（Qwen2.5-3B backbone）：
  hidden_size=2048, intermediate_size=11008
  num_heads=16, num_kv_heads=2 → kv_dim=256
  num_layers=36
"""

import os, sys, argparse, time
import numpy as np
import torch

# ---- 参数 ----
SCALE = 1 << 16
LORA_ALPHA = 32
LORA_R     = 32
LORA_SCALE = LORA_ALPHA / LORA_R   # = 1.0

MODEL_DIR   = "/root/autodl-tmp/models/jina-embeddings-v4"
ADAPTER_DIR = f"{MODEL_DIR}/adapters"
WORKDIR     = "./zkllm-workdir/jina-v4"

# safetensors index: 哪个 shard 包含哪些 key
def build_shard_map():
    import json
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
    return idx["weight_map"]   # key → filename

def load_tensor(key, shard_map):
    from safetensors import safe_open
    fname = shard_map[key]
    with safe_open(f"{MODEL_DIR}/{fname}", framework="pt") as f:
        return f.get_tensor(key).float()   # bf16 → float32

def load_lora(task="retrieval"):
    """加载全部 LoRA 权重到 dict"""
    from safetensors import safe_open
    lora = {}
    with safe_open(f"{ADAPTER_DIR}/adapter_model.safetensors", framework="pt") as f:
        for k in f.keys():
            if f".{task}." in k:
                # 简化 key：去掉 "base_model.model.model.language_model." 前缀
                short = k.split("language_model.")[-1]
                lora[short] = f.get_tensor(k).float()
    return lora

def get_effective_weight(layer_idx, param_name, shard_map, lora):
    """
    返回有效权重（float32，PyTorch 原始 out×in 排列）
    param_name 例如 "self_attn.q_proj.weight"
    """
    key = f"model.layers.{layer_idx}.{param_name}"
    W = load_tensor(key, shard_map)                    # (out, in)

    # LoRA key 格式：layers.{i}.{module}.lora_A.{task}.weight
    module = ".".join(param_name.split(".")[:2])       # e.g. "self_attn.q_proj"
    a_key = f"layers.{layer_idx}.{module}.lora_A.weight"
    b_key = f"layers.{layer_idx}.{module}.lora_B.weight"

    if a_key in lora and b_key in lora:
        A = lora[a_key]   # (r, in)
        B = lora[b_key]   # (out, r)
        W = W + LORA_SCALE * (B @ A)

    return W   # (out, in) float32

def quantize_weight(W_out_in):
    """(out, in) → 转置为 (in, out) → 量化为 int32"""
    W_T = W_out_in.T.contiguous()      # (in, out)
    W_int = torch.round(W_T * SCALE).to(torch.int32)
    return W_int                        # (in, out)

def dequant_weight(W_int_in_out):
    """(in, out) int32 → (out, in) float32（还原为 PyTorch 乘法用的格式）"""
    return (W_int_in_out.float() / SCALE).T.contiguous()

# ---- 精度验证 ----
def verify_layer_precision(layer_idx, shard_map, lora, seq_len=64, verbose=True):
    print(f"\n{'='*60}")
    print(f"精度验证 — Layer {layer_idx}  (seq_len={seq_len})")
    print(f"{'='*60}")

    results = {}

    # 构造随机输入
    torch.manual_seed(42)
    x = torch.randn(seq_len, 2048, dtype=torch.float32)

    params = [
        ("self_attn.q_proj.weight", 2048, 2048),
        ("self_attn.k_proj.weight", 256,  2048),
        ("self_attn.v_proj.weight", 256,  2048),
        ("self_attn.o_proj.weight", 2048, 2048),
        ("mlp.up_proj.weight",      11008, 2048),
        ("mlp.gate_proj.weight",    11008, 2048),
        ("mlp.down_proj.weight",    2048,  11008),
    ]

    for param_name, out_dim, in_dim in params:
        W_eff = get_effective_weight(layer_idx, param_name, shard_map, lora)
        # 参考输出：float32 直接 matmul
        inp = x if in_dim == 2048 else torch.randn(seq_len, in_dim, dtype=torch.float32)
        ref_out = inp @ W_eff.T                       # (seq, out)

        # 量化路径
        W_int = quantize_weight(W_eff)                # (in, out) int32
        W_dq  = dequant_weight(W_int)                 # (out, in) float32
        quant_out = inp @ W_dq.T                      # (seq, out)

        # 误差分析
        diff = (ref_out - quant_out).abs()
        max_err  = diff.max().item()
        mean_err = diff.mean().item()
        # 余弦相似度（逐 token 均值）
        cos = torch.nn.functional.cosine_similarity(
            ref_out.reshape(-1, out_dim),
            quant_out.reshape(-1, out_dim), dim=-1
        ).mean().item()

        results[param_name] = dict(max_err=max_err, mean_err=mean_err, cos=cos,
                                   W_int=W_int, shape=(in_dim, out_dim))
        if verbose:
            short = param_name.replace("self_attn.","attn.").replace(".weight","")
            print(f"  {short:25s}  max_err={max_err:.4f}  mean_err={mean_err:.5f}  cos={cos:.6f}")

    return results

# ---- zkLLM 格式保存 ----
def save_layer_for_zkllm(layer_idx, precision_results, task="retrieval"):
    os.makedirs(WORKDIR, exist_ok=True)
    prefix = f"layer-{layer_idx}"

    # 维度映射（in_dim, out_dim） → ppgen 用 in_dim
    pp_info = {
        "self_attn.q_proj.weight": (2048, 2048),
        "self_attn.k_proj.weight": (2048, 256),   # GQA：in=2048, out=256
        "self_attn.v_proj.weight": (2048, 256),
        "self_attn.o_proj.weight": (2048, 2048),
        "mlp.up_proj.weight":      (2048, 11008),
        "mlp.gate_proj.weight":    (2048, 11008),
        "mlp.down_proj.weight":    (11008, 2048),
    }

    # zkLLM 文件名映射
    zkllm_name = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "mlp.up_proj.weight":      "mlp.up_proj.weight",
        "mlp.gate_proj.weight":    "mlp.gate_proj.weight",
        "mlp.down_proj.weight":    "mlp.down_proj.weight",
    }

    print(f"\n{'='*60}")
    print(f"保存 Layer {layer_idx} 权重到 {WORKDIR}/")
    print(f"{'='*60}")

    for param_name, (in_dim, out_dim) in pp_info.items():
        W_int = precision_results[param_name]["W_int"]   # (in, out) int32

        zname   = zkllm_name[param_name]
        pp_path   = f"{WORKDIR}/{zname}-pp.bin"
        int_path  = f"{WORKDIR}/{prefix}-{zname}-int.bin"
        cm_path   = f"{WORKDIR}/{prefix}-{zname}-commitment.bin"

        # 1. 保存 int32 权重二进制
        W_int.cpu().numpy().astype(np.int32).tofile(int_path)
        print(f"  saved {os.path.basename(int_path)} shape=({in_dim},{out_dim})")

        # 2. ppgen（只在首次生成）
        if not os.path.exists(pp_path):
            t0 = time.time()
            # ppgen 的 size = in_dim（行数）
            pp_size = in_dim
            v = pp_size
            log2_pp = 0
            while v > 1:
                v = (v + 1) >> 1
                log2_pp += 1
            pp_size_padded = (1 << log2_pp) << 5   # log_off_factor=5
            os.system(f"./ppgen {pp_size_padded} {pp_path}")
            print(f"    ppgen done ({time.time()-t0:.1f}s)")

        # 3. 提交承诺
        if not os.path.exists(cm_path):
            t0 = time.time()
            os.system(f"./commit-param {pp_path} {int_path} {cm_path} {in_dim} {out_dim}")
            print(f"    commit done ({time.time()-t0:.1f}s)")

    print(f"\nLayer {layer_idx} 权重文件全部就绪。")
    return WORKDIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=35,
                        help="验证哪一层（默认 35 = 最后一层）")
    parser.add_argument("--task", default="retrieval",
                        choices=["retrieval", "code", "text-matching"])
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--save", action="store_true",
                        help="是否保存到 zkLLM 二进制格式")
    args = parser.parse_args()

    print(f"加载模型权重 (task={args.task})...")
    t0 = time.time()
    shard_map = build_shard_map()
    lora      = load_lora(args.task)
    print(f"  LoRA keys loaded: {len(lora)}  ({time.time()-t0:.1f}s)")

    results = verify_layer_precision(args.layer, shard_map, lora,
                                     seq_len=args.seq_len)

    # 汇总
    cos_vals = [v["cos"] for v in results.values()]
    print(f"\n平均余弦相似度: {sum(cos_vals)/len(cos_vals):.6f}")
    print(f"最大余弦偏差:   {1-min(cos_vals):.2e}")

    if args.save:
        save_layer_for_zkllm(args.layer, results, args.task)

"""
Step 4：K 层性能曲线
对 jina-v4 最后 K 层（layer 35-K+1 … 35）跑 FFN + Attn-linear proof
测量各 K 值的 prove 时间，绘制 time-security trade-off 曲线数据

用法：
  python bench_k_layers.py --k_values 1 6 18 36
"""
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from load_jina_weights import (
    build_shard_map, load_lora, get_effective_weight,
    quantize_weight, save_layer_for_zkllm, WORKDIR, SCALE
)

SEQ_LEN   = 512   # seq_len × kv_dim=256 → 131072 = 2×65536 ✓
EMBED_DIM = 2048
KV_DIM    = 256
HIDDEN_DIM = 11008

def ensure_layer_weights(layer_idx, shard_map, lora):
    """确保该层权重已保存为 zkLLM 格式（有缓存则跳过）"""
    prefix   = f"layer-{layer_idx}"
    q_int    = f"{WORKDIR}/{prefix}-self_attn.q_proj.weight-int.bin"
    if os.path.exists(q_int):
        return  # 已有缓存

    print(f"  [Layer {layer_idx}] 量化并保存权重...")
    # 复用 load_jina_weights 的精度验证结果
    from load_jina_weights import verify_layer_precision
    results = verify_layer_precision(layer_idx, shard_map, lora,
                                     seq_len=4, verbose=False)
    save_layer_for_zkllm(layer_idx, results)

def make_input(path, seq_len=SEQ_LEN, dim=EMBED_DIM):
    if os.path.exists(path):
        return
    inp = (np.random.randn(seq_len, dim) * SCALE).astype(np.int32)
    inp.tofile(path)

def bench_one_layer(layer_idx):
    """返回 (ffn_sec, attn_sec)"""
    prefix   = f"layer-{layer_idx}"
    in_file  = f"{WORKDIR}/{prefix}-bench-input.bin"
    make_input(in_file)

    # FFN
    out_ffn = f"{WORKDIR}/{prefix}-bench-ffn-out.bin"
    t0 = time.time()
    ret = os.system(
        f"./ffn {in_file} {SEQ_LEN} {EMBED_DIM} {HIDDEN_DIM} "
        f"{WORKDIR} {prefix} {out_ffn} 2>/dev/null"
    )
    ffn_sec = time.time() - t0
    if ret != 0:
        print(f"  !! FFN proof failed for layer {layer_idx}")
        ffn_sec = float('nan')

    # Attn-linear（GQA，kv_dim=256）
    out_attn = f"{WORKDIR}/{prefix}-bench-attn-out.bin"
    t0 = time.time()
    ret = os.system(
        f"./self-attn linear {in_file} {SEQ_LEN} {EMBED_DIM} "
        f"{WORKDIR} {prefix} {out_attn} {KV_DIM} 2>/dev/null"
    )
    attn_sec = time.time() - t0
    if ret != 0:
        print(f"  !! Attn proof failed for layer {layer_idx}")
        attn_sec = float('nan')

    return ffn_sec, attn_sec

def run_k_benchmark(k_values, shard_map, lora):
    print(f"\n{'='*65}")
    print(f"K 层性能曲线  —  jina-v4 真实权重 (seq_len={SEQ_LEN})")
    print(f"{'='*65}")
    print(f"{'K':>4}  {'层范围':^12}  {'FFN(s)':>8}  {'Attn(s)':>8}  {'总计(s)':>8}")
    print(f"{'-'*4}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")

    results = []
    for K in sorted(k_values):
        layers = list(range(36 - K, 36))   # 最后 K 层：35-K+1 … 35
        layer_range = f"{36-K}–35"

        # 准备权重（仅第一次需要量化，后续直接读缓存）
        print(f"\n  [K={K}] 准备 {K} 层权重...")
        for li in layers:
            ensure_layer_weights(li, shard_map, lora)

        # 逐层计时
        total_ffn = total_attn = 0.0
        for li in layers:
            ffn_s, attn_s = bench_one_layer(li)
            total_ffn  += ffn_s
            total_attn += attn_s

        total = total_ffn + total_attn
        results.append(dict(K=K, ffn=total_ffn, attn=total_attn, total=total))
        print(f"  {K:>4}  {layer_range:^12}  {total_ffn:>8.1f}  {total_attn:>8.1f}  {total:>8.1f}")

    # 汇总表
    print(f"\n{'='*65}")
    print(f"汇总（含估算 LoRA ×1.03，zkAttn+Norm ×1.5 系数）：")
    print(f"{'='*65}")
    print(f"{'K':>4}  {'纯 Sumcheck(s)':>14}  {'估算全量(s)':>12}  {'估算全量(min)':>13}  {'覆盖率':>8}")
    for r in results:
        full_est = r['total'] * 1.03 * 1.5
        print(f"  {r['K']:>2}  {r['total']:>14.1f}  {full_est:>12.1f}  {full_est/60:>13.1f}  {r['K']/36*100:>7.0f}%")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 6, 18, 36])
    parser.add_argument("--task", default="retrieval")
    args = parser.parse_args()

    np.random.seed(42)
    print("加载 shard map 和 LoRA 权重...")
    shard_map = build_shard_map()
    lora      = load_lora(args.task)
    print(f"  LoRA keys: {len(lora)}")

    run_k_benchmark(args.k_values, shard_map, lora)

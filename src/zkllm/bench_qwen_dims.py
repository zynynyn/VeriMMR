"""
独立 zkLLM timing benchmark
使用 Qwen2.5-3B (jina-v4 LM backbone) 维度的随机合成权重
无需 LLaMA-2，不依赖任何 gated model

Qwen2.5-3B 关键尺寸:
  embed_dim  = 2048  (hidden_size)
  hidden_dim = 11008 (intermediate_size, SwiGLU)
  num_heads  = 16
  head_dim   = 128
  kv_heads   = 2  => k/v dim = 256 (GQA, 但 self-attn.cu 是 MHA，暂用 embed_dim)
  num_layers = 36
"""

import os, sys, time, argparse
import numpy as np

SCALE = 1 << 16   # 与 llama-commit.py 保持一致

# ---------- 尺寸参数 ----------
EMBED_DIM   = 2048
HIDDEN_DIM  = 11008   # SwiGLU gate/up/down
SEQ_LEN_VIT = 512     # jina-v4 处理 448×448 图像后约 392 个视觉 token，取 512 便于测试

LOG_OFF = 5  # ppgen 需要的 log_offset_factor（与 llama-ppgen.py 一致）

WORKDIR = "./zkllm-workdir/qwen-3b"
os.makedirs(WORKDIR, exist_ok=True)

def elapsed(label, t0):
    dt = time.time() - t0
    print(f"  [{label}] {dt*1000:.1f} ms")
    return dt

def ppgen(size, out_path):
    """生成 public parameters 文件"""
    if os.path.exists(out_path):
        return
    pp_size = 1
    v = size
    while v > 1:
        v = (v + 1) >> 1
        pp_size <<= 1
    pp_size <<= LOG_OFF
    os.system(f"./ppgen {pp_size} {out_path}")

def save_rand_int(shape, path):
    """保存随机 int32 权重文件"""
    if os.path.exists(path):
        return
    w = (np.random.randn(*shape) * SCALE).astype(np.int32)
    w.tofile(path)

def commit_weight(pp_path, int_path, commit_path, rows, cols):
    if os.path.exists(commit_path):
        return
    os.system(f"./commit-param {pp_path} {int_path} {commit_path} {rows} {cols}")

def prepare_ffn_weights(layer_idx):
    """为 FFN 层生成并提交权重"""
    prefix = f"layer-{layer_idx}"
    # pp (每个 pp size 只需一份，跨 layer 共享)
    up_pp    = f"{WORKDIR}/mlp.up_proj.weight-pp.bin"
    gate_pp  = f"{WORKDIR}/mlp.gate_proj.weight-pp.bin"
    down_pp  = f"{WORKDIR}/mlp.down_proj.weight-pp.bin"

    print(f"  Generating ppgen for FFN weights (if needed)...")
    t0 = time.time()
    ppgen(EMBED_DIM,  up_pp)
    ppgen(EMBED_DIM,  gate_pp)
    ppgen(HIDDEN_DIM, down_pp)
    elapsed("ppgen FFN", t0)

    # int 权重
    up_int   = f"{WORKDIR}/{prefix}-mlp.up_proj.weight-int.bin"
    gate_int = f"{WORKDIR}/{prefix}-mlp.gate_proj.weight-int.bin"
    down_int = f"{WORKDIR}/{prefix}-mlp.down_proj.weight-int.bin"
    save_rand_int((HIDDEN_DIM, EMBED_DIM), up_int)    # stored as (out, in)
    save_rand_int((HIDDEN_DIM, EMBED_DIM), gate_int)
    save_rand_int((EMBED_DIM,  HIDDEN_DIM), down_int)

    # commitments
    up_cm   = f"{WORKDIR}/{prefix}-mlp.up_proj.weight-commitment.bin"
    gate_cm = f"{WORKDIR}/{prefix}-mlp.gate_proj.weight-commitment.bin"
    down_cm = f"{WORKDIR}/{prefix}-mlp.down_proj.weight-commitment.bin"

    print(f"  Committing FFN weights...")
    t0 = time.time()
    commit_weight(up_pp,   up_int,   up_cm,   EMBED_DIM, HIDDEN_DIM)
    commit_weight(gate_pp, gate_int, gate_cm, EMBED_DIM, HIDDEN_DIM)
    commit_weight(down_pp, down_int, down_cm, HIDDEN_DIM, EMBED_DIM)
    elapsed("commit FFN", t0)

def prepare_attn_weights(layer_idx):
    """为 self-attn 层生成并提交权重（使用 MHA，embed_dim×embed_dim）"""
    prefix = f"layer-{layer_idx}"

    q_pp = f"{WORKDIR}/self_attn.q_proj.weight-pp.bin"
    k_pp = f"{WORKDIR}/self_attn.k_proj.weight-pp.bin"
    v_pp = f"{WORKDIR}/self_attn.v_proj.weight-pp.bin"
    o_pp = f"{WORKDIR}/self_attn.o_proj.weight-pp.bin"

    print(f"  Generating ppgen for Attn weights (if needed)...")
    t0 = time.time()
    ppgen(EMBED_DIM, q_pp)
    ppgen(EMBED_DIM, k_pp)
    ppgen(EMBED_DIM, v_pp)
    ppgen(EMBED_DIM, o_pp)
    elapsed("ppgen Attn", t0)

    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        int_path = f"{WORKDIR}/{prefix}-self_attn.{name}.weight-int.bin"
        cm_path  = f"{WORKDIR}/{prefix}-self_attn.{name}.weight-commitment.bin"
        pp_path  = f"{WORKDIR}/self_attn.{name}.weight-pp.bin"
        save_rand_int((EMBED_DIM, EMBED_DIM), int_path)

    print(f"  Committing Attn weights...")
    t0 = time.time()
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        int_path = f"{WORKDIR}/{prefix}-self_attn.{name}.weight-int.bin"
        cm_path  = f"{WORKDIR}/{prefix}-self_attn.{name}.weight-commitment.bin"
        pp_path  = f"{WORKDIR}/self_attn.{name}.weight-pp.bin"
        commit_weight(pp_path, int_path, cm_path, EMBED_DIM, EMBED_DIM)
    elapsed("commit Attn", t0)

def prepare_swiglu_table():
    """生成 SwiGLU 查找表（与 llama-ffn.py 中相同）"""
    if os.path.exists("swiglu-table.bin"):
        return
    try:
        import torch
        in_range = 10; in_prec = 12; out_prec = 16
        Xs = torch.arange(-(1 << (in_range-1)), 1 << (in_range-1),
                          step=1/(1 << in_prec), device='cpu')
        Ys = Xs * torch.sigmoid(Xs)
        t = torch.round(Ys * (1 << out_prec)).to(torch.int32)
        t.numpy().tofile("swiglu-table.bin")
        print("  swiglu-table.bin generated")
    except ImportError:
        # numpy fallback
        step = 1.0 / (1 << 12)
        Xs = np.arange(-(1 << 9), 1 << 9, step, dtype=np.float64)
        Ys = Xs / (1 + np.exp(-Xs))
        t = np.round(Ys * (1 << 16)).astype(np.int32)
        t.tofile("swiglu-table.bin")
        print("  swiglu-table.bin generated (numpy fallback)")

def make_input_file(path, seq_len, embed_dim):
    if os.path.exists(path):
        return
    inp = (np.random.randn(seq_len, embed_dim) * SCALE).astype(np.int32)
    inp.tofile(path)

def bench_ffn(layer_idx, seq_len):
    prepare_ffn_weights(layer_idx)
    prepare_swiglu_table()

    prefix  = f"layer-{layer_idx}"
    in_file = f"{WORKDIR}/{prefix}-ffn-input.bin"
    out_file = f"{WORKDIR}/{prefix}-ffn-output.bin"
    make_input_file(in_file, seq_len, EMBED_DIM)

    print(f"\n=== FFN proof (layer={layer_idx}, seq_len={seq_len}, embed={EMBED_DIM}, hidden={HIDDEN_DIM}) ===")
    t0 = time.time()
    ret = os.system(
        f"./ffn {in_file} {seq_len} {EMBED_DIM} {HIDDEN_DIM} {WORKDIR} {prefix} {out_file}"
    )
    dt = elapsed("FFN prove+verify", t0)
    print(f"  exit={ret}, time={dt:.2f}s")
    return dt

def bench_attn_linear(layer_idx, seq_len):
    prepare_attn_weights(layer_idx)

    prefix  = f"layer-{layer_idx}"
    in_file = f"{WORKDIR}/{prefix}-attn-input.bin"
    out_file = f"{WORKDIR}/{prefix}-attn-output.bin"
    make_input_file(in_file, seq_len, EMBED_DIM)

    print(f"\n=== Self-Attn Linear proof (layer={layer_idx}, seq_len={seq_len}, embed={EMBED_DIM}) ===")
    t0 = time.time()
    ret = os.system(
        f"./self-attn linear {in_file} {seq_len} {EMBED_DIM} {WORKDIR} {prefix} {out_file}"
    )
    dt = elapsed("Attn-linear prove+verify", t0)
    print(f"  exit={ret}, time={dt:.2f}s")
    return dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN_VIT,
                        help="Sequence length (default: 512, approx jina-v4 visual tokens)")
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of layers to benchmark")
    parser.add_argument("--skip_attn", action="store_true")
    parser.add_argument("--skip_ffn",  action="store_true")
    args = parser.parse_args()

    print(f"zkLLM benchmark — Qwen2.5-3B dims (embed={EMBED_DIM}, hidden={HIDDEN_DIM})")
    print(f"seq_len={args.seq_len}, layers={args.layers}")
    print(f"GPU: RTX 4090 D (sm_89), CUDA 12.8")
    print("="*60)

    total = 0.0
    for li in range(args.layers):
        if not args.skip_ffn:
            total += bench_ffn(li, args.seq_len)
        if not args.skip_attn:
            total += bench_attn_linear(li, args.seq_len)

    print(f"\n{'='*60}")
    print(f"Total wall time for {args.layers} layer(s): {total:.2f}s")
    if args.layers > 0:
        per_layer = total / args.layers
        est_36 = per_layer * 36
        print(f"Per-layer estimate: {per_layer:.2f}s")
        print(f"Estimated 36-layer full LM proof: {est_36:.1f}s = {est_36/60:.1f} min")

"""
生成 ViT 32 blocks 权重文件及承诺参数

jina-v4 (Qwen2.5-VL) 的 ViT block 结构（Qwen2_5_VLVisionBlock）：
  norm1:        Qwen2RMSNorm(1280)   → input_layernorm
  norm2:        Qwen2RMSNorm(1280)   → post_attention_layernorm
  attn.qkv:     Linear(1280, 3840, bias=True)  — fused QKV，拆为 q/k/v
  attn.proj:    Linear(1280, 1280, bias=True)  — o_proj
  mlp.gate_proj: Linear(1280, 3420, bias=True)
  mlp.up_proj:   Linear(1280, 3420, bias=True)
  mlp.down_proj: Linear(3420, 1280, bias=True)
  mlp.act_fn:   SiLUActivation (= SwiGLU，可复用 swiglu-table.bin)

num_heads=16, head_dim=80, MHA (kv_dim=embed_dim=1280, num_kv_heads=16)

证明策略：
  - 权重 W (无 bias) 通过 IPA 承诺验证
  - bias b 是公开模型参数，单独保存为 {prefix}-*.bias.bin，公开可验证

pp 文件（全块共用，按 in_dim 分类）：
  vit_self_attn.q_proj.weight-pp.bin  (in=1280)
  vit_mlp.gate_proj.weight-pp.bin     (in=1280)
  vit_mlp.down_proj.weight-pp.bin     (in=3420)
  vit_input_layernorm.weight-pp.bin   (size=1280，rmsnorm)

用法：
  python script/setup_vit_params.py [--blocks 0 1 ...] [--workdir ...]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).parent.parent.resolve()
BIN_DIR    = ROOT / "src" / "zkllm"
MODEL_PATH = "/root/autodl-tmp/models/jina-embeddings-v4"

VIT_HIDDEN  = 1280
VIT_INTER   = 3420
SCALE       = 1 << 16


def _get_vit_blocks(model):
    base = list(model.children())[0]
    try:
        return list(base.model.base_model.model.model.visual.blocks)
    except AttributeError:
        pass
    try:
        return list(base.model.model.visual.blocks)
    except AttributeError:
        pass
    raise AttributeError("Cannot locate ViT blocks.")


def _run(cmd: list, cwd: str) -> bool:
    r = subprocess.run(cmd, capture_output=True, cwd=cwd,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    if r.returncode != 0:
        print(f"  ERROR: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(r.stderr.decode(errors="replace")[-300:], file=sys.stderr)
        return False
    return True


def _save_weight(w_np: np.ndarray, path: Path, transpose: bool = True):
    """量化权重为 int32，transpose=True 表示从 PyTorch (out,in) → zkFC (in,out)。"""
    if transpose:
        w_np = w_np.T
    np.round(w_np * SCALE).astype(np.int32).tofile(str(path))


def _save_bias(b_np: np.ndarray, path: Path):
    """量化 bias 为 int64（公开参数，不进入 IPA；用 int64 避免 SCALE^2 溢出）。"""
    np.round(b_np * SCALE * SCALE).astype(np.int64).tofile(str(path))


def extract_block_weights(blocks, block_idx: int, workdir: Path):
    block  = blocks[block_idx]
    prefix = f"vit-block-{block_idx}"

    # ── RMSNorm (norm1 = input_layernorm, norm2 = post_attention_layernorm) ──
    for which, attr in [("input_layernorm", "norm1"),
                        ("post_attention_layernorm", "norm2")]:
        w = getattr(block, attr).weight.detach().float().cpu().numpy()
        (workdir / f"{prefix}-{which}.weight-int.bin").write_bytes(
            np.round(w * SCALE).astype(np.int32).tobytes()
        )

    # ── Fused QKV → split into q/k/v ───────────────────────────────────────
    qkv_w = block.attn.qkv.weight.detach().float().cpu().numpy()  # (3840, 1280)
    qkv_b = block.attn.qkv.bias.detach().float().cpu().numpy() if block.attn.qkv.bias is not None else None
    for i, name in enumerate(["q_proj", "k_proj", "v_proj"]):
        w_part = qkv_w[i*VIT_HIDDEN:(i+1)*VIT_HIDDEN, :]   # (1280, 1280)
        _save_weight(w_part, workdir / f"{prefix}-self_attn.{name}.weight-int.bin")
        if qkv_b is not None:
            b_part = qkv_b[i*VIT_HIDDEN:(i+1)*VIT_HIDDEN]  # (1280,)
            _save_bias(b_part, workdir / f"{prefix}-self_attn.{name}.bias.bin")

    # ── o_proj ──────────────────────────────────────────────────────────────
    _save_weight(block.attn.proj.weight.detach().float().cpu().numpy(),
                 workdir / f"{prefix}-self_attn.o_proj.weight-int.bin")
    if block.attn.proj.bias is not None:
        _save_bias(block.attn.proj.bias.detach().float().cpu().numpy(),
                   workdir / f"{prefix}-self_attn.o_proj.bias.bin")

    # ── MLP ─────────────────────────────────────────────────────────────────
    mlp = block.mlp
    for name, module in [("mlp.gate_proj", mlp.gate_proj),
                          ("mlp.up_proj",   mlp.up_proj),
                          ("mlp.down_proj", mlp.down_proj)]:
        _save_weight(module.weight.detach().float().cpu().numpy(),
                     workdir / f"{prefix}-{name}.weight-int.bin")
        if module.bias is not None:
            _save_bias(module.bias.detach().float().cpu().numpy(),
                       workdir / f"{prefix}-{name}.bias.bin")

    print(f"  Block {block_idx}: weights saved (q/k/v split from fused QKV)")


def generate_pp(workdir: Path):
    """生成全块共用的 pp 文件（in_dim 为 key）。"""
    specs = [
        # (pp_filename, in_dim)  in_dim = pp size
        ("vit_self_attn.q_proj.weight", VIT_HIDDEN),
        ("vit_self_attn.k_proj.weight", VIT_HIDDEN),
        ("vit_self_attn.v_proj.weight", VIT_HIDDEN),
        ("vit_self_attn.o_proj.weight", VIT_HIDDEN),
        ("vit_mlp.gate_proj.weight",    VIT_HIDDEN),
        ("vit_mlp.up_proj.weight",      VIT_HIDDEN),
        ("vit_mlp.down_proj.weight",    VIT_INTER),
        ("vit_input_layernorm.weight",  VIT_HIDDEN),
        ("vit_post_attention_layernorm.weight", VIT_HIDDEN),
    ]
    for name, size in specs:
        pp = workdir / f"{name}-pp.bin"
        if pp.exists():
            print(f"  pp exists: {pp.name}")
            continue
        print(f"  ppgen {pp.name} size={size} ...", end=" ", flush=True)
        ok = _run([str(BIN_DIR / "ppgen"), str(size), str(pp)], cwd=str(BIN_DIR))
        print("OK" if ok else "FAIL")


def generate_commitment(block_idx: int, workdir: Path):
    prefix = f"vit-block-{block_idx}"
    specs = [
        # (pp_name, weight_name, in_dim, out_dim)
        ("vit_self_attn.q_proj.weight", "self_attn.q_proj.weight", VIT_HIDDEN, VIT_HIDDEN),
        ("vit_self_attn.k_proj.weight", "self_attn.k_proj.weight", VIT_HIDDEN, VIT_HIDDEN),
        ("vit_self_attn.v_proj.weight", "self_attn.v_proj.weight", VIT_HIDDEN, VIT_HIDDEN),
        ("vit_self_attn.o_proj.weight", "self_attn.o_proj.weight", VIT_HIDDEN, VIT_HIDDEN),
        ("vit_mlp.gate_proj.weight",    "mlp.gate_proj.weight",    VIT_HIDDEN, VIT_INTER),
        ("vit_mlp.up_proj.weight",      "mlp.up_proj.weight",      VIT_HIDDEN, VIT_INTER),
        ("vit_mlp.down_proj.weight",    "mlp.down_proj.weight",    VIT_INTER,  VIT_HIDDEN),
        ("vit_input_layernorm.weight",  "input_layernorm.weight",  1,          VIT_HIDDEN),
        ("vit_post_attention_layernorm.weight",
         "post_attention_layernorm.weight", 1, VIT_HIDDEN),
    ]
    for pp_name, w_name, in_d, out_d in specs:
        pp  = workdir / f"{pp_name}-pp.bin"
        w   = workdir / f"{prefix}-{w_name}-int.bin"
        com = workdir / f"{prefix}-{w_name}-commitment.bin"
        if com.exists():
            print(f"    commitment exists: {com.name}")
            continue
        if not w.exists():
            print(f"    ERROR: missing {w.name}", file=sys.stderr)
            continue
        print(f"    commit-param {com.name} ...", end=" ", flush=True)
        ok = _run([str(BIN_DIR / "commit-param"),
                   str(pp), str(w), str(com), str(in_d), str(out_d)],
                  cwd=str(BIN_DIR))
        print("OK" if ok else "FAIL")


def setup_vit_params(model=None, blocks=None, workdir=None):
    if blocks is None:
        blocks = list(range(32))
    if workdir is None:
        workdir = ROOT / "zkllm-workdir" / "jina-v4"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    own_model = False
    if model is None:
        print("Loading jina-v4 via SentenceTransformer ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device="cpu")
        own_model = True
        print("  loaded.")

    vit_blocks = _get_vit_blocks(model)

    print(f"\n── 提取 ViT block 权重 (blocks {blocks}) ──────────────────────────")
    for bi in blocks:
        extract_block_weights(vit_blocks, bi, workdir)

    if own_model:
        del model

    print("\n── 生成公共参数 (ppgen) ────────────────────────────────────────")
    generate_pp(workdir)

    print("\n── 生成承诺 (commit-param) ──────────────────────────────────────")
    for bi in blocks:
        print(f"  Block {bi}:")
        generate_commitment(bi, workdir)

    print("\n完成。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks",  nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    setup_vit_params(blocks=args.blocks, workdir=ROOT / args.workdir)


if __name__ == "__main__":
    main()

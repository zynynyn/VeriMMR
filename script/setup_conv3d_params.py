"""
生成 Conv3d Patch Embedding 权重文件及承诺参数

jina-v4 (Qwen2.5-VL) 的 ViT patch embedding：
  Conv3d(in=3, out=1280, kernel=(2,14,14), stride=(2,14,14), bias=False)
  等价于矩阵乘法：patch_flat (N, 1176) × W^T (1176, 1280)

输出文件（在 workdir 中）：
  conv3d_embed.weight-pp.bin                   (ppgen size=1176)
  {prefix}-conv3d_embed.weight-int.bin
  {prefix}-conv3d_embed.weight-commitment.bin

prefix 默认为 "conv3d"（此模块全局只有一个实例）。

用法（独立）：
  cd /root/autodl-tmp/UltraRAG
  python script/setup_conv3d_params.py

用法（pipeline 调用）：
  from script.setup_conv3d_params import setup_conv3d_params
  setup_conv3d_params(model=model, workdir=workdir)
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
SCALE      = 1 << 16
PREFIX     = "conv3d"


def _run(cmd: list, cwd: str) -> bool:
    r = subprocess.run(cmd, capture_output=True, cwd=cwd,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    if r.returncode != 0:
        print(f"  ERROR: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(r.stderr.decode(errors="replace")[-300:], file=sys.stderr)
        return False
    return True


def _get_conv3d(model):
    base = list(model.children())[0]
    try:
        return base.model.base_model.model.model.visual.patch_embed.proj
    except AttributeError:
        pass
    try:
        return base.model.model.visual.patch_embed.proj
    except AttributeError:
        pass
    raise AttributeError("Cannot locate Conv3d patch_embed.proj.")


def extract_weight(model, workdir: Path):
    """提取 Conv3d 权重，转置为 (patch_dim, out_dim) 并量化。"""
    conv = _get_conv3d(model)
    w = conv.weight.detach().float().cpu().numpy()   # (1280, 3, 2, 14, 14)
    out_dim  = w.shape[0]
    patch_dim = w[0].size                            # 3*2*14*14 = 1176

    # reshape: (out_dim, patch_dim) → transpose → (patch_dim, out_dim)
    w_flat  = w.reshape(out_dim, patch_dim)          # (1280, 1176)
    w_zkfc  = w_flat.T                               # (1176, 1280) — zkFC (in, out) layout

    out = workdir / f"{PREFIX}-conv3d_embed.weight-int.bin"
    np.round(w_zkfc * SCALE).astype(np.int32).tofile(str(out))
    print(f"  weight saved: {out.name}  shape={w_zkfc.shape}  mean_abs={abs(w_zkfc).mean():.5f}")
    return patch_dim, out_dim


def setup_conv3d_params(model=None, workdir=None):
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

    print("\n── 提取 Conv3d 权重 ─────────────────────────────────────────────")
    patch_dim, out_dim = extract_weight(model, workdir)

    if own_model:
        del model

    print("\n── 生成公共参数 (ppgen) ─────────────────────────────────────────")
    pp = workdir / "conv3d_embed.weight-pp.bin"
    if pp.exists():
        print(f"  pp exists: {pp.name}")
    else:
        print(f"  ppgen size={patch_dim} ...", end=" ", flush=True)
        ok = _run([str(BIN_DIR / "ppgen"), str(patch_dim), str(pp)], cwd=str(BIN_DIR))
        print("OK" if ok else "FAIL")

    print("\n── 生成承诺 (commit-param) ──────────────────────────────────────")
    w_path   = workdir / f"{PREFIX}-conv3d_embed.weight-int.bin"
    com_path = workdir / f"{PREFIX}-conv3d_embed.weight-commitment.bin"
    if com_path.exists():
        print(f"  commitment exists: {com_path.name}")
    else:
        print(f"  commit-param {com_path.name} ...", end=" ", flush=True)
        ok = _run([str(BIN_DIR / "commit-param"),
                   str(pp), str(w_path), str(com_path),
                   str(patch_dim), str(out_dim)],
                  cwd=str(BIN_DIR))
        print("OK" if ok else "FAIL")

    print("\n完成。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    setup_conv3d_params(workdir=ROOT / args.workdir)


if __name__ == "__main__":
    main()

"""
生成 PatchMerger 权重文件及承诺参数

jina-v4 (Qwen2.5-VL) 的 PatchMerger 结构：
  ln_q:   RMSNorm(vit_dim=1280)
  mlp.0:  Linear(5120, 5120)
  mlp.1:  GELU()
  mlp.2:  Linear(5120, 2048)

输出文件（在 workdir 中）：
  全局共用：
    patchmerger_layernorm.weight-pp.bin   (ppgen size=1280)
    patchmerger_fc1.weight-pp.bin         (ppgen size=5120)
    patchmerger_fc2.weight-pp.bin         (ppgen size=5120)
  per-instance（prefix="patchmerger"）：
    patchmerger-patchmerger_layernorm.weight-int.bin
    patchmerger-patchmerger_layernorm.weight-commitment.bin
    patchmerger-patchmerger_fc1.weight-int.bin
    patchmerger-patchmerger_fc1.weight-commitment.bin
    patchmerger-patchmerger_fc2.weight-int.bin
    patchmerger-patchmerger_fc2.weight-commitment.bin

用法（独立）：
  cd /root/autodl-tmp/UltraRAG
  python script/setup_patchmerger_params.py [--workdir zkllm-workdir/jina-v4]

用法（pipeline 调用）：
  from script.setup_patchmerger_params import setup_patchmerger_params
  setup_patchmerger_params(model=model, workdir=workdir)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT      = Path(__file__).parent.parent.resolve()
BIN_DIR   = ROOT / "src" / "zkllm"
MODEL_PATH = "/root/autodl-tmp/models/jina-embeddings-v4"
VIT_DIM    = 1280
MERGED_DIM = 5120   # VIT_DIM * group_size(4)
OUT_DIM    = 2048
SCALE      = 1 << 16
PREFIX     = "patchmerger"


def _get_merger(model):
    """从 SentenceTransformer 包装的 jina-v4 获取 PatchMerger 模块。"""
    base = list(model.children())[0]
    # 通常路径：base.model.base_model.model.model.visual.merger
    try:
        return base.model.base_model.model.model.visual.merger
    except AttributeError:
        pass
    # 备用：直接从 visual 里找
    try:
        return base.model.model.visual.merger
    except AttributeError:
        pass
    raise AttributeError("Cannot locate PatchMerger in model. Check model hierarchy.")


def _run(cmd: list, cwd: str) -> bool:
    r = subprocess.run(cmd, capture_output=True, cwd=cwd,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    if r.returncode != 0:
        print(f"  ERROR: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(r.stderr.decode(errors="replace")[-300:], file=sys.stderr)
        return False
    return True


def extract_weights(model, workdir: Path) -> bool:
    """提取三个权重，量化为 int32 binary。"""
    merger = _get_merger(model)

    # ln_q: RMSNorm weight, shape=(1280,)
    w_ln = merger.ln_q.weight.detach().float().cpu().numpy()
    out  = workdir / f"{PREFIX}-patchmerger_layernorm.weight-int.bin"
    np.round(w_ln * SCALE).astype(np.int32).tofile(str(out))
    print(f"    {out.name}  std={abs(w_ln).mean():.4f}")

    # mlp.0: Linear(5120→5120) weight, shape=(5120,5120)
    w_fc1 = merger.mlp[0].weight.detach().float().cpu().numpy()  # (out=5120, in=5120)
    out   = workdir / f"{PREFIX}-patchmerger_fc1.weight-int.bin"
    # zkFC expects row-major (in_dim, out_dim) → transpose from PyTorch (out, in)
    np.round(w_fc1.T * SCALE).astype(np.int32).tofile(str(out))
    print(f"    {out.name}  shape={w_fc1.shape}  std={abs(w_fc1).mean():.4f}")

    # mlp.2: Linear(5120→2048) weight, shape=(2048,5120)
    w_fc2 = merger.mlp[2].weight.detach().float().cpu().numpy()  # (out=2048, in=5120)
    out   = workdir / f"{PREFIX}-patchmerger_fc2.weight-int.bin"
    np.round(w_fc2.T * SCALE).astype(np.int32).tofile(str(out))
    print(f"    {out.name}  shape={w_fc2.shape}  std={abs(w_fc2).mean():.4f}")
    return True


def generate_pp(name: str, size: int, workdir: Path) -> bool:
    pp = workdir / f"patchmerger_{name}.weight-pp.bin"
    if pp.exists():
        print(f"  pp exists: {pp.name}")
        return True
    print(f"  ppgen {pp.name} size={size} ...", end=" ", flush=True)
    ok = _run([str(BIN_DIR / "ppgen"), str(size), str(pp)], cwd=str(BIN_DIR))
    print("OK" if ok else "FAIL")
    return ok


def generate_commitment(name: str, in_dim: int, out_dim: int, workdir: Path) -> bool:
    pp  = workdir / f"patchmerger_{name}.weight-pp.bin"
    w   = workdir / f"{PREFIX}-patchmerger_{name}.weight-int.bin"
    com = workdir / f"{PREFIX}-patchmerger_{name}.weight-commitment.bin"
    if com.exists():
        print(f"    commitment exists: {com.name}")
        return True
    print(f"    commit-param {com.name} ...", end=" ", flush=True)
    ok = _run([str(BIN_DIR / "commit-param"),
               str(pp), str(w), str(com), str(in_dim), str(out_dim)],
              cwd=str(BIN_DIR))
    print("OK" if ok else "FAIL")
    return ok


def setup_patchmerger_params(model=None, workdir=None):
    """主入口：可从流水线调用（传入已加载的 SentenceTransformer model）。"""
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

    print("\n── 提取 PatchMerger 权重 ───────────────────────────────────────")
    extract_weights(model, workdir)

    if own_model:
        del model

    print("\n── 生成公共参数 (ppgen) ────────────────────────────────────────")
    generate_pp("layernorm", VIT_DIM,    workdir)
    generate_pp("fc1",       MERGED_DIM, workdir)
    generate_pp("fc2",       MERGED_DIM, workdir)

    print("\n── 生成承诺 (commit-param) ──────────────────────────────────────")
    generate_commitment("layernorm", 1,          VIT_DIM,    workdir)
    generate_commitment("fc1",       MERGED_DIM, MERGED_DIM, workdir)
    generate_commitment("fc2",       MERGED_DIM, OUT_DIM,    workdir)

    print("\n完成。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    setup_patchmerger_params(workdir=ROOT / args.workdir)


if __name__ == "__main__":
    main()

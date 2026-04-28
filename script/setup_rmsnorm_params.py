"""
生成 RMSNorm 权重文件及承诺参数

对 jina-embeddings-v4 指定层的 input_layernorm 和 post_attention_layernorm：
  1. 从模型提取权重 (float32, shape=(embed_dim,))，量化为 int32 保存
  2. 用 ppgen 生成公共参数（全层共用，只生成一次）
  3. 用 commit-param 生成每层的承诺文件

设计为可独立运行，也可从流水线中以已加载模型调用：
  from script.setup_rmsnorm_params import setup_rmsnorm_params
  setup_rmsnorm_params(model, layers=[30..35], workdir=...)

用法（独立）：
  cd /root/autodl-tmp/UltraRAG
  python script/setup_rmsnorm_params.py [--layers 30 31 32 33 34 35]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).parent.parent.resolve()
BIN_DIR  = ROOT / "src" / "zkllm"
EMBED_DIM = 2048
SCALE     = 1 << 16
MODEL_PATH = "/root/autodl-tmp/models/jina-embeddings-v4"


def _get_layers(model):
    """与 build_corpus_zkllm_proofs.py 保持一致的层路径。"""
    return (list(model.children())[0]
            .model.base_model.model.model.language_model.layers)


def _run(cmd: list, cwd: str) -> bool:
    r = subprocess.run(cmd, capture_output=True, cwd=cwd,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    if r.returncode != 0:
        print(f"  ERROR: {' '.join(map(str, cmd))}", file=sys.stderr)
        print(r.stderr.decode(errors="replace")[-300:], file=sys.stderr)
        return False
    return True


def extract_layernorm_weights(model, layer_idx: int, workdir: Path):
    """提取单层两个 RMSNorm 权重，量化保存为 int32 binary。"""
    layers = _get_layers(model)
    layer  = layers[layer_idx]
    for which, module in [
        ("input_layernorm",          layer.input_layernorm),
        ("post_attention_layernorm", layer.post_attention_layernorm),
    ]:
        w     = module.weight.detach().float().cpu().numpy()   # (2048,)
        w_int = np.round(w * SCALE).astype(np.int32)
        out   = workdir / f"layer-{layer_idx}-{which}.weight-int.bin"
        w_int.tofile(str(out))
        print(f"    {out.name}  std={w_int.std():.1f}")


def generate_pp(which: str, workdir: Path) -> bool:
    """生成全层共用的公共参数文件（已存在则跳过）。"""
    pp = workdir / f"{which}.weight-pp.bin"
    if pp.exists():
        print(f"  pp exists: {pp.name}")
        return True
    print(f"  ppgen {pp.name} size={EMBED_DIM} ...", end=" ", flush=True)
    ok = _run([str(BIN_DIR / "ppgen"), str(EMBED_DIM), str(pp)],
              cwd=str(BIN_DIR))
    print("OK" if ok else "FAIL")
    return ok


def generate_commitment(which: str, layer_idx: int, workdir: Path) -> bool:
    """用 commit-param 为单层生成承诺文件。"""
    pp  = workdir / f"{which}.weight-pp.bin"
    w   = workdir / f"layer-{layer_idx}-{which}.weight-int.bin"
    com = workdir / f"layer-{layer_idx}-{which}.weight-commitment.bin"
    if com.exists():
        print(f"    commitment exists: {com.name}")
        return True
    if not w.exists():
        print(f"    ERROR: missing {w}", file=sys.stderr)
        return False
    print(f"    commit-param {com.name} ...", end=" ", flush=True)
    ok = _run([str(BIN_DIR / "commit-param"),
               str(pp), str(w), str(com), "1", str(EMBED_DIM)],
              cwd=str(BIN_DIR))
    print("OK" if ok else "FAIL")
    return ok


def setup_rmsnorm_params(model=None, layers=None, workdir=None):
    """
    主入口：可从流水线直接调用（传入已加载的 SentenceTransformer model）。
    model=None 时自动加载。
    """
    if layers is None:
        layers = list(range(30, 36))
    if workdir is None:
        workdir = ROOT / "zkllm-workdir" / "jina-v4"
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    own_model = False
    if model is None:
        print(f"Loading jina-v4 via SentenceTransformer ...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_PATH, trust_remote_code=True,
                                    device="cpu")   # cpu 足够，只需取权重
        own_model = True
        print("  loaded.")

    # 1. 提取权重
    print("\n── 提取 RMSNorm 权重 ──────────────────────────────────────")
    for li in layers:
        print(f"  Layer {li}:")
        extract_layernorm_weights(model, li, workdir)

    if own_model:
        del model

    # 2. 生成 pp（全层共用）
    print("\n── 生成公共参数 (ppgen) ────────────────────────────────────")
    for which in ["input_layernorm", "post_attention_layernorm"]:
        generate_pp(which, workdir)

    # 3. 生成承诺
    print("\n── 生成承诺 (commit-param) ──────────────────────────────────")
    for li in layers:
        print(f"  Layer {li}:")
        for which in ["input_layernorm", "post_attention_layernorm"]:
            generate_commitment(which, li, workdir)

    print("\n完成。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int,
                        default=list(range(30, 36)))
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    setup_rmsnorm_params(
        layers=args.layers,
        workdir=ROOT / args.workdir,
    )


if __name__ == "__main__":
    main()

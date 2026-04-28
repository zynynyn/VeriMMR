"""
生成 GELU lookup table (gelu-table.bin)

格式与 swiglu-table.bin 完全一致：
  - 每个条目一个 int32（fr-tensor.cu from_int_bin 格式）
  - 与 tLookupRangeMapping 配合使用：
      tLookupRangeMapping gelu(-(1<<18), 1<<19, gelu_values)
  - 输入整数 x 代表实数 x / 2^12（与 SwiGLU 相同的缩放）
  - 输出值：gelu(x / 2^12) × 2^16（量化为 int32）
  - 表大小：2^19 = 524288 条目（2MB，适配 n_patches ≤ 256 的 PatchMerger）

覆盖范围：
  - 整数 x ∈ [-2^18, 2^18) = [-262144, 262144)
  - 实数 x/2^12 ∈ [-64.0, 64.0)
  - GELU 在 |x| > 6 时已基本饱和（gelu(-large) ≈ 0, gelu(large) ≈ x）

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/gen_gelu_table.py [--out src/zkllm/gelu-table.bin]
"""

import argparse
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()

SCALE_IN  = 1 << 12   # 输入整数 → 实数：x / SCALE_IN
SCALE_OUT = 1 << 16   # 实数 → 输出整数：gelu(x) × SCALE_OUT
LOW       = -(1 << 18)
LEN       = 1 << 19   # 524288


def gelu(x: np.ndarray) -> np.ndarray:
    """精确 GELU（erf 版本，与 PyTorch default 一致）。"""
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2)))


def generate_table(out_path: str):
    print(f"Generating GELU table: {LEN} entries, low={LOW} ...")
    xs = (np.arange(LEN, dtype=np.float64) + LOW) / SCALE_IN
    ys = gelu(xs)
    table = np.round(ys * SCALE_OUT).astype(np.int32)
    table.tofile(out_path)
    print(f"Saved {LEN} int32 entries ({LEN * 4 // 1024 // 1024} MB) → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="src/zkllm/gelu-table.bin")
    args = parser.parse_args()

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_table(str(out))

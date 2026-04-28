"""
一次性生成所有视觉组件的权重参数（加载一次模型）

按顺序执行：
  1. Conv3d patch embedding (仅 1 个全局权重)
  2. PatchMerger (仅 1 个全局实例)
  3. ViT 32 blocks (--blocks 指定，默认全部 0-31)

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/setup_all_visual_params.py [--blocks 0 1 ... 31] [--workdir ...]
  # 仅跑部分（例如先测试 block 0 和 31）:
  python script/setup_all_visual_params.py --blocks 0 31
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks",  nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    workdir = (ROOT / args.workdir).resolve()

    print("Loading jina-v4 via SentenceTransformer (cpu) ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "/root/autodl-tmp/models/jina-embeddings-v4",
        trust_remote_code=True, device="cpu"
    )
    print("  model loaded.\n")

    # 1. Conv3d
    print("=" * 60)
    print("Step 1: Conv3d patch embedding")
    print("=" * 60)
    from script.setup_conv3d_params import setup_conv3d_params
    setup_conv3d_params(model=model, workdir=workdir)

    # 2. PatchMerger
    print("\n" + "=" * 60)
    print("Step 2: PatchMerger")
    print("=" * 60)
    from script.setup_patchmerger_params import setup_patchmerger_params
    setup_patchmerger_params(model=model, workdir=workdir)

    # 3. ViT blocks
    print("\n" + "=" * 60)
    print(f"Step 3: ViT blocks {args.blocks}")
    print("=" * 60)
    from script.setup_vit_params import setup_vit_params
    setup_vit_params(model=model, blocks=args.blocks, workdir=workdir)

    del model
    print("\n全部完成！")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()

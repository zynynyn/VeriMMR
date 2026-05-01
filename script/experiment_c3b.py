"""
C3b v2：随机化全组件 zkLLM 权重篡改检测率实验

改进（相比 v1）：
  - 随机抽样 ViT 块（默认 8 个，从 0-31）和 LLM 层（默认 12 层，从 0-35）
  - 每个块/层随机选择 N 个权重矩阵（FFN: gate/up/down；Attn: q/k/v）
  - 双模式：grid（固定档位 × 重复，论文表格）/ random（对数均匀采样）
  - 跨模态标注：side=image（ViT/Conv3d/PatchMerger）vs side=text（LLM）
  - 输出到 c3b_detection_rate_v2.json

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/experiment_c3b.py --mode grid --vit-blocks 8 --llm-layers 12
  python script/experiment_c3b.py --mode random --n-random 30 --vit-blocks 4 --llm-layers 6
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).parent.parent.resolve()
BIN_DIR  = ROOT / "src" / "zkllm"
CWD      = ROOT / "src" / "zkllm"
WORKDIR  = ROOT / "zkllm-workdir" / "jina-v4"
OUT_FILE = ROOT / "notes" / "experiment_results" / "c3b_detection_rate_v2.json"

SCALE = 1 << 16

# ViT 架构参数（jina-v4 ViT）
VIT_SEQ = 1024
VIT_HID = 1280
VIT_FFN = 3420
VIT_KV  = 1280  # MHA: kv_dim = embed_dim

# LLM 架构参数（jina-v4 文本侧，Qwen2.5 风格）
LLM_SEQ = 512
LLM_HID = 2048
LLM_FFN = 11008
LLM_KV  = 256   # GQA: kv_dim = 256

# 每个模块可测试的权重矩阵
FFN_KEYS  = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ATTN_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
ALL_WEIGHT_KEYS = FFN_KEYS + ATTN_KEYS  # 6 种


# ── helpers ────────────────────────────────────────────────────────────────────

def _bin(name): return str(BIN_DIR / name)
def _env(gpu=0): return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}

sys.path.insert(0, str(ROOT))
from script.verify_layers import verify_ipa_cpp, verify_ipa_python


def _verify_ipa(proof_path: str) -> dict:
    com_path = proof_path.replace("-ipa-proof.bin", ".weight-commitment.bin")
    if (BIN_DIR / "verify-ipa").exists() and Path(com_path).exists():
        return verify_ipa_cpp(proof_path, com_path, gpu_id=0)
    return verify_ipa_python(proof_path, com_path)


def _tamper(src: Path, dst: Path, ratio: float, rng: np.random.Generator) -> dict:
    W = np.fromfile(str(src), dtype=np.int32)
    sigma = float(W.std()) or 1.0
    n = max(1, int(len(W) * ratio))
    idx = rng.choice(len(W), n, replace=False)
    W2 = W.copy()
    W2[idx] = rng.normal(0, sigma, n).round().astype(np.int32)
    W2.tofile(str(dst))
    return {"n_total": int(len(W)), "n_tampered": n, "sigma": round(sigma, 2)}


def _run(cmd, cwd=None, gpu=0):
    r = subprocess.run(cmd, capture_output=True,
                       cwd=str(cwd or CWD), env=_env(gpu))
    return r.returncode, r.stderr.decode(errors="replace")


def _rand_input(path: Path, rows: int, cols: int):
    if not path.exists():
        rng_local = np.random.default_rng(42)
        (rng_local.standard_normal((rows, cols)) * SCALE).astype(np.int32).tofile(str(path))


# ── ViT prover functions ────────────────────────────────────────────────────────

def prove_vit_ffn(block: int, workdir: Path) -> tuple[int, str]:
    prefix = f"vit-block-{block}"
    bdir   = workdir / f"vit-b{block}"
    inp    = bdir / f"{prefix}-c3b-ffn-inp.bin"
    out    = bdir / f"{prefix}-c3b-ffn-out.bin"
    _rand_input(inp, VIT_SEQ, VIT_HID)
    rc, err = _run([_bin("ffn"), str(inp), str(VIT_SEQ), str(VIT_HID), str(VIT_FFN),
                    str(bdir), prefix, str(out)], cwd=CWD)
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return rc, err


def prove_vit_attn(block: int, workdir: Path) -> tuple[int, str]:
    prefix = f"vit-block-{block}"
    bdir   = workdir / f"vit-b{block}"
    inp    = bdir / f"{prefix}-c3b-attn-inp.bin"
    out    = bdir / f"{prefix}-c3b-attn-out.bin"
    _rand_input(inp, VIT_SEQ, VIT_HID)
    rc, err = _run([_bin("self-attn"), "linear", str(inp), str(VIT_SEQ), str(VIT_HID),
                    str(bdir), prefix, str(out), str(VIT_KV)], cwd=CWD)
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    # clean up temp Q/K/V written by linear mode
    for sfx in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (bdir / f"{prefix}-{sfx}").unlink(missing_ok=True)
    return rc, err


# ── LLM prover functions ────────────────────────────────────────────────────────

def prove_llm_ffn(layer: int, workdir: Path) -> tuple[int, str]:
    prefix = f"layer-{layer}"
    inp = workdir / f"{prefix}-c3b-ffn-inp.bin"
    out = workdir / f"{prefix}-c3b-ffn-out.bin"
    _rand_input(inp, LLM_SEQ, LLM_HID)
    rc, err = _run([_bin("ffn"), str(inp), str(LLM_SEQ), str(LLM_HID), str(LLM_FFN),
                    str(workdir), prefix, str(out)])
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return rc, err


def prove_llm_attn(layer: int, workdir: Path) -> tuple[int, str]:
    prefix = f"layer-{layer}"
    inp = workdir / f"{prefix}-c3b-attn-inp.bin"
    out = workdir / f"{prefix}-c3b-attn-out.bin"
    _rand_input(inp, LLM_SEQ, LLM_HID)
    rc, err = _run([_bin("self-attn"), "linear", str(inp), str(LLM_SEQ), str(LLM_HID),
                    str(workdir), prefix, str(out), str(LLM_KV)])
    inp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    for sfx in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (workdir / f"{prefix}-{sfx}").unlink(missing_ok=True)
    return rc, err


# ── component builder ──────────────────────────────────────────────────────────

def build_components(workdir: Path, vit_blocks: list[int], llm_layers: list[int],
                     n_weights: int, rng: np.random.Generator) -> list[dict]:
    """
    For each selected block/layer, randomly pick up to n_weights weight keys.
    Weight keys include FFN (gate/up/down) and Attn-QKV (q/k/v).
    Entries are skipped if the weight-int.bin file doesn't exist.
    """
    components = []

    for block in sorted(vit_blocks):
        bdir   = workdir / f"vit-b{block}"
        prefix = f"vit-block-{block}"
        if not bdir.exists():
            print(f"  [skip] vit-b{block} directory not found")
            continue

        chosen = rng.choice(ALL_WEIGHT_KEYS, min(n_weights, len(ALL_WEIGHT_KEYS)),
                             replace=False).tolist()
        for wkey in chosen:
            wf = bdir / f"{prefix}-{wkey}.weight-int.bin"
            pf = bdir / f"{prefix}-{wkey}-ipa-proof.bin"
            if not wf.exists():
                print(f"  [skip] {wf.name} not found")
                continue
            pfn = (lambda b=block, wd=workdir: prove_vit_ffn(b, wd)
                   if wkey.startswith("mlp.") else
                   lambda b=block, wd=workdir: prove_vit_attn(b, wd))
            # Need correct lambda capture per wkey
            if wkey.startswith("mlp."):
                pfn = lambda b=block, wd=workdir: prove_vit_ffn(b, wd)
            else:
                pfn = lambda b=block, wd=workdir: prove_vit_attn(b, wd)
            components.append({
                "label":       f"vit_b{block}/{wkey}",
                "component":   f"vit_block_{block}/{wkey}",
                "side":        "image",
                "weight_file": wf,
                "proof_file":  pf,
                "prove_fn":    pfn,
            })

    for layer in sorted(llm_layers):
        prefix = f"layer-{layer}"
        chosen = rng.choice(ALL_WEIGHT_KEYS, min(n_weights, len(ALL_WEIGHT_KEYS)),
                             replace=False).tolist()
        for wkey in chosen:
            wf = workdir / f"{prefix}-{wkey}.weight-int.bin"
            pf = workdir / f"{prefix}-{wkey}-ipa-proof.bin"
            if not wf.exists():
                print(f"  [skip] {wf.name} not found")
                continue
            if wkey.startswith("mlp."):
                pfn = lambda l=layer, wd=workdir: prove_llm_ffn(l, wd)
            else:
                pfn = lambda l=layer, wd=workdir: prove_llm_attn(l, wd)
            components.append({
                "label":       f"llm_l{layer}/{wkey}",
                "component":   f"llm_layer_{layer}/{wkey}",
                "side":        "text",
                "weight_file": wf,
                "proof_file":  pf,
                "prove_fn":    pfn,
            })

    return components


# ── single trial ───────────────────────────────────────────────────────────────

def run_one(comp: dict, ratio: float | None, rep: int,
            rng: np.random.Generator) -> dict:
    weight_src = Path(comp["weight_file"])
    proof_path = str(comp["proof_file"])

    if not weight_src.exists():
        return {"error": f"weight not found: {weight_src}"}

    tmp_weight  = weight_src.parent / (weight_src.name + ".tamper_bak")
    tamper_info = None
    if ratio is not None:
        shutil.copy(str(weight_src), str(tmp_weight))
        tamper_info = _tamper(weight_src, weight_src, ratio, rng)

    try:
        t0 = time.perf_counter()
        rc, stderr = comp["prove_fn"]()
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
    finally:
        if ratio is not None and tmp_weight.exists():
            shutil.copy(str(tmp_weight), str(weight_src))
            tmp_weight.unlink(missing_ok=True)

    if rc != 0:
        return {
            "ratio": ratio, "rep": rep, "prover_ok": False,
            "elapsed_ms": elapsed_ms,
            "stderr_tail": stderr[-300:],
        }

    ipa = _verify_ipa(proof_path)
    detected = (ratio is not None) and not ipa.get("binding_ok", True)

    return {
        "ratio":       ratio,
        "rep":         rep,
        "tamper_info": tamper_info,
        "prover_ok":   True,
        "elapsed_ms":  elapsed_ms,
        "fold_ok":     ipa.get("fold_ok"),
        "binding_ok":  ipa.get("binding_ok"),
        "detected":    detected,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C3b v2 随机化篡改检测率实验")
    parser.add_argument("--workdir",    default="zkllm-workdir/jina-v4")
    parser.add_argument("--mode",       choices=["grid", "random"], default="grid",
                        help="grid: 固定比例 × 重复；random: 对数均匀采样")
    parser.add_argument("--vit-blocks", type=int, default=8,
                        help="随机抽取的 ViT 块数（从 0-31）")
    parser.add_argument("--llm-layers", type=int, default=12,
                        help="随机抽取的 LLM 层数（从 0-35）")
    parser.add_argument("--n-weights",  type=int, default=3,
                        help="每个块/层随机选取的权重矩阵数（最多 6）")
    parser.add_argument("--ratios",     nargs="+", type=float,
                        default=[0.001, 0.01, 0.05],
                        help="grid 模式篡改比例")
    parser.add_argument("--repeats",    type=int, default=3,
                        help="grid 模式每个比例的重复次数")
    parser.add_argument("--n-random",   type=int, default=30,
                        help="random 模式每个组件的随机采样次数")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out",        default=str(OUT_FILE))
    args = parser.parse_args()

    rng     = np.random.default_rng(args.seed)
    workdir = (ROOT / args.workdir).resolve()

    # 随机抽取块/层
    n_vit = min(args.vit_blocks, 32)
    n_llm = min(args.llm_layers, 36)
    vit_blocks = sorted(rng.choice(32, n_vit, replace=False).tolist())
    llm_layers = sorted(rng.choice(36, n_llm, replace=False).tolist())

    print("=" * 64)
    print("C3b v2  随机化全组件篡改检测率实验")
    print(f"  模式        : {args.mode}")
    print(f"  ViT 块      : {vit_blocks}")
    print(f"  LLM 层      : {llm_layers}")
    print(f"  权重/组件   : {args.n_weights}")
    if args.mode == "grid":
        print(f"  篡改比例    : {args.ratios}  × {args.repeats} repeats")
    else:
        print(f"  随机采样    : {args.n_random} 次，对数均匀 [0.001, 0.1]")
    print("=" * 64)

    components = build_components(workdir, vit_blocks, llm_layers, args.n_weights, rng)
    print(f"\n有效组件数: {len(components)}\n")
    if not components:
        print("没有找到有效组件，请检查 workdir 路径和已提取的权重文件。")
        return

    # 构建 (ratio, rep) 列表
    if args.mode == "grid":
        ratio_reps: list[tuple[float, int]] = [
            (r, rep) for r in args.ratios for rep in range(args.repeats)
        ]
    else:
        sampled = np.exp(
            rng.uniform(np.log(0.001), np.log(0.1), args.n_random)
        ).tolist()
        ratio_reps = [(float(r), i) for i, r in enumerate(sampled)]

    all_results: list[dict] = []
    t_total = time.perf_counter()

    for comp in components:
        label = comp["label"]
        side  = comp["side"]
        print(f"\n[{label}]  ({side})")

        # 正常权重验证
        r = run_one(comp, ratio=None, rep=0, rng=rng)
        r.update({"label": label, "component": comp["component"],
                  "side": side, "tamper": False})
        all_results.append(r)
        status = "✓" if r.get("binding_ok") else "✗"
        print(f"  clean      {status}  ({r.get('elapsed_ms', 0)}ms)")

        # 篡改验证
        for ratio, rep in ratio_reps:
            r = run_one(comp, ratio=ratio, rep=rep, rng=rng)
            r.update({"label": label, "component": comp["component"],
                      "side": side, "tamper": True})
            all_results.append(r)
            flag = "✓det" if r.get("detected") else "✗miss"
            print(f"  ratio={ratio:.4f}  rep={rep}  {flag}  ({r.get('elapsed_ms', 0)}ms)")

    elapsed_total = round(time.perf_counter() - t_total)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    clean_pass   = sum(1 for r in all_results if not r.get("tamper") and r.get("binding_ok"))
    clean_total  = sum(1 for r in all_results if not r.get("tamper"))
    detect_total = sum(1 for r in all_results if r.get("tamper") and r.get("detected"))
    tamper_total = sum(1 for r in all_results if r.get("tamper"))

    print("\n" + "=" * 64)
    print("汇总")
    print(f"  正常权重通过率: {clean_pass}/{clean_total}")
    print(f"  篡改总检出率:   {detect_total}/{tamper_total}"
          f"  ({detect_total / max(tamper_total, 1) * 100:.1f}%)")

    for side in ["image", "text"]:
        det = sum(1 for r in all_results
                  if r.get("tamper") and r.get("side") == side and r.get("detected"))
        tot = sum(1 for r in all_results
                  if r.get("tamper") and r.get("side") == side)
        print(f"  {side} 检出率:  {det}/{tot}  ({det / max(tot, 1) * 100:.1f}%)")

    if args.mode == "grid":
        for ratio in args.ratios:
            det = sum(1 for r in all_results
                      if r.get("tamper") and abs(r.get("ratio", -1) - ratio) < 1e-9
                      and r.get("detected"))
            tot = sum(1 for r in all_results
                      if r.get("tamper") and abs(r.get("ratio", -1) - ratio) < 1e-9)
            print(f"  ratio={ratio}  {det}/{tot}  ({det / max(tot, 1) * 100:.1f}%)")

    print(f"\n总耗时: {elapsed_total}s")
    print("=" * 64)

    summary = {
        "experiment":              "C3b_v2",
        "description":             "随机化全组件 zkLLM 权重篡改检测率（图像+文字侧，IPA 承诺绑定）",
        "mode":                    args.mode,
        "vit_blocks":              vit_blocks,
        "llm_layers":              llm_layers,
        "n_weights_per_component": args.n_weights,
        "tamper_ratios":           args.ratios if args.mode == "grid" else "log-uniform[0.001,0.1]",
        "n_repeats":               args.repeats if args.mode == "grid" else args.n_random,
        "seed":                    args.seed,
        "clean_pass":              clean_pass,
        "clean_total":             clean_total,
        "detect_total":            detect_total,
        "tamper_total":            tamper_total,
        "detection_rate":          round(detect_total / max(tamper_total, 1), 4),
        "elapsed_s":               elapsed_total,
        "results":                 all_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()

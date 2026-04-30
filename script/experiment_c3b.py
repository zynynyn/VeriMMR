"""
C3b：全组件 zkLLM 权重篡改检测率实验

覆盖所有 Phase 3 组件（图像侧 + 文字侧）：
  - Conv3d patch embedding
  - ViT blocks（抽样 4 个：0, 7, 15, 31）
  - PatchMerger（fc1, fc2）
  - LLM decoder layers（抽样 6 层：0, 7, 14, 21, 28, 35）

每个组件 × 每个权重矩阵：
  - 1 次正常验证（期望 binding_ok=True）
  - 3 个篡改比例 × N_REPEATS 次随机篡改（期望 binding_ok=False）

篡改方法：随机选取比例 p 的元素，替换为 N(0, σ²) 高斯噪声（σ = std(W)）

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/experiment_c3b.py [--repeats 3] [--ratios 0.001 0.01 0.05]
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
OUT_FILE = ROOT / "notes" / "experiment_results" / "c3b_detection_rate.json"

SCALE = 1 << 16

# ── binary 路径 ────────────────────────────────────────────────────────────────

def _bin(name): return str(BIN_DIR / name)
def _env(gpu=0): return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}


# ── IPA 验证（复用 verify_layers.py 里的 verify_ipa_cpp）────────────────────────

sys.path.insert(0, str(ROOT))
from script.verify_layers import verify_ipa_cpp, verify_ipa_python


def _verify_ipa(proof_path: str) -> dict:
    com_path = proof_path.replace("-ipa-proof.bin", ".weight-commitment.bin")
    if (BIN_DIR / "verify-ipa").exists() and Path(com_path).exists():
        return verify_ipa_cpp(proof_path, com_path, gpu_id=0)
    return verify_ipa_python(proof_path)


# ── 权重篡改 ────────────────────────────────────────────────────────────────────

def _tamper(src: Path, dst: Path, ratio: float, rng: np.random.Generator) -> dict:
    W = np.fromfile(str(src), dtype=np.int32)
    sigma = float(W.std()) or 1.0
    n = max(1, int(len(W) * ratio))
    idx = rng.choice(len(W), n, replace=False)
    W2 = W.copy()
    W2[idx] = rng.normal(0, sigma, n).round().astype(np.int32)
    W2.tofile(str(dst))
    return {"n_total": int(len(W)), "n_tampered": n, "sigma": round(sigma, 2)}


# ── 各组件 prover 调用 ─────────────────────────────────────────────────────────

def _run(cmd, cwd=None, gpu=0):
    r = subprocess.run(cmd, capture_output=True,
                       cwd=str(cwd or CWD), env=_env(gpu))
    return r.returncode, r.stderr.decode(errors="replace")


def _rand_input(path: Path, rows: int, cols: int):
    if not path.exists():
        rng = np.random.default_rng(42)
        (rng.standard_normal((rows, cols)) * SCALE).astype(np.int32).tofile(str(path))


def prove_conv3d(workdir: Path) -> tuple[int, str]:
    """运行 conv3d-embed，生成 IPA proof。"""
    prefix = "conv3d"
    n_patches, patch_dim, out_dim = 32, 1176, 1280
    patches = workdir / f"{prefix}-patches.bin"
    output  = workdir / f"{prefix}-tamper-out.bin"
    rng = np.random.default_rng(42)
    (rng.standard_normal((n_patches, patch_dim)) * SCALE).astype(np.int32).tofile(str(patches))
    rc, err = _run([_bin("conv3d-embed"),
                    str(patches), str(n_patches), str(patch_dim), str(out_dim),
                    str(workdir), prefix, str(output)])
    patches.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    return rc, err


def prove_vit_ffn(block: int, workdir: Path) -> tuple[int, str]:
    prefix = f"vit-block-{block}"
    bdir   = workdir / f"vit-b{block}"
    inp    = bdir / f"{prefix}-tamper-ffn-inp.bin"
    out    = bdir / f"{prefix}-tamper-ffn-out.bin"
    _rand_input(inp, 1024, 1280)
    rc, err = _run([_bin("ffn"), str(inp), "1024", "1280", "3420",
                    str(bdir), prefix, str(out)], cwd=CWD)
    inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return rc, err


def prove_vit_attn(block: int, workdir: Path) -> tuple[int, str]:
    prefix = f"vit-block-{block}"
    bdir   = workdir / f"vit-b{block}"
    inp    = bdir / f"{prefix}-tamper-attn-inp.bin"
    out    = bdir / f"{prefix}-tamper-attn-out.bin"
    _rand_input(inp, 1024, 1280)
    for tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (CWD / tmp).unlink(missing_ok=True)
    rc, err = _run([_bin("self-attn"), "linear", str(inp), "1024", "1280",
                    str(bdir), prefix, str(out), "1280"], cwd=CWD)
    inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return rc, err


def prove_patchmerger(weight_key: str, workdir: Path) -> tuple[int, str]:
    """运行 patch-merger，只验证 fc1 或 fc2（通过完整 binary 覆盖）。"""
    prefix = "patchmerger"
    n_patches, vit_dim, merged_dim, out_dim = 256, 1280, 5120, 2048
    inp     = workdir / f"{prefix}-tamper-input.bin"
    rms_inv = workdir / f"{prefix}-tamper-rms_inv.bin"
    out     = workdir / f"{prefix}-tamper-output.bin"
    rng = np.random.default_rng(42)
    X   = (rng.standard_normal((n_patches, vit_dim)) * SCALE).astype(np.int32)
    X.tofile(str(inp))
    X_f = X / SCALE
    inv = 1.0 / np.sqrt((X_f**2).mean(axis=1) + 1e-6)
    (inv * SCALE).round().astype(np.int32).tofile(str(rms_inv))
    rc, err = _run([_bin("patch-merger"),
                    str(inp), str(n_patches), str(vit_dim),
                    str(merged_dim), str(out_dim),
                    str(workdir), prefix, str(out), str(rms_inv)])
    for p in [inp, rms_inv, out]:
        p.unlink(missing_ok=True)
    return rc, err


def prove_llm_ffn(layer: int, workdir: Path) -> tuple[int, str]:
    prefix = f"layer-{layer}"
    inp = workdir / f"{prefix}-tamper-ffn-inp.bin"
    out = workdir / f"{prefix}-tamper-ffn-out.bin"
    _rand_input(inp, 512, 2048)
    rc, err = _run([_bin("ffn"), str(inp), "512", "2048", "11008",
                    str(workdir), prefix, str(out)])
    inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return rc, err


def prove_llm_attn(layer: int, workdir: Path) -> tuple[int, str]:
    prefix = f"layer-{layer}"
    inp = workdir / f"{prefix}-tamper-attn-inp.bin"
    out = workdir / f"{prefix}-tamper-attn-out.bin"
    _rand_input(inp, 512, 2048)
    for tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
        (CWD / tmp).unlink(missing_ok=True)
    rc, err = _run([_bin("self-attn"), "linear", str(inp), "512", "2048",
                    str(workdir), prefix, str(out), "256"])
    inp.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return rc, err


# ── 组件定义 ───────────────────────────────────────────────────────────────────
#
# 每个条目：
#   label       : 显示名称
#   weight_file : 原始权重文件路径（相对 workdir 或绝对）
#   proof_file  : IPA proof 文件路径
#   prove_fn    : 调用 prover 的函数 (workdir) -> (rc, err)
#   side        : "image" | "text"

def build_components(workdir: Path) -> list[dict]:
    components = []

    # ── Conv3d ────────────────────────────────────────────────────────────────
    components.append({
        "label":       "conv3d_embed",
        "side":        "image",
        "weight_file": workdir / "conv3d-conv3d_embed.weight-int.bin",
        "proof_file":  workdir / "conv3d-conv3d_embed-ipa-proof.bin",
        "prove_fn":    lambda wd=workdir: prove_conv3d(wd),
    })

    # ── ViT blocks（抽样 4 个）────────────────────────────────────────────────
    for block in [0, 7, 15, 31]:
        bdir = workdir / f"vit-b{block}"
        prefix = f"vit-block-{block}"
        for wkey, prove_fn in [
            ("mlp.gate_proj",    lambda b=block, wd=workdir: prove_vit_ffn(b, wd)),
            ("self_attn.q_proj", lambda b=block, wd=workdir: prove_vit_attn(b, wd)),
        ]:
            components.append({
                "label":       f"vit_b{block}_{wkey}",
                "side":        "image",
                "weight_file": bdir / f"{prefix}-{wkey}.weight-int.bin",
                "proof_file":  bdir / f"{prefix}-{wkey}-ipa-proof.bin",
                "prove_fn":    prove_fn,
            })

    # ── PatchMerger ───────────────────────────────────────────────────────────
    for wkey in ["patchmerger_fc1", "patchmerger_fc2"]:
        components.append({
            "label":       wkey,
            "side":        "image",
            "weight_file": workdir / f"patchmerger-{wkey}.weight-int.bin",
            "proof_file":  workdir / f"patchmerger-{wkey}-ipa-proof.bin",
            "prove_fn":    lambda k=wkey, wd=workdir: prove_patchmerger(k, wd),
        })

    # ── LLM decoder layers（抽样 6 层）────────────────────────────────────────
    for layer in [0, 7, 14, 21, 28, 35]:
        prefix = f"layer-{layer}"
        for wkey, prove_fn in [
            ("mlp.gate_proj",    lambda l=layer, wd=workdir: prove_llm_ffn(l, wd)),
            ("self_attn.q_proj", lambda l=layer, wd=workdir: prove_llm_attn(l, wd)),
        ]:
            components.append({
                "label":       f"llm_l{layer}_{wkey}",
                "side":        "text",
                "weight_file": workdir / f"{prefix}-{wkey}.weight-int.bin",
                "proof_file":  workdir / f"{prefix}-{wkey}-ipa-proof.bin",
                "prove_fn":    prove_fn,
            })

    return components


# ── 单次实验 ───────────────────────────────────────────────────────────────────

def run_one(comp: dict, ratio: float | None, rng: np.random.Generator) -> dict:
    weight_src  = Path(comp["weight_file"])
    proof_path  = str(comp["proof_file"])

    if not weight_src.exists():
        return {"error": f"weight not found: {weight_src}"}

    # 可选篡改
    tmp_weight = weight_src.parent / (weight_src.name + ".tamper_bak")
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
            "ratio": ratio, "prover_ok": False,
            "elapsed_ms": elapsed_ms,
            "stderr_tail": stderr[-200:],
        }

    ipa = _verify_ipa(proof_path)
    detected = (ratio is not None) and not ipa.get("binding_ok", True)

    return {
        "ratio":        ratio,
        "tamper_info":  tamper_info,
        "prover_ok":    True,
        "elapsed_ms":   elapsed_ms,
        "fold_ok":      ipa.get("fold_ok"),
        "binding_ok":   ipa.get("binding_ok"),
        "detected":     detected,
    }


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir",  default="zkllm-workdir/jina-v4")
    parser.add_argument("--repeats",  type=int,   default=3)
    parser.add_argument("--ratios",   nargs="+",  type=float, default=[0.001, 0.01, 0.05])
    parser.add_argument("--out",      default=str(OUT_FILE))
    args = parser.parse_args()

    workdir    = (ROOT / args.workdir).resolve()
    ratios     = args.ratios
    n_repeats  = args.repeats
    rng        = np.random.default_rng(42)
    components = build_components(workdir)

    print("=" * 60)
    print(f"C3b 全组件篡改检测率实验")
    print(f"  组件数  : {len(components)}")
    print(f"  篡改比例: {ratios}")
    print(f"  重复次数: {n_repeats}")
    print("=" * 60)

    all_results = []
    t_total = time.perf_counter()

    for comp in components:
        label = comp["label"]
        side  = comp["side"]
        print(f"\n[{label}]  ({side})")

        # 1. 正常权重验证
        r = run_one(comp, ratio=None, rng=rng)
        r.update({"label": label, "side": side, "tamper": False})
        all_results.append(r)
        status = "✓" if r.get("binding_ok") else "✗"
        print(f"  clean     {status}  ({r.get('elapsed_ms',0)}ms)")

        # 2. 各比例 × 重复
        for ratio in ratios:
            detected_count = 0
            for rep in range(n_repeats):
                r = run_one(comp, ratio=ratio, rng=rng)
                r.update({"label": label, "side": side, "tamper": True, "rep": rep})
                all_results.append(r)
                if r.get("detected"):
                    detected_count += 1
            rate = detected_count / n_repeats
            print(f"  ratio={ratio:<6}  detected={detected_count}/{n_repeats}  ({rate*100:.0f}%)")

    elapsed_total = round((time.perf_counter() - t_total))

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    clean_pass   = sum(1 for r in all_results if not r["tamper"] and r.get("binding_ok"))
    clean_total  = sum(1 for r in all_results if not r["tamper"])
    detect_total = sum(1 for r in all_results if r["tamper"] and r.get("detected"))
    tamper_total = sum(1 for r in all_results if r["tamper"])

    print("\n" + "=" * 60)
    print("汇总")
    print(f"  正常权重通过率: {clean_pass}/{clean_total}")
    print(f"  篡改总检出率:   {detect_total}/{tamper_total}  ({detect_total/max(tamper_total,1)*100:.1f}%)")

    # 按 side 分组
    for side in ["image", "text"]:
        det = sum(1 for r in all_results if r["tamper"] and r.get("side")==side and r.get("detected"))
        tot = sum(1 for r in all_results if r["tamper"] and r.get("side")==side)
        print(f"  {side} 检出率:  {det}/{tot}  ({det/max(tot,1)*100:.1f}%)")

    # 按比例分组
    for ratio in ratios:
        det = sum(1 for r in all_results if r["tamper"] and r.get("ratio")==ratio and r.get("detected"))
        tot = sum(1 for r in all_results if r["tamper"] and r.get("ratio")==ratio)
        print(f"  ratio={ratio}  {det}/{tot}  ({det/max(tot,1)*100:.1f}%)")

    print(f"\n总耗时: {elapsed_total}s")
    print("=" * 60)

    summary = {
        "experiment":    "C3b",
        "description":   "全组件 zkLLM 权重篡改检测率（IPA 承诺绑定，图像+文字侧）",
        "components":    [c["label"] for c in components],
        "tamper_ratios": ratios,
        "n_repeats":     n_repeats,
        "clean_pass":    clean_pass,
        "clean_total":   clean_total,
        "detect_total":  detect_total,
        "tamper_total":  tamper_total,
        "detection_rate": round(detect_total / max(tamper_total, 1), 4),
        "elapsed_s":     elapsed_total,
        "results":       all_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()

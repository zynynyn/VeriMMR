"""
zkLLM 权重篡改检测实验

对覆盖层（30–35）的 gate_proj 和 q_proj 权重，分别：
  1. 正确权重  → 运行 prover + 双重验证（fold + 承诺绑定）→ 期望全 PASS
  2. 篡改权重  → 替换式高斯噪声（σ=std(W), 比例 p）→ 期望 binding_ok=False

篡改方法（同 C3a/C3b）：
  sigma = std(W)，随机选取比例 p 的元素，替换为 N(0, sigma²) 采样值（四舍五入为整数）

用法：
  python script/tamper_experiment.py [--layers 30 31 ...] [--ratios 0.001 0.01 0.05]
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).parent.parent.resolve()
BIN_DIR = ROOT / "src" / "zkllm"
ZKLLM_CWD = ROOT / "src" / "zkllm"

EMBED_DIM    = 2048
HIDDEN_DIM   = 11008
KV_DIM       = 256
NUM_KV_HEADS = 2
SEQ_LEN      = 512

sys.path.insert(0, str(ROOT))
from script.verify_layers import verify_ipa


def _tamper_weight(src: Path, dst: Path, ratio: float, rng: np.random.Generator) -> dict:
    """读取 int32 权重文件，替换比例 ratio 的元素为高斯噪声，写到 dst。"""
    W = np.fromfile(str(src), dtype=np.int32)
    sigma = float(W.std())
    n_tamper = max(1, int(len(W) * ratio))
    idx = rng.choice(len(W), n_tamper, replace=False)
    noise = rng.normal(0, sigma, n_tamper).round().astype(np.int32)
    W_tampered = W.copy()
    W_tampered[idx] = noise
    W_tampered.tofile(str(dst))
    return {"n_total": len(W), "n_tampered": n_tamper, "sigma": round(sigma, 2)}


def _run_prover(binary: str, args: list, cwd: str) -> tuple[int, str]:
    r = subprocess.run([binary] + args, capture_output=True, cwd=cwd,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    return r.returncode, r.stderr.decode(errors="replace")


def _verify(proof_path: str) -> dict:
    try:
        return verify_ipa(proof_path)
    except Exception as e:
        return {"fold_ok": False, "binding_ok": False, "error": str(e)}


def run_one(layer: int, weight_key: str, ratio: float | None,
            workdir: Path, rng: np.random.Generator) -> dict:
    """
    针对一个层 + 权重 + 篡改比例，运行一次完整实验。
    ratio=None 表示使用原始权重。
    """
    prefix = f"layer-{layer}"
    is_ffn = weight_key.startswith("mlp.")

    # 权重文件路径
    if is_ffn:
        wname = weight_key  # mlp.gate_proj
        weight_src = workdir / f"{prefix}-{wname}.weight-int.bin"
        proof_path  = str(workdir / f"{prefix}-{wname}-ipa-proof.bin")
        out_bin     = workdir / f"{prefix}-tamper-ffn-out.bin"
        inp_path    = workdir / f"{prefix}-tamper-ffn-inp.bin"
    else:
        wname = weight_key  # self_attn.q_proj
        weight_src = workdir / f"{prefix}-{wname}.weight-int.bin"
        proof_path  = str(workdir / f"{prefix}-{wname}-ipa-proof.bin")
        out_bin     = workdir / f"{prefix}-tamper-attn-out.bin"
        inp_path    = workdir / f"{prefix}-tamper-attn-inp.bin"

    if not weight_src.exists():
        return {"error": f"weight file not found: {weight_src}"}

    # 生成输入激活
    if not inp_path.exists():
        (rng.standard_normal((SEQ_LEN, EMBED_DIM)) * (1 << 16)) \
            .astype(np.int32).tofile(str(inp_path))

    # 确定使用的权重文件
    if ratio is None:
        weight_path = weight_src
        tamper_info = None
    else:
        weight_path = workdir / f"{prefix}-{wname}.weight-int.tampered.bin"
        tamper_info = _tamper_weight(weight_src, weight_path, ratio, rng)

    # 临时替换权重文件（binary 从固定路径加载）
    weight_active = workdir / f"{prefix}-{wname}.weight-int.bin"
    if ratio is not None:
        shutil.copy(str(weight_src), str(workdir / f"{prefix}-{wname}.weight-int.orig.bin"))
        shutil.copy(str(weight_path), str(weight_active))

    try:
        t0 = time.perf_counter()
        if is_ffn:
            rc, stderr = _run_prover(
                str(BIN_DIR / "ffn"),
                [str(inp_path), str(SEQ_LEN), str(EMBED_DIM), str(HIDDEN_DIM),
                 str(workdir), prefix, str(out_bin)],
                str(ZKLLM_CWD))
        else:
            for tmp in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
                (ZKLLM_CWD / tmp).unlink(missing_ok=True)
            rc, stderr = _run_prover(
                str(BIN_DIR / "self-attn"),
                ["linear", str(inp_path), str(SEQ_LEN), str(EMBED_DIM),
                 str(workdir), prefix, str(out_bin), str(KV_DIM)],
                str(ZKLLM_CWD))
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
    finally:
        # 还原权重文件
        if ratio is not None:
            orig = workdir / f"{prefix}-{wname}.weight-int.orig.bin"
            if orig.exists():
                shutil.copy(str(orig), str(weight_active))
                orig.unlink()
            weight_path.unlink(missing_ok=True)
        for f in [out_bin, inp_path]:
            f.unlink(missing_ok=True)

    prover_ok = (rc == 0)
    if not prover_ok:
        return {"layer": layer, "weight": weight_key, "ratio": ratio,
                "prover_ok": False, "stderr_tail": stderr[-300:]}

    result = _verify(proof_path)
    return {
        "layer": layer,
        "weight": weight_key,
        "ratio": ratio,
        "tamper_info": tamper_info,
        "prover_ok": prover_ok,
        "elapsed_ms": elapsed_ms,
        "fold_ok": result.get("fold_ok"),
        "binding_ok": result.get("binding_ok"),
        "detected": not result.get("binding_ok", True),  # 篡改时期望 binding_ok=False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[30, 33, 35])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.001, 0.01, 0.05])
    parser.add_argument("--weights", nargs="+",
                        default=["mlp.gate_proj", "self_attn.q_proj"])
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    parser.add_argument("--out", default="notes/experiment_results/tamper_ipa_detection.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workdir = (ROOT / args.workdir).resolve()
    rng = np.random.default_rng(args.seed)

    print(f"zkLLM 权重篡改检测实验（IPA 承诺绑定）")
    print(f"层: {args.layers}  权重: {args.weights}  比例: {args.ratios}")
    print()

    all_results = []

    for layer in args.layers:
        for wkey in args.weights:
            # ── 1. 正确权重 ──────────────────────────────────────────────────
            print(f"[Layer {layer} / {wkey}] 正确权重 ...", end=" ", flush=True)
            r = run_one(layer, wkey, None, workdir, rng)
            fold  = r.get("fold_ok")
            bind  = r.get("binding_ok")
            tag   = "✓ PASS" if (fold and bind is not False) else "✗ FAIL"
            print(f"{tag}  fold={fold}  binding={bind}  ({r.get('elapsed_ms')}ms)")
            r["tamper"] = False
            all_results.append(r)

            # ── 2. 篡改权重 ──────────────────────────────────────────────────
            for ratio in args.ratios:
                print(f"[Layer {layer} / {wkey}] 篡改 {ratio*100:.1f}% ...", end=" ", flush=True)
                r = run_one(layer, wkey, ratio, workdir, rng)
                fold     = r.get("fold_ok")
                bind     = r.get("binding_ok")
                detected = r.get("detected", None)
                tag      = "✓ DETECTED" if detected else ("✗ MISSED" if detected is False else "? err")
                print(f"{tag}  fold={fold}  binding={bind}  ({r.get('elapsed_ms')}ms)")
                r["tamper"] = True
                all_results.append(r)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    correct_pass  = sum(1 for r in all_results if not r["tamper"] and
                        r.get("fold_ok") and r.get("binding_ok") is not False)
    correct_total = sum(1 for r in all_results if not r["tamper"])
    detected      = sum(1 for r in all_results if r["tamper"] and r.get("detected"))
    tamper_total  = sum(1 for r in all_results if r["tamper"])

    print()
    print("━" * 52)
    print(f"正确权重通过: {correct_pass}/{correct_total}")
    print(f"篡改检出率:   {detected}/{tamper_total}  ({detected/tamper_total*100:.0f}%)")
    print("━" * 52)

    summary = {
        "experiment": "tamper_ipa_detection",
        "description": "IPA 承诺绑定校验检测权重篡改",
        "layers": args.layers,
        "weights": args.weights,
        "ratios": args.ratios,
        "correct_pass": correct_pass,
        "correct_total": correct_total,
        "detected": detected,
        "tamper_total": tamper_total,
        "results": all_results,
    }

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()

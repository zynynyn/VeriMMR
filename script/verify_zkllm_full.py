"""
Phase 3 完整 zkLLM 证明流水线计时实验（全并行版）

覆盖组件（全部同时启动）：
  1. Conv3d patch embedding          → GPU1
  2. ViT 32 blocks（2-GPU 并行）     → GPU0 + GPU1
  3. PatchMerger                     → GPU1
  4. Pooling head                    → 纯 Python
  5. LLM 36 decoder 层（2-GPU 并行）→ GPU0 + GPU1

用法：
  cd /root/autodl-tmp/UltraRAG
  python script/verify_zkllm_full.py [--workdir zkllm-workdir/jina-v4]
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT     = Path(__file__).parent.parent.resolve()
SCRIPT   = ROOT / "script"
BIN_DIR  = ROOT / "src" / "zkllm"
OUT_FILE = ROOT / "notes" / "experiment_results" / "full_proof_timing.json"


def _run_component(label: str, func, *args, **kwargs):
    print(f"[{label}] 开始...", flush=True)
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = round((time.perf_counter() - t0) * 1000)
        ok = result.get("all_ok", result.get("ok", False)) if isinstance(result, dict) else bool(result)
        print(f"[{label}] {'✓ PASS' if ok else '✗ FAIL'}  ({elapsed}ms)", flush=True)
        return {"label": label, "ok": ok, "elapsed_ms": elapsed, "detail": result}
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"[{label}] ✗ ERROR: {e}  ({elapsed}ms)", flush=True)
        return {"label": label, "ok": False, "elapsed_ms": elapsed, "error": str(e)}


def _run_subprocess_component(label: str, cmd: list, timeout: int = 7200):
    print(f"[{label}] 开始...  cmd: {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=False, timeout=timeout)
    elapsed = round((time.perf_counter() - t0) * 1000)
    ok = (r.returncode == 0)
    print(f"[{label}] {'✓ PASS' if ok else '✗ FAIL'}  ({elapsed}ms)", flush=True)
    return {"label": label, "ok": ok, "elapsed_ms": elapsed, "returncode": r.returncode}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="zkllm-workdir/jina-v4")
    args = parser.parse_args()
    workdir = (ROOT / args.workdir).resolve()

    print("=" * 60)
    print("Phase 3 完整 zkLLM 证明流水线（全并行）")
    print(f"  workdir : {workdir}")
    print(f"  bindir  : {BIN_DIR}")
    print("=" * 60)

    sys.path.insert(0, str(ROOT))
    from script.prove_conv3d_embed import prove_conv3d_embed
    from script.prove_patchmerger  import prove_patchmerger
    from script.prove_pooling      import prove_pooling

    blocks = list(range(32))
    layers = list(range(36))

    task_order = [
        "Conv3d embed",
        "ViT 32 blocks (2-GPU)",
        "PatchMerger",
        "Pooling head",
        "LLM 36 layers (2-GPU)",
    ]
    task_fns = {
        "Conv3d embed": lambda: _run_component(
            "Conv3d embed", prove_conv3d_embed,
            frames_npy=None, workdir=workdir, gpu_id=1),

        "ViT 32 blocks (2-GPU)": lambda: _run_subprocess_component(
            "ViT 32 blocks (2-GPU)",
            [sys.executable, str(SCRIPT / "verify_vit.py"),
             "--blocks"] + [str(b) for b in blocks] + [
             "--workdir", str(args.workdir),
             "--out", str(ROOT / "notes/experiment_results/verify_vit.json"),
             "--parallel"]),

        "PatchMerger": lambda: _run_component(
            "PatchMerger", prove_patchmerger,
            n_patches=256, workdir=workdir, gpu_id=1),

        "Pooling head": lambda: _run_component(
            "Pooling head", prove_pooling,
            H_path=None, mask=None, seq_len=1024, embed_dim=2048),

        "LLM 36 layers (2-GPU)": lambda: _run_subprocess_component(
            "LLM 36 layers (2-GPU)",
            [sys.executable, str(SCRIPT / "verify_layers.py"),
             "--layers"] + [str(l) for l in layers] + [
             "--workdir", str(args.workdir),
             "--out", str(ROOT / "notes/experiment_results/verify_36layers.json"),
             "--parallel"]),
    }

    print("\n全部组件同时启动...\n" + "─" * 60, flush=True)
    t_total = time.perf_counter()
    results_map = {}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(task_fns[label]): label for label in task_order}
        for future in as_completed(futures):
            results_map[futures[future]] = future.result()

    total_ms = round((time.perf_counter() - t_total) * 1000)

    results = [results_map[label] for label in task_order]
    n_pass  = sum(1 for r in results if r["ok"])

    print("\n" + "=" * 60)
    print("Phase 3 完整流水线汇总（全并行）")
    print("=" * 60)
    fmt = "{:<28} {:>10}  {}"
    print(fmt.format("组件", "耗时(ms)", "状态"))
    print("─" * 52)
    for r in results:
        print(fmt.format(r["label"], r["elapsed_ms"], "✓" if r["ok"] else "✗"))
    print("─" * 52)
    print(fmt.format("总计（墙钟）", total_ms, f"{n_pass}/{len(results)} 通过"))
    print("=" * 60)

    summary = {
        "total_ms":   total_ms,
        "total_min":  round(total_ms / 60000, 2),
        "n_pass":     n_pass,
        "n_total":    len(results),
        "all_pass":   (n_pass == len(results)),
        "components": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        import numpy as _np
        def _ser(o):
            if isinstance(o, _np.bool_): return bool(o)
            if isinstance(o, (_np.integer,)): return int(o)
            if isinstance(o, (_np.floating,)): return float(o)
            raise TypeError(f"Not serializable: {type(o)}")
        json.dump(summary, f, indent=2, default=_ser)
    print(f"\n结果已保存: {OUT_FILE}")

    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()

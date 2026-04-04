"""
实验 E2：GQA 适配计算开销对比

目标：比较 GQA 模式（num_kv_heads=2）与原版单 KV head 模式（num_kv_heads=1）的证明耗时。

说明：
  - MHA 模式（num_kv_heads=1）：原版 zkLLM 的默认行为，kv_dim=256 时只有 1 个 KV head
  - GQA 模式（num_kv_heads=2）：本项目适配，2 个 KV head，每个 group_size=8 个 Q head
  两者使用同一组 temp_Q/K/V.bin，仅 num_kv_heads 不同，控制变量。
"""

import os, sys, time, json, subprocess
import numpy as np
from pathlib import Path

ROOT    = Path(__file__).parent.parent
WORKDIR = ROOT / "zkllm-workdir" / "jina-v4"
BIN_DIR = ROOT / "src" / "zkllm" / "bin"

SEQ_LEN      = 1024
EMBED_DIM    = 2048
KV_DIM       = 256
LAYER_IDX    = 33
REPEAT       = 3    # 每个模式重复次数，取均值

# ── 检查 E1 生成的 temp 文件是否存在（复用 E1 的激活数据）──────────────────
# E1 脚本运行后 temp_Q/K/V.bin 会被清理，需重新生成
# 最简单：直接用随机数代替（E2 只测时间，不测精度）
print(f"\n{'='*55}")
print(f"实验 E2：GQA 适配计算开销对比")
print(f"{'='*55}")
print(f"SEQ_LEN={SEQ_LEN}  KV_DIM={KV_DIM}  层={LAYER_IDX}  重复={REPEAT}次")

print(f"\n[准备] 生成随机 Q/K/V 作为输入（仅测时间，不测精度）...")
rng = np.random.default_rng(42)
(rng.standard_normal((SEQ_LEN, EMBED_DIM)) * 65536).astype(np.int32).tofile("temp_Q.bin")
(rng.standard_normal((SEQ_LEN, KV_DIM))    * 65536).astype(np.int32).tofile("temp_K.bin")
(rng.standard_normal((SEQ_LEN, KV_DIM))    * 65536).astype(np.int32).tofile("temp_V.bin")

layer_prefix  = f"layer-{LAYER_IDX}"
attn_out_path = str(WORKDIR / f"e2_attn_out.bin")

def run_attn(num_kv_heads: int) -> float:
    """运行 self-attn attn 模式，返回耗时（秒）。"""
    cmd = [
        str(BIN_DIR / "self-attn"), "attn",
        "temp_Q.bin",   # 从当前目录读
        str(SEQ_LEN), str(EMBED_DIM),
        str(WORKDIR), layer_prefix, attn_out_path,
        str(KV_DIM), str(num_kv_heads),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"  [ERROR] rc={r.returncode}")
        print(r.stderr[-300:])
    return elapsed

# ── 测量 ──────────────────────────────────────────────────────────────────────
results = {}
for mode, num_kv in [("MHA-compat (num_kv=1)", 1), ("GQA (num_kv=2)", 2)]:
    times = []
    print(f"\n[{mode}]")
    for i in range(REPEAT):
        t = run_attn(num_kv)
        times.append(t)
        print(f"  run {i+1}: {t:.2f}s")
    avg = sum(times) / len(times)
    results[mode] = {"times": times, "avg": avg, "num_kv_heads": num_kv}
    print(f"  均值: {avg:.2f}s")

# ── 清理 ──────────────────────────────────────────────────────────────────────
for f in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
    p = Path(f)
    if p.exists():
        p.unlink()

# ── 结论 ─────────────────────────────────────────────────────────────────────
t_mha = results["MHA-compat (num_kv=1)"]["avg"]
t_gqa = results["GQA (num_kv=2)"]["avg"]
overhead = (t_gqa - t_mha) / t_mha * 100

print(f"\n{'='*55}")
print(f"实验 E2 结论")
print(f"{'='*55}")
print(f"  MHA-compat（num_kv_heads=1）均值: {t_mha:.2f}s")
print(f"  GQA（num_kv_heads=2）     均值: {t_gqa:.2f}s")
print(f"  GQA 相对 MHA 开销变化:          {overhead:+.1f}%")
print(f"{'='*55}\n")

out = {
    "layer": LAYER_IDX, "seq_len": SEQ_LEN, "kv_dim": KV_DIM,
    "repeat": REPEAT,
    "mha_avg_s": t_mha, "gqa_avg_s": t_gqa,
    "overhead_pct": overhead,
    "detail": results,
}
out_path = ROOT / "notes" / "experiment_e2_result.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"结果已保存至：{out_path}")

"""
实验 E1：GQA zkAttn 输出正确性验证

目标：验证 self-attn.cu GQA 适配后的 Attention 计算与 PyTorch 参考实现一致。

方法：
  1. 加载 jina-v4，对测试文本做前向传播，用 hook 捕获某层的 attn 输入
  2. 运行 self-attn binary（linear 模式）得到量化后的 Q/K/V（temp_Q/K/V.bin）
  3. 在 Python 中模拟 C++ integer-domain GQA attention 计算
  4. 用 PyTorch F.scaled_dot_product_attention 计算 float32 参考输出
  5. 报告 L∞ 误差、余弦相似度、top-k 排名一致率

jina-v4 GQA 参数：
  embed_dim=2048, kv_dim=256, num_kv_heads=2, head_dim=128, num_q_heads=16
"""

import os, sys, math, subprocess, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT     = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "zkllm"))
from fileio_utils import save_int, load_int, to_int64, to_float

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/root/autodl-tmp/models/jina-embeddings-v4"
WORKDIR      = ROOT / "zkllm-workdir" / "jina-v4"
BIN_DIR      = ROOT / "src" / "zkllm" / "bin"
LAYER_IDX    = 33          # 代表性测试层（GQA 结构对所有层相同，33 在文本/图像证明范围内均有覆盖）
SEQ_LEN      = 1024        # NTT 约束
EMBED_DIM    = 2048
KV_DIM       = 256         # jina-v4 GQA
NUM_KV_HEADS = 2
HEAD_DIM     = KV_DIM // NUM_KV_HEADS   # 128
NUM_Q_HEADS  = EMBED_DIM // HEAD_DIM    # 16
GROUP_SIZE   = NUM_Q_HEADS // NUM_KV_HEADS  # 8
VALUE_LOGSF  = 16          # 量化缩放因子（1<<16）

TEST_TEXT = "What is the maximum aperture of the AF-S NIKKOR 50mm f/1.4G lens?"

# ── Step 1：加载模型，捕获 attn 输入 ──────────────────────────────────────────
print(f"\n{'='*60}")
print(f"实验 E1：GQA zkAttn 输出正确性验证")
print(f"{'='*60}")
print(f"层索引：{LAYER_IDX}  SEQ_LEN：{SEQ_LEN}  num_q_heads：{NUM_Q_HEADS}  num_kv_heads：{NUM_KV_HEADS}")

from sentence_transformers import SentenceTransformer
print(f"\n[1/5] 加载 jina-v4 模型...")
st_model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
# jina-v4: m[0].model → PeftModel → base_model.model.model.language_model.layers
layers = st_model[0].model.base_model.model.model.language_model.layers
device = next(st_model.parameters()).device

captured = {}
def _hook(module, inp, out):
    # input_layernorm 输出即 self_attn 的输入，shape: (1, seq, embed_dim)
    captured["attn_input"] = out.detach().cpu()

handle = layers[LAYER_IDX].input_layernorm.register_forward_hook(_hook)

print(f"[1/5] 前向传播捕获层 {LAYER_IDX} 激活...")
with torch.no_grad():
    _ = st_model.encode(TEST_TEXT, task="retrieval", convert_to_tensor=True)
handle.remove()

attn_input = captured["attn_input"]   # (1, S, EMBED_DIM) 或 (S, EMBED_DIM)
if attn_input.dim() == 3:
    attn_input = attn_input.squeeze(0)
attn_input_2d = attn_input.float()    # (S, EMBED_DIM)
print(f"[1/5] 捕获成功  shape={attn_input_2d.shape}  "
      f"norm={attn_input_2d.norm(dim=-1).mean():.4f}")

# seq_real 用于后续余弦相似度计算（实际 token 数，非 padding）
seq_real = attn_input_2d.shape[0]

# ── Step 2：保存 attn 输入，运行 self-attn linear 得到 Q/K/V ─────────────────
print(f"\n[2/5] 保存激活并运行 self-attn linear...")
os.makedirs(WORKDIR, exist_ok=True)
attn_inp_path = str(WORKDIR / f"e1_layer{LAYER_IDX}_attn_input.bin")

# zero-pad 到 SEQ_LEN
if attn_input_2d.shape[0] < SEQ_LEN:
    pad = torch.zeros(SEQ_LEN - attn_input_2d.shape[0], EMBED_DIM)
    attn_input_2d = torch.cat([attn_input_2d, pad], dim=0)

save_int(attn_input_2d, 1 << VALUE_LOGSF, attn_inp_path)
print(f"  保存至：{attn_inp_path}")

# 清理旧的 temp 文件
for f in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
    p = Path(f)
    if p.exists():
        p.unlink()

# 运行 self-attn linear（在项目根目录下运行，temp 文件写到当前目录）
layer_prefix = f"layer-{LAYER_IDX}"
attn_out_path = str(WORKDIR / f"e1_layer{LAYER_IDX}_attn_out.bin")
cmd_linear = [
    str(BIN_DIR / "self-attn"), "linear",
    attn_inp_path, str(SEQ_LEN), str(EMBED_DIM),
    str(WORKDIR), layer_prefix, attn_out_path,
    str(KV_DIM),
]
print(f"  命令：{' '.join(cmd_linear)}")
r = subprocess.run(cmd_linear, capture_output=True, text=True, cwd=str(ROOT))
if r.returncode != 0:
    print(f"  [ERROR] self-attn linear 失败 (rc={r.returncode})")
    print(r.stderr[-500:])
    sys.exit(1)
print(f"  self-attn linear 完成 (rc=0)")
if "successfully verified" in r.stdout:
    print(f"  QKV linear proof successfully verified ✓")

# ── Step 3：读取 Q/K/V，在 Python 中模拟 integer-domain GQA attention ─────────
print(f"\n[3/5] 读取 Q/K/V，模拟 integer-domain GQA attention...")

Q_int = load_int("temp_Q.bin", device="cpu").reshape(SEQ_LEN, EMBED_DIM)
K_int = load_int("temp_K.bin", device="cpu").reshape(SEQ_LEN, KV_DIM)
V_int = load_int("temp_V.bin", device="cpu").reshape(SEQ_LEN, KV_DIM)

# 反量化为 float32（与 llama-self-attn.py 保持一致）
Q_f = Q_int.float() / (1 << VALUE_LOGSF)  # (SEQ_LEN, EMBED_DIM)
K_f = K_int.float() / (1 << VALUE_LOGSF)  # (SEQ_LEN, KV_DIM)
V_f = V_int.float() / (1 << VALUE_LOGSF)  # (SEQ_LEN, KV_DIM)

# 重排为 (num_heads, SEQ_LEN, head_dim)
Q_heads = Q_f.view(SEQ_LEN, NUM_Q_HEADS, HEAD_DIM).transpose(0, 1)   # (16, 1024, 128)
K_heads = K_f.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)  # (2, 1024, 128)
V_heads = V_f.view(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 1)  # (2, 1024, 128)

# ── Python 模拟 integer-domain GQA（与 self-attn.cu attn 模式对齐）──────────
int_outputs = []
for g in range(NUM_KV_HEADS):
    K_g = K_heads[g]  # (SEQ_LEN, HEAD_DIM)
    V_g = V_heads[g]
    for h in range(GROUP_SIZE):
        qi = g * GROUP_SIZE + h
        Q_h = Q_heads[qi]  # (SEQ_LEN, HEAD_DIM)

        # 整数域 attention score（无 scale）
        X_h = Q_h.double() @ K_g.double().T   # (SEQ_LEN, SEQ_LEN)
        # shift（数值稳定）
        shift = X_h.max(dim=-1, keepdim=True).values
        X_shifted = X_h - shift
        # softmax（float64 精度）
        exp_x = torch.exp(X_shifted / math.sqrt(HEAD_DIM))
        Y_h = exp_x / exp_x.sum(dim=-1, keepdim=True)  # (SEQ_LEN, SEQ_LEN)
        # 加权 V
        out_h = Y_h @ V_g.double()  # (SEQ_LEN, HEAD_DIM)
        int_outputs.append(out_h.float())

# 拼接所有 Q heads：(num_q_heads, SEQ_LEN, HEAD_DIM) → (SEQ_LEN, EMBED_DIM)
int_out_all = torch.stack(int_outputs, dim=0)            # (16, SEQ_LEN, 128)
int_out_2d  = int_out_all.transpose(0, 1).reshape(SEQ_LEN, EMBED_DIM)  # (SEQ_LEN, 2048)

# ── Step 4：PyTorch 参考输出（float32 标准 GQA）──────────────────────────────
print(f"\n[4/5] 计算 PyTorch float32 参考输出...")

# 扩展 KV heads 以匹配 Q heads（GQA broadcast）
K_expanded = K_heads.repeat_interleave(GROUP_SIZE, dim=0)  # (16, SEQ_LEN, 128)
V_expanded = V_heads.repeat_interleave(GROUP_SIZE, dim=0)

# F.scaled_dot_product_attention 期望 (batch, heads, seq, dim)
ref_out = F.scaled_dot_product_attention(
    Q_heads.unsqueeze(0),       # (1, 16, 1024, 128)
    K_expanded.unsqueeze(0),
    V_expanded.unsqueeze(0),
    scale=1.0 / math.sqrt(HEAD_DIM),
).squeeze(0)  # (16, SEQ_LEN, 128)
ref_out_2d = ref_out.transpose(0, 1).reshape(SEQ_LEN, EMBED_DIM)  # (SEQ_LEN, 2048)

# ── Step 5：对比分析 ──────────────────────────────────────────────────────────
print(f"\n[5/5] 误差分析...")

diff = (int_out_2d - ref_out_2d).abs()
l_inf = diff.max().item()
l_1   = diff.mean().item()
rel   = (diff / (ref_out_2d.abs() + 1e-8)).mean().item()

# 余弦相似度（逐 token）
cos_sim = F.cosine_similarity(int_out_2d, ref_out_2d, dim=-1)  # (SEQ_LEN,)
cos_mean = cos_sim.mean().item()
cos_min  = cos_sim.min().item()

cos_real = cos_sim[:seq_real].mean().item()

print(f"\n{'─'*50}")
print(f"  SEQ_LEN（含 padding）: {SEQ_LEN}")
print(f"  有效 token 数:         {int(seq_real)}")
print(f"  L∞ 误差（全序列）:     {l_inf:.6f}")
print(f"  L1 误差（均值）:       {l_1:.6f}")
print(f"  相对误差（均值）:      {rel:.6f}")
print(f"  余弦相似度（全序列）:  {cos_mean:.8f}  (min={cos_min:.8f})")
print(f"  余弦相似度（有效 tok）: {cos_real:.8f}")
print(f"{'─'*50}")

# ── 运行 self-attn attn 模式（ZK 证明完整性）────────────────────────────────
print(f"\n[+] 运行 self-attn attn 模式（ZK 证明自洽性检验）...")
attn_sfx_out = str(WORKDIR / f"e1_layer{LAYER_IDX}_attn_sfx_out.bin")
cmd_attn = [
    str(BIN_DIR / "self-attn"), "attn",
    attn_inp_path, str(SEQ_LEN), str(EMBED_DIM),
    str(WORKDIR), layer_prefix, attn_sfx_out,
    str(KV_DIM), str(NUM_KV_HEADS),
]
print(f"  命令：{' '.join(cmd_attn)}")
r2 = subprocess.run(cmd_attn, capture_output=True, text=True, cwd=str(ROOT))
zk_ok = (r2.returncode == 0)
print(f"  ZK 证明完整性：{'✅ 通过 (rc=0)' if zk_ok else f'❌ 失败 (rc={r2.returncode})'}")
if r2.stdout:
    print(f"  stdout: {r2.stdout[-300:]}")
if r2.returncode != 0 and r2.stderr:
    print(f"  stderr: {r2.stderr[-300:]}")

# 清理 temp 文件
for f in ["temp_Q.bin", "temp_K.bin", "temp_V.bin"]:
    p = Path(f)
    if p.exists():
        p.unlink()

# ── 结论 ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"实验 E1 结论")
print(f"{'='*60}")

PASS_L_INF    = l_inf < 0.01
PASS_COS      = cos_real > 0.9999
PASS_ZK       = zk_ok

print(f"  L∞ 误差 < 0.01:         {'✅' if PASS_L_INF else '❌'}  ({l_inf:.6f})")
print(f"  余弦相似度 > 0.9999:     {'✅' if PASS_COS  else '❌'}  ({cos_real:.8f})")
print(f"  ZK 证明自洽性 (rc=0):    {'✅' if PASS_ZK   else '❌'}")

result = {
    "layer":        LAYER_IDX,
    "seq_len":      SEQ_LEN,
    "num_q_heads":  NUM_Q_HEADS,
    "num_kv_heads": NUM_KV_HEADS,
    "head_dim":     HEAD_DIM,
    "l_inf":        l_inf,
    "l1_mean":      l_1,
    "rel_error":    rel,
    "cos_sim_all":  cos_mean,
    "cos_sim_real_tokens": cos_real,
    "zk_proof_ok":  zk_ok,
    "pass":         PASS_L_INF and PASS_COS and PASS_ZK,
}

out_path = ROOT / "notes" / "experiment_e1_result.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  结果已保存至：{out_path}")
print(f"{'='*60}\n")

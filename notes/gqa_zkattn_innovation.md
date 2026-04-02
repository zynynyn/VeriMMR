# GQA zkAttn：面向 Grouped Query Attention 的零知识 Softmax 证明

*记录时间：2026-03-30*

---

## 一、背景与问题发现

### 原始 zkLLM 的局限

zkLLM（CCS 2024）提出了对大语言模型推理过程的零知识 Sumcheck 证明框架，覆盖：

- **FFN（前馈网络）**：SwiGLU 门控 + 两次矩阵乘 → Sumcheck 证明
- **Self-Attn linear**：Q/K/V 线性投影 → Sumcheck 证明
- **Self-Attn attn**：Softmax Attention → zkSoftmax 证明

其中 `self-attn attn` 模式在原论文中仅针对 **MHA（Multi-Head Attention）** 设计，即假设 `kv_dim == embed_dim`（所有头共享相同的 K/V 维度）。

### jina-v4 使用 GQA

jina-embeddings-v4（基于 Qwen2.5-3B backbone）采用 **GQA（Grouped Query Attention）**：

| 参数 | MHA（原论文） | GQA（jina-v4）|
|------|-------------|--------------|
| `embed_dim` | 4096 | 2048 |
| `kv_dim` | 4096（= embed_dim）| **256**（≠ embed_dim）|
| `num_q_heads` | N | **16** |
| `num_kv_heads` | N | **2** |
| `head_dim` | embed_dim/N | **128** |
| `group_size` | 1 | **8** |

当我们尝试直接运行 `self-attn attn` 模式时，报错：

```
transpose: incompatible dimensions
```

原因：原代码用 `d = Q.size / seq_len = 2048` 作为 K 的维度，
然后执行 `K.transpose(seq_len, 2048)`，但 K 实际大小为 `seq × 256`，
2048 ≠ 256，故转置失败。

---

## 二、GQA 的数学结构

### 2.1 标准 MHA 回顾

设输入 `X: (L, d_model)`，单头的注意力计算为：

```
Q_h = X @ W_Q  →  (L, head_dim)
K_h = X @ W_K  →  (L, head_dim)
V_h = X @ W_V  →  (L, head_dim)

X_h = Q_h @ K_h^T               # (L, L) 注意力分数
Y_h = softmax(X_h / √head_dim)  # (L, L) 注意力权重
out_h = Y_h @ V_h               # (L, head_dim) 头输出
```

MHA 中所有 N 个头各自拥有独立的 K/V 投影：
`kv_dim = N × head_dim = embed_dim`

### 2.2 GQA 核心思想

GQA 将 N 个 Q 头划分为 G 组，每组共用一对 K/V 头：

```
Q: (L, embed_dim)  =  (L, num_q_heads × head_dim)
K: (L, kv_dim)     =  (L, num_kv_heads × head_dim)   ← 远小于 embed_dim
V: (L, kv_dim)     =  (L, num_kv_heads × head_dim)

group_size = num_q_heads / num_kv_heads  # jina-v4: 16/2 = 8
```

对于 Q-head `h`，其对应的 KV-group `g = h // group_size`：

```
Q_h: (L, head_dim)
K_g: (L, head_dim)   # 被 group_size 个 Q-head 共享
V_g: (L, head_dim)

X_h = Q_h @ K_g^T              # (L, L)
Y_h = softmax(X_h)             # (L, L)
out_h = Y_h @ V_g              # (L, head_dim)
```

**关键维度关系**：

```
embed_dim = num_q_heads × head_dim  = 16 × 128 = 2048
kv_dim    = num_kv_heads × head_dim =  2 × 128 = 256
```

模型权重维度固定，与输入序列长度 L 无关。

### 2.3 维度与序列长度的区分

这是理解本文改造的关键：

| 类型 | 来源 | 是否随输入变化 |
|------|------|--------------|
| `d_model`, `kv_dim`, `head_dim` | 模型架构 | **固定** |
| 序列长度 `L` | 输入 token 数 | **随输入变化** |

零知识证明的 NTT 约束作用在**注意力矩阵** `X_h: (L, L)` 上，
其总大小 `L²` 必须满足 `L² % (1<<20) == 0`。
这与模型维度完全无关，仅与序列长度有关。

---

## 三、实现方案：per-head GQA 证明

### 3.1 头提取的技巧

原始 `temp_Q.bin` 的内存布局为 `(L, embed_dim)` 行优先：

```
row 0:  [q00, q01, ..., q0_{d-1}]   # token 0 的全部维度
row 1:  [q10, q11, ..., q1_{d-1}]
...
```

要提取第 h 个头（维度范围 `[h*head_dim, (h+1)*head_dim]`）而不引入新的 CUDA kernel，
使用**转置技巧**：

```
Q_full: (L, embed_dim)

Step 1: Q_T = Q_full.transpose(L, embed_dim)
        → (embed_dim, L)  列变行，行变列
        内存中第 h 个头的连续区间：[h*head_dim*L, (h+1)*head_dim*L)

Step 2: Q_h_flat = Q_T.trunc(h*head_dim*L, (h+1)*head_dim*L)
        → (head_dim * L,)  连续的内存区间

Step 3: Q_h = Q_h_flat.transpose(head_dim, L)
        → (L, head_dim)   还原为每行一个 token 的布局
```

同理处理 K、V（以 KV-group g 为粒度提取）。

### 3.2 Rescaling 缩放因子的修正

zkLLM 使用 `Rescaling(sf)` 处理定点数溢出，其内部创建大小为 `sf` 的 tLookup 表。
约束：**输入张量大小必须是 sf 的倍数**。

原代码使用 `Rescaling(1<<20)` 是为 MHA 设计的：
`out.size = L × embed_dim ≥ 1<<20`（对整个 attention 输出一次性证明）。

GQA 改造后，每次 rescale 的对象是**单个头的输出**：
`out_h.size = L × head_dim = 1024 × 128 = 131072 = 2^17`

`131072 % (1<<20) = 131072 % 1048576 ≠ 0` → 报错。

**修正**：动态计算 `rs_sf` 为能整除 `head_out_size` 的最大 2 的幂：

```cpp
uint head_out_size = seq_len * head_dim;   // 131072 = 2^17
uint rs_sf = 1;
{ uint tmp = head_out_size; while (tmp > 1) { tmp >>= 1; rs_sf <<= 1; } }
// rs_sf = head_out_size = 131072，table.size=131072 整除 out_h.size=131072 ✓
Rescaling rs1(rs_sf), rs2(rs_sf);
```

### 3.3 zkSoftmax 的 NTT 约束

zkSoftmax 构造参数 `bs = {1<<8, 1<<20, 1<<20}` 要求注意力矩阵大小整除 `1<<20`：

| 序列长度 L | 注意力矩阵大小 L² | 满足约束？ |
|-----------|----------------|----------|
| 1024 | 1,048,576 = **2^20** | **✓** |
| 512 | 262,144 = 2^18 | ✗ |
| 256 | 65,536 = 2^16 | ✗ |

因此，无论是语料库侧还是查询侧，都需要将序列填充到 **L = 1024**。

---

## 四、序列长度的统一填充策略

### 4.1 语料库侧（图像）

jina-v4 对图像编码时产生约 641 个 visual tokens（ViT patch tokens）：

```
图像 → jina-v4 ViT → ~641 tokens → 零填充到 SEQ_LEN_IMG = 1024
```

1024² = 2^20 满足所有约束，FFN + linear + zkAttn 全部可证。

### 4.2 查询侧（文本）

文本查询通常仅有 10-100 个 tokens，原始实现填充到 256：

```
文本 → tokenizer → ~N tokens (N<<100) → 零填充到 SEQ_LEN_PAD = 256
```

这满足 FFN 和 linear 的约束（256×2048=2^19），但 256² = 2^16 不满足 zkAttn 约束。

**修正后**（本工作）：

```
文本 → tokenizer → ~N tokens → 零填充到 SEQ_LEN_PAD = 1024
```

与语料库侧完全对称。零填充的有效性：
- 前 N 行承载真实语义，有完整的 ZK 证明
- 后 1024-N 行全零，对应注意力权重为零，证明平凡成立
- 两侧使用相同的模型权重承诺，可跨侧验证

### 4.3 为什么两侧可以用相同的证明框架

本质上，GQA zkAttn 证明的是：

> 给定量化权重 W（已通过 KZG 承诺），对于输入激活 X（序列长度 L，零填充至 1024），
> 模型确实按照正确的数学公式计算了 Attention 输出。

"语料库/查询侧" 的区分是系统层面的，对证明系统透明。
两侧的模型权重承诺文件（`*-commitment.bin`）是共用的，
只有输入激活（`*-attn-input.bin` / `temp_Q/K/V.bin`）不同。

---

## 五、相关工作调研

截至 2026 年初，未发现已发表的工作将 GQA 纳入零知识证明框架：

| 工作 | 发表时间 | 是否涉及 GQA |
|------|---------|-------------|
| zkLLM (CCS 2024) | 2024.11 | 仅 MHA |
| zkGPT | 2024 | 仅 MHA |
| zkLoRA | 2024 | 仅验证 LoRA 参数，无 Attn 证明 |
| zkPyTorch | 2024 | 通用框架，未针对 GQA |
| ZK-DeepSeek | 2025.01 | MLA 结构（非 GQA）|
| Jolt Atlas | 2025 | RISC-V zkVM，无 Transformer 专项优化 |

**本工作的创新点**：首次实现面向 GQA 架构的 zkAttn Softmax 证明，
关键技术贡献：
1. **转置技巧**：无需新 CUDA kernel 的 per-head 张量提取
2. **动态 Rescaling**：适配 per-head 输出大小的缩放因子计算
3. **统一填充策略**：查询侧与语料库侧对称的 L=1024 零填充

---

## 六、最终实现文件

| 文件 | 修改内容 |
|------|---------|
| `zkllm-ccs2024/self-attn.cu` | 重写 `attn` 模式，支持 GQA per-head 循环 + 动态 Rescaling |
| `src/zkllm/bin/self-attn` | 重新编译的二进制 |
| `script/build_corpus_zkllm_proofs.py` | 每层加入 `self-attn attn` 调用（kv_dim=256, num_kv_heads=2）|
| `script/interactive_demo.py` | SEQ_LEN_PAD 改为 1024；`_run_zkllm_query_bg` 加入 attn 证明 |

### 运行示例（单层测试）

```bash
cd /root/autodl-tmp/zkllm-ccs2024
WD=./zkllm-workdir/jina-v4

# Step 1: 线性投影证明（生成 temp_Q/K/V.bin）
./self-attn linear input.bin 1024 2048 $WD layer-35 /tmp/out.bin 256

# Step 2: GQA Softmax 证明（读取 temp_Q/K/V.bin）
./self-attn attn input.bin 1024 2048 $WD layer-35 /tmp/out.bin 256 2
# 输出：GQA zkAttn proof complete. (16 Q-heads, 2 KV-heads, head_dim=128)
```

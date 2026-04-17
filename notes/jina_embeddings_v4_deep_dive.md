# jina-embeddings-v4 深度理解

> 本文从零开始，结合源代码与模型架构，逐层讲清楚 jina-embeddings-v4 是如何把一张图像或一段文本变成一个向量的。涉及公式的地方使用标准 LaTeX 语法。

---

## 目录

1. [为什么要理解这个模型](#1-为什么要理解这个模型)
2. [从"Embedding"的本质说起](#2-从embedding的本质说起)
3. [整体架构鸟瞰](#3-整体架构鸟瞰)
4. [图像路径：视觉编码器（ViT）](#4-图像路径视觉编码器vit)
5. [语言主干：Qwen2.5-3B Transformer](#5-语言主干qwen25-3b-transformer)
6. [任务适配：LoRA 多适配器](#6-任务适配lora-多适配器)
7. [输出头：从隐藏状态到 Embedding](#7-输出头从隐藏状态到-embedding)
8. [两种 Embedding 的区别](#8-两种-embedding-的区别)
9. [Matryoshka 表示学习](#9-matryoshka-表示学习)
10. [数据流全链路汇总](#10-数据流全链路汇总)
11. [对 Phase 3 可验证证明的启示](#11-对-phase-3-可验证证明的启示)

---

## 1. 为什么要理解这个模型

在本毕业设计系统中，jina-embeddings-v4 承担**图像检索的核心角色**：

- **离线建库**：把 PDF 每页图像 $I_i$ 编码为向量 $\mathbf{v}_i \in \mathbb{R}^{2048}$，存入 FAISS 索引
- **在线查询**：把用户文本 $q$ 编码为向量 $\mathbf{q} \in \mathbb{R}^{2048}$，用内积相似度检索最近邻

Phase 3 的目标是：**对"$\mathbf{v}_i$ 确实是由 jina-v4 从 $I_i$ 正确计算出来的"这件事生成 ZK 证明**。要设计这个证明，必须先弄清楚计算路径上每一步做了什么。

---

## 2. 从"Embedding"的本质说起

### 2.1 什么是 Embedding

Embedding（嵌入）是把任意对象（图像、文本）映射到一个固定维度的实数向量空间，使得**语义相似的对象在向量空间中距离相近**。

$$f: \mathcal{X} \rightarrow \mathbb{R}^d$$

其中 $\mathcal{X}$ 是输入空间（图像集合或文本集合），$d$ 是 embedding 维度（此处 $d = 2048$）。

### 2.2 相似度度量

jina-v4 使用**内积相似度**（Inner Product，与 FAISS IndexFlatIP 一致）：

$$\text{sim}(\mathbf{q}, \mathbf{v}) = \mathbf{q}^\top \mathbf{v} = \sum_{j=1}^{d} q_j v_j$$

由于输出向量经过 L2 归一化（$\|\mathbf{e}\|_2 = 1$），内积与余弦相似度等价：

$$\mathbf{q}^\top \mathbf{v} = \|\mathbf{q}\|_2 \cdot \|\mathbf{v}\|_2 \cdot \cos\theta = \cos\theta \quad \text{（若均已归一化）}$$

---

## 3. 整体架构鸟瞰

jina-embeddings-v4 是一个**视觉-语言统一 Embedding 模型**，基于 Qwen2.5-VL 架构，通过 LoRA 适配多个任务。

```
输入图像 I                         输入文本 q
    │                                  │
    ▼                                  ▼
[ViT 视觉编码器]                   [Tokenizer]
 32 blocks, hidden=1280           词表 151936
 patch 14×14, merge 2×2                │
    │                                  │
    ▼                                  ▼
visual tokens (T_v × 2048) ──┐   text tokens (T_t × 2048)
                              │          │
                              ▼          ▼
                    ┌─────────────────────────┐
                    │  Qwen2.5-3B LM 主干      │
                    │  36 层 Transformer       │
                    │  hidden_size = 2048       │
                    │  + LoRA 任务适配器        │
                    └─────────────────────────┘
                                  │
                    hidden_states[-1]: (T, 2048)
                          │               │
                          ▼               ▼
              [Single-vector head]  [Multi-vector head]
              MeanPool + L2Norm     Proj(2048→128) + L2Norm
                    │                     │
                    ▼                     ▼
             e ∈ ℝ²⁰⁴⁸             E ∈ ℝ^{T×128}
```

**参数规模汇总**：

| 组件 | 参数量 | 说明 |
|------|:------:|------|
| ViT 视觉编码器 | 669M | 32个 ViT block |
| LM 主干（Qwen2.5-3B base） | 3,755M | 36层 Transformer |
| LoRA 适配器 | 180M | rank=32，3个任务 |
| multi_vector_projector | 0.47M | 2048→128 线性层 |
| **合计** | **~3,935M** | **约 3.9B** |

---

## 4. 图像路径：视觉编码器（ViT）

### 4.1 为什么需要 ViT

语言模型只能处理离散的 token 序列，无法直接接受像素矩阵。ViT（Vision Transformer）的作用是把图像转化为**与文本 token 兼容的连续向量序列**，让语言模型可以统一处理。

### 4.2 图像分块（Patch Embedding）

ViT 首先把图像切成小块。jina-v4 使用 3D 卷积处理（以兼容视频），对于静态图像：

**配置参数**：
- `spatial_patch_size = 14`：每个 patch 是 $14 \times 14$ 像素
- `temporal_patch_size = 2`：时间维度，对静态图像相当于把同一帧重复 2 次后折叠
> **这里有一个时间维度是因为用的是3D卷积，要求必须能被2整除，如果是1，则时间上不能滑动**
<br>**2D 卷积**的卷积核是 `(H, W)`，在空间上滑动，捕捉空间特征。
**3D 卷积**的卷积核是 `(T, H, W)`，多了时间维度，同时在时间和空间上滑动，能捕捉**时空联合特征**。
<br>**直觉**
<br>2D 卷积看的是"一张图里的局部区域"，3D 卷积看的是"连续几帧里同一位置的变化"。
<br>比如检测"挥手"动作：
<br>- 2D 卷积只能看单帧，看不出运动
<br>- 3D 卷积同时看多帧，能感知手的移动轨迹
<br>**代价**
<br>参数量更大，计算量是 2D 的 T 倍，显存消耗也更高。所以实际使用时 T 一般取较小的值（比如 2 或 4），不会太大。
- `in_channels = 3`：RGB 三通道

**Conv3d patch embedding**：

$$\text{patch\_embed}(\mathbf{X}) = \text{Conv3d}(\mathbf{X},\ \text{kernel}=(2, 14, 14),\ \text{stride}=(2, 14, 14))$$

输入张量 $\mathbf{X} \in \mathbb{R}^{3 \times 2 \times H \times W}$（通道 × 时间 × 高 × 宽），卷积核大小与步长均为 $(2, 14, 14)$，输出每个 patch 的特征维度为 $1280$（ViT hidden_size）。

对于一张 $H \times W$ 的图像，patch 数量为：

$$N_\text{patch} = \frac{H}{14} \times \frac{W}{14}$$

### 4.3 动态分辨率（Dynamic Tiling）

不同于固定分辨率，jina-v4 支持任意分辨率图像。处理器会把图像按 $14 \times 14$ 对齐后裁剪/缩放，保留原始宽高比。实际 patch 数量用 `image_grid_thw`（time, height-patches, width-patches）记录：

```python
offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
# offsets[i] = 图像 i 的 patch 数量（H_patches × W_patches）
```

### 4.4 PatchMerger（Spatial Merge + MLP 投影）

32 个 ViT block 跑完后，输出 patch 序列经过 `PatchMerger` 模块压缩并投影到 LM 维度。PatchMerger 内部依次完成两件事：

1. **Spatial Merge（空间合并）**：将 ViT 输出中空间上相邻的 $2 \times 2 = 4$ 个 patch，通过 `.view(-1, 5120)` 拼成一个 5120 维向量，token 数量缩减为原来的 $\frac{1}{4}$
2. **MLP 投影**：用两层 MLP 将 5120 维映射到 LM 所需的 2048 维

合并后 token 数量：$T_v = \frac{N_\text{patch}}{4} = \frac{H \times W}{14^2 \times 4} = \frac{H \times W}{784}$

**PatchMerger 完整结构**（`Qwen2_5_VLPatchMerger`，代码实测）：

```
输入: [N_patch, 1280]
  → RMSNorm(1280)
  → .view(-1, 5120)          ← Spatial Merge：4 patch 拼接
  → Linear(5120, 5120) + GELU
  → Linear(5120, 2048)
输出: [N_patch/4, 2048]      ← 对齐 LM 输入维度
```

**注意**：PatchMerger 的激活函数是 **GELU**，不是 SwiGLU。这是整个 jina-v4 架构中唯一使用 GELU 的地方；ViT Block 的 FFN 和 LM 的 FFN 均使用 SwiGLU。

**例**：一张 $448 \times 448$ 的图像：
$$N_\text{patch} = \frac{448}{14} \times \frac{448}{14} = 32 \times 32 = 1024, \quad T_v = \frac{1024}{4} = 256 \text{ tokens}$$

### 4.5 ViT Block 结构

ViT 由 32 个相同的 Block 组成，每个 Block 包含：

$$\mathbf{h}' = \mathbf{h} + \text{SelfAttention}(\text{RMSNorm}(\mathbf{h}))$$
$$\mathbf{h}'' = \mathbf{h}' + \text{MLP}(\text{RMSNorm}(\mathbf{h}'))$$

其中 MLP 采用 **SwiGLU**（中间维度 $3420$）：

$$\text{SwiGLU}(\mathbf{x}) = (\text{SiLU}(\mathbf{W}_\text{gate}\,\mathbf{x})) \odot (\mathbf{W}_\text{up}\,\mathbf{x})$$
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**注意力机制**：大部分 Block 使用 **Window Attention**（局部窗口，`window_size = 112`），仅在第 8, 16, 24, 32 层（索引 7, 15, 23, 31）使用 **Full Attention**（全局注意力）。这大幅减少了计算量。
> **整体流程**
<br>每个 Block 做两件事，残差连接：
```
输入 h
  ↓
RMSNorm → SelfAttention → 加回 h → h'
  ↓
RMSNorm → MLP(SwiGLU) → 加回 h' → h''
  ↓
输出 h''
```
> **残差连接**（加回原始输入）的作用是防止梯度消失，让信息可以直接跳过这一层传递。

---

## RMSNorm

归一化层，把输入缩放到合适的数值范围，防止训练不稳定：

```
RMSNorm(x) = x / sqrt(mean(x²)) * γ
```

比 LayerNorm 简单，少了减均值的步骤，速度更快。

---

## SelfAttention

每个 patch 的 token 和其他 token 互相计算相关性，让模型知道"哪些位置应该关注哪些位置"。

计算流程：
```
Q = x @ Wq    ← 查询
K = x @ Wk    ← 键
V = x @ Wv    ← 值

attention = softmax(Q @ Kᵀ / √d) @ V
```

计算量是 token 数量的平方，所以 token 多时很贵。

---

## Window Attention vs Full Attention

**Full Attention**：每个 token 和所有 token 计算注意力。

```
token数=256，计算量 ∝ 256² = 65536
```

**Window Attention**：只和局部窗口内的 token 计算注意力，窗口大小 `112×112` 像素 = `8×8=64` 个 patch。

```
token数=64，计算量 ∝ 64² = 4096   ← 缩小16倍
```

32 层里只有 4 层用 Full Attention，其余 28 层用 Window Attention，整体计算量大幅下降。Full Attention 层负责融合全局信息，Window Attention 层负责提取局部特征。

---

## SwiGLU

MLP 里的激活函数，比普通 ReLU 效果更好：

```
输入 x (1280维)
  ↓
两路并行：
  gate = Wgate @ x → SiLU激活     (1280→3420)
  up   = Wup   @ x → 不激活       (1280→3420)
  ↓
逐元素相乘：gate ⊙ up             (3420维)
  ↓
Wdown @ 结果                      (3420→1280)
  ↓
输出 (1280维)
```

**SiLU** 是带门控的激活函数：
```
SiLU(x) = x × sigmoid(x)
```
输出在 x 较大时接近 x，x 较小时接近 0，比 ReLU 更平滑。

**门控机制**的直觉：gate 路决定"哪些特征重要"，up 路提供特征值，两者相乘相当于动态过滤，让模型自适应地选择信息。

ViT 的输出（`out_hidden_size = 2048`）通过 MLP merger 投影到语言模型的隐藏维度。

---

## 5. 语言主干：Qwen2.5-3B Transformer

### 5.1 输入序列的拼接

语言模型接收的是**混合 token 序列**。以图像为例，序列结构为：

```
<|im_start|> user \n
<|vision_start|> [visual_token_1] ... [visual_token_{T_v}] <|vision_end|>
Describe the image.
<|im_end|> \n
```

视觉 token（来自 ViT 输出）和文本 token（来自词表 embedding）被拼接成统一序列，维度均为 $d = 2048$。

### 5.2 RoPE（旋转位置编码）

Qwen2.5-VL 使用 3D RoPE（Rotary Position Embedding），对视觉 token 和文本 token 分别编码位置：
* 文本 token：1D 位置编码（序列位置 $t$）
* 视觉 token：3D 位置编码（时间 $t$，行 $y$，列 $x$）

#### 为什么需要位置编码

Transformer 的注意力计算本质是每个 token 的向量之间做点积，点积只看向量的内容，不感知顺序。把所有 token 打乱顺序，结果完全一样。但语言和图像都有位置信息（"猫追狗"和"狗追猫"完全不同），所以必须把位置信息注入模型。

传统做法是给每个位置学一个额外的 embedding 向量加到输入上，但这需要额外参数，且对训练时没见过的长度泛化性差。RoPE 的做法是不添加额外参数，而是在计算注意力时，直接把位置信息旋转进 Q 和 K 向量里。
                                          
#### 复数旋转是什么

先从二维说起。平面上一个向量 $(x, y)$，可以写成复数 $z = x + iy$。

把这个复数乘以 $e^{i\theta}$：

$$z' = z \cdot e^{i\theta} = (x + iy)(\cos\theta + i\sin\theta)$$

展开：

$$z' = (x\cos\theta - y\sin\theta) + i(x\sin\theta + y\cos\theta)$$

这正好是把向量 $(x, y)$ 在平面上**旋转 $\theta$ 角**得到的新向量。

所以"乘以 $e^{i\theta}$"在几何上就是旋转，旋转角度是 $\theta$。

对于高维向量（比如 128 维），把它拆成 64 对二维向量，每对独立旋转：

$$\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}, \begin{bmatrix} x_3 \\ x_4 \end{bmatrix}, \cdots, \begin{bmatrix} x_{127} \\ x_{128} \end{bmatrix}$$

每一对用各自的旋转角度 $\theta_j$ 旋转，合起来就是对整个向量做了 RoPE 变换。

#### 旋转频率是什么

不同的维度对用不同的旋转角度，位置 $m$ 处第 $j$ 对的旋转角度是：

$$m\theta_j, \quad \theta_j = 10000^{-2j/d}$$

$\theta_j$ 就是**旋转频率**，控制这一对维度旋转得快还是慢：

- $j$ 小（靠前的维度对）：$\theta_j$ 大，旋转快，对相邻位置敏感
- $j$ 大（靠后的维度对）：$\theta_j$ 小，旋转慢，对长距离位置敏感

类比时钟：秒针转得快（短程位置），分针转得慢（长程位置），时针更慢（超长程位置）。不同频率的维度组合，让模型同时感知短程和长程的位置关系。

#### 为什么内积只依赖相对位置

位置 $m$ 处的 Q 向量，经过 RoPE 变换后第 $j$ 对变成：

$$q_j e^{im\theta_j}$$

位置 $n$ 处的 K 向量，经过 RoPE 变换后第 $j$ 对变成：

$$k_j e^{in\theta_j}$$

两者的内积（点积）：

$$q_j e^{im\theta_j} \cdot k_j e^{-in\theta_j} = q_j k_j e^{i(m-n)\theta_j}$$

结果只含 $m - n$，不含 $m$ 和 $n$ 各自的值。

这意味着注意力分数只取决于两个 token 的**相对距离**，不取决于它们的绝对位置。把整个序列向右平移 100 位，任意两个 token 之间的注意力分数完全不变，模型具有**平移不变性**。

#### 3D RoPE 怎么扩展到视觉

对视觉 token，位置有三个维度 $(t, y, x)$，把 $d$ 维的旋转平均分成三份：

$$\theta_{\text{3D}} = [\underbrace{\theta_t}_{\text{时间，占}d/3\text{维}},\; \underbrace{\theta_y}_{\text{行，占}d/3\text{维}},\; \underbrace{\theta_x}_{\text{列，占}d/3\text{维}}]$$

每个维度组独立旋转，最后拼接。文本 token 只用一维位置 $t$，视觉 token 用三维位置，两者在同一个序列里共存，互不干扰。

---

### 5.3 Transformer Block 结构

36 个 Transformer Block，每层包含：

#### 1. GQA 自注意力（Grouped Query Attention）

$$Q = h W_Q \in \mathbb{R}^{T \times d}, \quad K = h W_K \in \mathbb{R}^{T \times d_{kv}}, \quad V = h W_V \in \mathbb{R}^{T \times d_{kv}}$$

##### Self-Attention 的完整过程

每个 token 的向量 $h$（2048 维）先通过三个线性层投影成 Q、K、V：
```
Q：我想找什么样的信息
K：我能提供什么样的信息
V：我实际提供的内容是什么
```

然后计算注意力分数，表示"我应该关注谁"：

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V$$

逐步拆开：

**第一步**，$QK^\top$ 计算每对 token 之间的相似度，结果是一个 $T \times T$ 的矩阵，第 $(i, j)$ 个元素表示第 $i$ 个 token 对第 $j$ 个 token 的关注程度（原始分数）。

**第二步**，除以 $\sqrt{d_h}$ 缩放。$d_h = 128$，点积结果的数量级约为 $\sqrt{d_h}$，不缩放的话数值过大，softmax 会把几乎所有权重集中在一个位置，梯度接近 0，训练无法进行。

**第三步**，softmax 把每行归一化成概率分布（和为 1），得到注意力权重矩阵。

**第四步**，乘以 $V$，对所有 token 的 Value 加权求和，得到每个 token 融合了其他 token 信息后的新表示。

##### 为什么要多头

单头注意力只能学一种"关注模式"。多头把 $d=2048$ 维拆成 16 份，每份 128 维，各自独立计算注意力，最后拼接：
```
头1 (128维)：关注语法依存关系
头2 (128维)：关注语义相似性
头3 (128维)：关注空间位置邻近
...
头16(128维)：关注其他模式
```

拼接后再经过线性层投影回 2048 维，综合了 16 种不同的关注视角。

##### GQA 是什么

普通多头注意力（MHA）：16 个 Q 头对应 16 个 K/V 头，每个头各自有 K/V 参数。

**问题**：推理时需要缓存所有头的 K/V（叫 KV Cache），16 个头的缓存量很大，速度慢。

GQA 的解决方案：Q 头保持 16 个，K/V 头减少到 2 个，每 8 个 Q 头共享 1 个 K/V 头：
```
Q头1  ─┐
Q头2  ─┤
Q头3  ─┤
Q头4  ─┼─→ K/V头1（共享）
Q头5  ─┤
Q头6  ─┤
Q头7  ─┤
Q头8  ─┘

Q头9  ─┐
Q头10 ─┤
Q头11 ─┤
Q头12 ─┼─→ K/V头2（共享）
Q头13 ─┤
Q头14 ─┤
Q头15 ─┤
Q头16 ─┘
```

K/V 参数量和 KV Cache 减少 8 倍，推理速度大幅提升，实验表明效果损失极小。

配置：
* `num_attention_heads = 16`（Query 头数）
* `num_key_value_heads = 2`（Key/Value 头数）
* 每个 Query 头维度：$d_h = 2048 / 16 = 128$
* GQA 比例 8:1

#### 2. SwiGLU 前馈网络（FFN）

$$\text{FFN}(h) = W_{\text{down}}(\text{SiLU}(W_{\text{gate}}\, h) \odot (W_{\text{up}}\, h))$$

其中 `intermediate_size = 11008`，即 $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{11008 \times 2048}$，$W_{\text{down}} \in \mathbb{R}^{2048 \times 11008}$。

注意力层负责 token 之间的**交互**（信息从其他位置汇聚进来），FFN 负责对每个 token **独立地**做特征变换，可以理解为：
```
Attention：我从别的 token 那里收集了什么信息
FFN：我拿到这些信息后，怎么进一步加工和提炼
```

维度变化完整过程：
```
输入 h：(T, 2048)
    ↓ W_gate 和 W_up 各自线性投影（并行）
gate 路：(T, 11008) → SiLU → (T, 11008)
up   路：(T, 11008)
    ↓ 逐元素相乘 ⊙
中间：(T, 11008)
    ↓ W_down 线性投影
输出：(T, 2048)
```

先升维到 11008（约为 2048 的 5.4 倍）再降回来，中间的高维空间给模型足够大的"操作空间"来提取和变换特征，11008 这个数字本身是超参数，实践中通常取 hidden_size 的 $\frac{8}{3}$ 倍左右再取整到合适的值。

#### 3. RMSNorm

$$\text{RMSNorm}(h) = \frac{h}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} h_j^2 + \epsilon}} \odot \gamma$$

其中 $\gamma \in \mathbb{R}^d$ 是可学习的缩放参数，$\epsilon = 10^{-6}$ 防止分母为零。

经过矩阵乘法和激活函数，向量的数值范围可能变得很大或很小，导致训练不稳定。RMSNorm 用向量自身各维度的均方根做分母，把向量缩放到数值稳定的范围：
```
归一化前 h：[1000, -500, 3000, ...]   数值混乱
归一化后  ：[0.31, -0.15, 0.93, ...]  统一量级
```

$\gamma$ 是可学习的，让模型在归一化后自适应地恢复每个维度的合理 scale，而不是强制缩成固定大小。

比 LayerNorm 少了减均值的步骤，计算更简单，实验表明效果相当。

#### 完整的 Block 计算

$$h' = h + \text{Attn}(\text{RMSNorm}(h))$$

$$h'' = h' + \text{FFN}(\text{RMSNorm}(h'))$$

**先归一化再计算**：每次进入 Attention 或 FFN 之前先 RMSNorm，保证输入数值稳定。

**残差连接**（加回原始输入）：梯度可以通过加法直接反传到前面的层，防止 36 层之后梯度消失。同时信息也有一条"高速公路"直接跳过当前层传递下去，即使某一层学歪了也不会完全破坏之前积累的信息。

### 5.4 每层的参数量

以单个 Transformer Layer 为例：

| 子模块 | 矩阵 | 形状 | 参数量 |
|--------|------|------|:------:|
| Q proj | $\mathbf{W}_Q$ | $2048 \times 2048$ | 4.2M |
| K proj | $\mathbf{W}_K$ | $2048 \times 256$ | 0.5M |
| V proj | $\mathbf{W}_V$ | $2048 \times 256$ | 0.5M |
| O proj | $\mathbf{W}_O$ | $2048 \times 2048$ | 4.2M |
| gate proj | $\mathbf{W}_\text{gate}$ | $11008 \times 2048$ | 22.5M |
| up proj | $\mathbf{W}_\text{up}$ | $11008 \times 2048$ | 22.5M |
| down proj | $\mathbf{W}_\text{down}$ | $2048 \times 11008$ | 22.5M |
| RMSNorm ×2 | $\boldsymbol{\gamma}$ | $2048 \times 2$ | 0.004M |
| **单层合计** | | | **~76.9M** |

36 层合计：$36 \times 76.9\text{M} \approx 2.77\text{B}$（加上 embed_tokens 等共 3.3B）。

---

## 6. 任务适配：LoRA 多适配器

### 6.1 为什么要用 LoRA

jina-v4 需要支持三个不同任务（检索、文本匹配、代码）。朴素方法是训练三个独立的完整模型，但参数量巨大。**LoRA（Low-Rank Adaptation）** 是一种参数高效微调方法：冻结预训练权重，只训练极少数额外参数。

### 6.2 LoRA 原理

对于一个预训练线性层 $\mathbf{W}_0 \in \mathbb{R}^{m \times n}$，LoRA 添加低秩分解：

$$\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$$

其中 $\mathbf{A} \in \mathbb{R}^{r \times n}$，$\mathbf{B} \in \mathbb{R}^{m \times r}$，$r \ll \min(m, n)$ 是低秩维度（此处 $r = 32$）。

前向传播变为：

$$\mathbf{y} = \mathbf{x}\mathbf{W}_0^\top + \frac{\alpha}{r} \cdot \mathbf{x}\mathbf{A}^\top\mathbf{B}^\top$$

其中 $\alpha$ 是缩放超参数（通常 $\alpha = r$，即 `scaling = 1.0`）。

**参数效率**：对于 $\mathbf{W}_Q \in \mathbb{R}^{2048 \times 2048}$，
- 原始参数：$2048^2 = 4.2\text{M}$
- LoRA 参数（单任务）：$2 \times 32 \times 2048 = 131\text{K}$，**节省 32×**

### 6.3 MultiAdapterLinear：动态任务选择

jina-v4 的创新在于把三个任务的 LoRA 权重**同时加载到 GPU**，通过 `task_label` 在推理时动态选择，避免频繁切换适配器：

```python
# 代码路径：custom_lora_module.py
class MultiAdapterLinear(nn.Module, LoraLayer):
    def forward(self, x, task_label):
        result = self.base_layer(x)  # 基础权重 W₀ 的输出
        if isinstance(task_label, str):
            # 整批用同一任务
            lora_A = self.lora_A['default'][task_label]   # shape: (32, 2048)
            lora_B = self.lora_B['default'][task_label]   # shape: (2048, 32)
            result += lora_B(lora_A(dropout(x))) * scaling
        else:
            # 同批次不同任务（混合任务推理）
            for task in unique_tasks:
                task_x = x[task_indices_for_task]
                result[task_indices] += lora_B(lora_A(task_x)) * scaling
        return result
```

### 6.4 LoRA 应用范围

LoRA 适配器作用于语言模型的所有层的注意力投影（Q、K、V）及 `multi_vector_projector`：

| 模块 | A 矩阵 | B 矩阵 | 每任务参数 |
|------|---------|---------|:----------:|
| `q_proj`（每层） | $(32, 2048)$ | $(2048, 32)$ | 131K |
| `k_proj`（每层） | $(32, 2048)$ | $(256, 32)$ | 73K |
| `v_proj`（每层） | $(32, 2048)$ | $(256, 32)$ | 73K |
| `multi_vector_projector` | $(32, 2048)$ | $(128, 32)$ | 69K |

36 层 × 3 模块 × 约 92K + projector = 约 10M 参数/任务，3 个任务共约 30M（外加 base LoRA 参数）。

---

## 7. 输出头：从隐藏状态到 Embedding

经过 36 层 Transformer 后，得到最后一层的隐藏状态：

$$\mathbf{H} = \text{LM}_{36}\bigl(\text{LM}_{35}\bigl(\cdots\text{LM}_1(\mathbf{X}_0)\bigr)\bigr) \in \mathbb{R}^{T \times 2048}$$

其中 $T$ 是整个序列的 token 数，$\mathbf{X}_0$ 是输入 embedding 序列。

### 7.1 Single-Vector Embedding（图像输入）

对于图像输入，仅对**视觉 token 区间**做均值池化：

**Step 1：定位图像 token 的位置**

```python
img_start_positions = torch.where(input_ids == vision_start_token_id)[1]
img_end_positions   = torch.where(input_ids == vision_end_token_id)[1]
```

**Step 2：构造 image_mask（布尔掩码）**

$$\text{mask}_{t} = \begin{cases} 1 & \text{若 } t_\text{start} \leq t \leq t_\text{end} \\ 0 & \text{否则} \end{cases}$$

**Step 3：均值池化（Masked Mean Pooling）**

$$\mathbf{p} = \frac{\sum_{t=1}^{T} \text{mask}_t \cdot \mathbf{H}_t}{\sum_{t=1}^{T} \text{mask}_t} \in \mathbb{R}^{2048}$$

等价代码：

```python
masked_hidden = hidden_states * image_mask.unsqueeze(-1)      # (T, 2048) × (T, 1)
pooled = masked_hidden.sum(dim=1) / image_mask.sum(dim=1, keepdim=True)
```

**Step 4：L2 归一化**

$$\mathbf{e} = \frac{\mathbf{p}}{\|\mathbf{p}\|_2} \in \mathbb{R}^{2048}$$

### 7.2 Single-Vector Embedding（文本输入）

文本输入没有视觉 token，对全部有效 token 做加权均值（权重为 attention_mask）：

$$\mathbf{p} = \frac{\sum_{t=1}^{T} \text{attn\_mask}_t \cdot \mathbf{H}_t}{\sum_{t=1}^{T} \text{attn\_mask}_t}$$

然后同样做 L2 归一化。这与图像路径唯一的区别是：用 `attention_mask`（排除 padding）而不是 `image_mask`（仅取视觉 token）。

### 7.3 两条路径的对比

```python
# 代码路径：modeling_jina_embeddings_v4.py
def get_single_vector_embeddings(hidden_states, attention_mask, input_ids):
    if _input_has_image(input_ids[0]):
        # 图像路径：只聚合视觉 token
        image_mask = (positions >= img_start) & (positions <= img_end)
        pooled = (hidden_states * image_mask).sum(1) / image_mask.sum(1)
    else:
        # 文本路径：聚合全部非 padding token
        pooled = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
    return F.normalize(pooled, dim=-1)
```

**关键洞察**：图像和文本的最终 embedding 都是 2048 维的 L2 归一化向量，**因此可以直接用内积比较图文相似度**（跨模态检索）。

---

## 8. 两种 Embedding 的区别

jina-v4 同时支持两种表示形式：

### 8.1 Single-Vector Embedding

- 形状：$\mathbf{e} \in \mathbb{R}^{2048}$（或 Matryoshka 截断后的 $128/256/512/1024$）
- 计算：$\text{MeanPool}(\mathbf{H}) \to \text{L2Norm}$
- 检索方式：ANN（Approximate Nearest Neighbor），用 FAISS / Milvus
- **本系统使用的就是这种**

### 8.2 Multi-Vector Embedding

- 形状：$\mathbf{E} \in \mathbb{R}^{T \times 128}$（每个 token 一个 128 维向量）
- 计算：经过 `multi_vector_projector`（2048→128 带 LoRA）后 L2 归一化，再乘以 `attention_mask`

$$\mathbf{E} = \text{L2Norm}\bigl(\mathbf{H}\,\mathbf{W}_\text{proj}^\top\bigr) \odot \text{mask}$$

其中 $\mathbf{W}_\text{proj} \in \mathbb{R}^{128 \times 2048}$（加上 LoRA delta）。

- 相似度：MaxSim（类似 ColBERT）：
$$\text{sim}(\mathbf{Q}, \mathbf{D}) = \sum_{i} \max_{j} \mathbf{Q}_i \cdot \mathbf{D}_j^\top$$

- 优点：细粒度匹配；缺点：存储量大（每张图 $T_v \times 128$ 浮点数）

---

## 9. Matryoshka 表示学习

jina-v4 支持将 2048 维 embedding 截断到更短的维度，且截断后的向量**仍然保持良好的语义质量**：

$$\mathbf{e}_{[:k]} = \text{L2Norm}\bigl(\mathbf{e}_{1:k}\bigr), \quad k \in \{128, 256, 512, 1024, 2048\}$$

这通过 **Matryoshka Representation Learning（MRL）** 训练实现：训练时同时优化多个维度的损失函数。物理意义是：前 $k$ 个维度已经编码了最重要的语义信息。

本系统使用默认的 2048 维（无截断），对应 `config.matryoshka_dims = [128, 256, 512, 1024, 2048]` 中的最大值。

---

## 10. 数据流全链路汇总

以一张图像 $I$（PDF 某页，$448 \times 448$ 像素）为例，追踪完整的计算路径：

```
输入：I ∈ ℝ^{3×448×448}（RGB图像）
  │
  ▼ [JinaEmbeddingsV4Processor.process_images]
  构建 prompt 模板 + 图像预处理
  pixel_values ∈ ℝ^{N_frames×C×H×W}，image_grid_thw = (1, 32, 32)
  │
  ▼ [ViT patch_embed: Conv3d(kernel=(2,14,14), stride=(2,14,14))]
  pixel_values → patches: 1024 patches，每个维度 1280
  │
  ▼ [ViT 32 blocks: Window Attn(×28) + Full Attn(×4) + SwiGLU]
  patches → visual_tokens: 1024 个特征，维度 1280
  │
  ▼ [Spatial Merge: 2×2 patch → 1 token, MLP Projector: 1280×4 → 2048]
  visual_tokens → merged: T_v = 256 tokens，维度 2048
  │
  ▼ [拼接 prompt token（text tokens）]
  input_sequence: T 个 tokens，维度 2048
  input_ids: (1, T)，image_mask: (1, T)，attention_mask: (1, T)
  │
  ▼ [LM: RoPE 位置编码 → 36层 Transformer（GQA + SwiGLU + RMSNorm + LoRA）]
  hidden_states[-1]: (1, T, 2048)
  │
  ▼ [get_single_vector_embeddings: Masked MeanPool on image token positions]
  pooled: (1, 2048)
  sum_{t: mask=1}(H_t) / count(mask=1)
  │
  ▼ [F.normalize(dim=-1)]
  e: (1, 2048)，‖e‖₂ = 1
  │
输出：e ∈ ℝ^{2048}（单位向量，可直接做内积相似度）
```

**关键参数统计（448×448 图像）**：

| 阶段 | 输入形状 | 输出形状 | 主要运算 |
|------|----------|----------|---------|
| Patch embed | $(3, 2, 448, 448)$ | $(1024, 1280)$ | Conv3d |
| ViT 32 blocks | $(1024, 1280)$ | $(1024, 1280)$ | Attn + SwiGLU |
| Spatial merge | $(1024, 1280)$ | $(256, 2048)$ | MLP projector |
| LM 36 layers | $(T, 2048)$，$T \approx 300$ | $(T, 2048)$ | GQA + SwiGLU |
| Pooling + Norm | $(T, 2048)$ | $(1, 2048)$ | MeanPool + L2Norm |

---

## 11. 对 Phase 3 可验证证明的启示

### 11.1 重新评估可行性

读完 zkLLM（CCS'24）和 zkGPT（Eurosys'24）后，Phase 3 的可行性远比最初估计的**乐观得多**：

| 参考点 | 模型 | 证明时间 | 证明大小 |
|--------|------|:--------:|:--------:|
| zkLLM | OPT-2.7B | ~352s | ~160KB |
| zkLLM | OPT-6.7B | ~548s | ~170KB |
| zkLLM | LLaMA-2-13B | ~803s | ~188KB |
| **jina-v4 LM（估计）** | **Qwen2.5-3B** | **~400–500s** | **~165KB** |

jina-v4 的语言塔（3.3B，36层 Transformer，hidden=2048）与 OPT-2.7B 规模相当，**zkLLM 框架原则上可以对其做完整推理证明**。

### 11.2 证明架构（修订后）

原先的"只证 pooling 头"方案可以升级为更完整的方案：

```
[被证明的范围（zkLLM 框架）]
┌─────────────────────────────────────────────────────┐
│ LM 全部 36 层（GQA + SwiGLU + RMSNorm + LoRA）        │
│ + MeanPool + L2Norm                                  │
└─────────────────────────────────────────────────────┘
       ↑ 输入：visual tokens（来自 ViT，信任边界）

[信任边界：ViT 视觉编码器（可选证明，但复杂度高）]
  Conv3d + 32 ViT blocks → visual tokens
  ViT 参数量 669M，tkLLM 未测试 Conv3d，此处暂设为信任假设
```

### 11.3 三个可行等级

| 等级 | 证明范围 | 实现难度 | 论文贡献度 |
|------|----------|:--------:|:----------:|
| 🥉 **基础** | 只证 MeanPool + L2Norm（pooling head）| 低 | 低 |
| 🥈 **中级** | 证明 LM 全部 36 层 + pooling（基于 zkLLM）| 中-高 | 中 |
| 🥇 **完整** | 证明 ViT + LM 全部层 + pooling | 高 | 高 |

**建议**：实现中级方案（LM 塔 + pooling），同时在论文中将 ViT 作为信任边界明确说明，引用 zkLLM 结论支撑可行性。

### 11.4 两篇论文的核心区别与选择建议

| 维度 | zkGPT | zkLLM |
|------|-------|-------|
| 硬件 | CPU（32线程） | GPU（A100） |
| 最大模型 | GPT-2（1.5B） | LLaMA-2-13B |
| 非线性 | Lasso lookup | tlookup（更通用）|
| 量化精度损失 | 较大（Q=16bit） | 极小（PPL <0.01）|
| 对 GQA 支持 | 未测试 | 未测试 |
| **适合 jina-v4** | **否**（规模不足）| **是**（规模匹配）|

本系统应基于 **zkLLM** 实现 Phase 3。jina-v4 使用 GQA（8:1 共享 K/V）和 LoRA，这两点需要对 zkLLM 做少量适配。

---

## 附录：模型参数速查表

| 参数名 | 值 | 说明 |
|--------|:--:|------|
| 总参数量 | ~3.9B | ViT 669M + LM 3.3B + LoRA 180M |
| LM hidden_size | 2048 | |
| LM num_layers | 36 | |
| LM num_heads (Q) | 16 | |
| LM num_heads (K/V) | 2 | GQA 8:1 |
| LM intermediate_size | 11008 | SwiGLU FFN |
| LM max_seq_len | 128000 | RoPE |
| vocab_size | 151936 | |
| ViT blocks | 32 | |
| ViT hidden_size | 1280 | |
| ViT out_hidden | 2048 | 投影后 |
| patch_size | 14×14 px | spatial |
| spatial_merge_size | 2×2 → 1 | token 压缩 |
| LoRA rank | 32 | |
| LoRA tasks | 3 | retrieval, text-matching, code |
| embedding_dim | 2048 | single-vector |
| multi_vec_dim | 128 | multi-vector |
| Matryoshka dims | 128/256/512/1024/2048 | |

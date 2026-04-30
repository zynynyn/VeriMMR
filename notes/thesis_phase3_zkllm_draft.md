# Phase 3 — 多模态嵌入推理完整性证明

> 章节定位：本章处理"编码可信性"问题——即检索服务器声称使用某多模态嵌入模型推理，实际上可能使用了低参数替代模型或篡改权重的模型。第一部分（理论基础）介绍 zkLLM 框架的三个核心协议（tlookup、zkAttn、IPA 权重承诺）及 GQA、LoRA 两个模型架构特性，Sumcheck 与 Fiat-Shamir 已在 Phase 2 中详述，此处不再重复。第二部分（框架设计）描述本文对 zkLLM 从纯语言模型扩展至多模态嵌入模型全链路的适配工作，以及 Fiat-Shamir 随机层挑战、GPU 并行调度、Batch Sumcheck 等工程优化。

---

# 第一部分：理论基础

## 1.1 zkLLM 框架概述

zkLLM \[[Sun et al., CCS 2024](https://dl.acm.org/doi/10.1145/3658644.3670391)\] 是第一个针对大语言模型推理过程的零知识证明框架，解决如下问题：Prover（AI 服务商）持有模型权重（可视为知识产权，不对外公开），Verifier（监管方或用户）提交输入 $x$，并要求对返回输出 $y$ 进行形式化验证——即在不暴露权重的前提下证明 $y = f_W(x)$。

zkLLM 的核心观察是：Transformer 的前向传播由**线性操作**（矩阵乘法）和**非线性操作**（激活函数、Softmax、LayerNorm）两类组成，可分别用两套不同的证明协议处理：

| 操作类型 | 代表操作 | 证明协议 |
|---------|---------|---------|
| 线性（矩阵乘法）| $Y = XW^\top$ | Sumcheck（第 Phase 2 已述） |
| 非线性（激活函数）| SiLU、GELU、ReLU | tlookup（§1.2） |
| 注意力（Softmax）| $\mathrm{Softmax}(QK^\top/\sqrt{d})V$ | zkAttn（§1.3） |
| 权重绑定（承诺）| $W$ 与公开承诺一致 | IPA（§1.4） |

权重量化至 int32（scale=$2^{16}$），所有计算在 $\mathbb{Z}_p$ 有限域上进行（$p = 2^{61}-1$ 或 BLS12-381 曲线标量域）。zkLLM 已在 LLaMA-2-13B（A100，803s/次）上完成验证，证明框架在十亿参数规模 LLM 上的实用可行性。

---

## 1.2 tlookup：非线性函数的可验证查表协议

### 1.2.1 协议动机

非线性激活函数（SiLU、GELU、RMSNorm 中的平方根、Softmax 中的指数）无法直接用多项式低次求和表示，因此 Sumcheck 协议无法直接处理。zkLLM 提出 **tlookup（table lookup）协议** \[[Sun et al., CCS 2024](https://dl.acm.org/doi/10.1145/3658644.3670391), §4\]，将非线性函数的每次求值视为一次查表操作，并用 Sumcheck 证明该查表的正确性。

### 1.2.2 协议构造

**离线阶段**（建库，与权重承诺同步）：对目标函数 $\sigma : \mathbb{Z} \to \mathbb{Z}$（量化整数版本），构造大小为 $M$ 的 lookup table $T$，满足 $T[j] = \sigma(j)$，$j \in [-M/2, M/2)$。

SiLU 的量化版本：$\sigma_{\text{SiLU}}(x) = \lfloor x \cdot \mathrm{sigmoid}(x/s) \cdot s \rfloor$，table 大小 $M = 2^{16}$（覆盖 scale=$2^{16}$ 下的合理激活范围）。

**在线阶段**（每次前向传播）：

设激活输入张量为 $A \in \mathbb{Z}^{S \times D}$，非线性输出为 $B \in \mathbb{Z}^{S \times D}$，满足 $B_{ij} = T[A_{ij}]$。

1. **Lookup 正确性**：Prover 声称 $\{(A_{ij}, B_{ij})\}$ 均为 $T$ 的合法 $(key, value)$ 对；
2. **Sumcheck 归约**：通过随机线性组合，将全部 $S \times D$ 对的查表正确性归约为对 $T$ 的 MLE 的一次点查询；
3. **Table 承诺**：$T$ 的 MLE 在随机点的值直接计算（verifier 持有 $T$），无需额外承诺。

**复杂度**：Prover $O(SD)$（线性于激活张量大小），Verifier $O(M)$（线性于表大小，一次性预处理）。证明大小为 $O(\log(SD))$ 个域元素，与 Sumcheck 同阶。

### 1.2.3 RMSNorm 处理（Rescaling）

RMSNorm 包含按 token 归一化（$h_i = x_i / \mathrm{rms}(x)$），其中 $\mathrm{rms}(x) = \sqrt{\frac{1}{d}\sum_j x_j^2}$ 涉及浮点平方根。zkLLM 将其处理为**可验证的 Rescaling 操作**：Prover 声称 $\mathrm{rms\_inv} = 1/\mathrm{rms}(x)$（量化整数），Verifier 用 Sumcheck 验证 $\mathrm{rms\_inv}^2 \cdot \sum_j x_j^2 \approx d$（允许量化误差 $\delta$），从而无需 ZK 电路内部做平方根运算。

---

## 1.3 zkAttn：可验证注意力计算

### 1.3.1 Softmax 的挑战

注意力机制的核心是 $\mathrm{Softmax}(\mathbf{QK}^\top / \sqrt{d_k}) \mathbf{V}$。其中 Softmax 涉及指数和归一化：

$$A_{ij} = \frac{\exp(S_{ij})}{\sum_k \exp(S_{ik})}$$

指数函数不是多项式，但在量化域上可用 tlookup 处理（table 大小约 $2^{16}$）。

### 1.3.2 zkAttn 协议

zkAttn \[[Sun et al., CCS 2024](https://dl.acm.org/doi/10.1145/3658644.3670391), §5\] 将完整注意力分为两个可证明步骤：

**Step 1**（线性部分）：$\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 的线性投影，用 Sumcheck 证明 $\mathbf{X}W_Q^\top = \mathbf{Q}$ 等；

**Step 2**（Softmax 与加权求和）：

1. $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$（矩阵乘，Sumcheck）
2. $\exp(\mathbf{S})$（逐元素指数，tlookup）
3. 行归一化 $\mathbf{A} = \mathrm{rowsum}(\exp(\mathbf{S}))^{-1} \odot \exp(\mathbf{S})$（逐元素除法，可验证 Rescaling）
4. $\mathbf{O} = \mathbf{A}\mathbf{V}$（矩阵乘，Sumcheck）

**NTT 约束**：$\mathbf{Q}\mathbf{K}^\top$ 的矩阵尺寸为 $(\text{seq} \times d_k)$，zkAttn 内部的数论变换（NTT，用于多项式快速乘法）要求 $\text{seq}^2$ 为 2 的幂次。对 seq=1024：$1024^2 = 2^{20}$，满足约束。这一约束也决定了系统中激活 padding 的目标 seq_len（详见 §2.2.3）。

---

## 1.4 IPA 权重承诺

### 1.4.1 承诺方案选择

zkLLM 使用 **IPA（Inner Product Argument）** \[[Bootle et al., 2016](https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21); [Bünz et al., 2018](https://eprint.iacr.org/2017/1066.pdf)\] 作为权重承诺方案。对每个权重矩阵 $W \in \mathbb{Z}^{m \times n}$，离线计算其多线性扩展（MLE）在生成元集上的承诺 $\mathrm{cm}_W \in \mathbb{G}_1$（BLS12-381）。

IPA 的核心性质：Verifier 可在不访问 $W$ 本身的前提下，通过 $k = \lceil \log_2 n \rceil$ 轮折叠协议（每轮发送 2 个 $\mathbb{G}_1$ 点）验证 $W$ 在任意随机点 $\mathbf{r}$ 处的 MLE 值：

$$\widetilde{W}(r_1, \ldots, r_\ell) = y \quad \text{（Prover 声称值）}$$

**折叠规则**（每轮 challenge $u$）：
$$C_\text{new} = (1-u)^2 \cdot L_0 + u(1-u) \cdot C + u^2 \cdot L_1$$

最终检查 $C_\text{final} \stackrel{?}{=} g_\text{final} \cdot w_\text{final}$，其中 $g_\text{final}$ 和 $w_\text{final}$ 为最后一轮的生成元与证人。

### 1.4.2 IPA proof 文件格式

本文实现将 C++ GPU prover（blstrs，Jacobian 坐标）生成的 IPA proof 序列化为二进制文件，供 Python verifier 独立验证（无需访问原始权重）：

```
[4B] magic=0x49504100  [4B] k  [4B] com_log
[144B] C_init  [k×32B] u_in（标准 Fr 元素）
[k×288B] 每轮 (L0_i=144B, L1_i=144B)
[144B] g_final  [32B] w_final
```

坐标系统差异：GPU 库（blstrs）使用 Jacobian 坐标 $(X:Y:Z)$ 表示仿射点 $(X/Z^2, Y/Z^3)$；Python 库（py_ecc）使用射影坐标。Python verifier 读取时先将 Jacobian 转换为仿射再调用 py_ecc，保证两端一致。

### 1.4.3 线性操作完整性证明流程

对权重 $W$、输入 $X$、输出 $Y = XW^\top$，完整证明流程：

1. Prover 对 $Y$ 做 Sumcheck，将证明归约至 $\widetilde{X}(\mathbf{u}_X) \cdot \widetilde{W}(\mathbf{u}_W) = c$（两个多线性 MLE 的点积）；
2. Prover 用 IPA 开放 $\widetilde{W}(\mathbf{u}_W)$，验证与承诺 $\mathrm{cm}_W$ 一致；
3. Verifier 独立计算 $\widetilde{X}(\mathbf{u}_X)$（Verifier 持有激活 $X$），验证 $\widetilde{X}(\mathbf{u}_X) \cdot y_W = c$。

---

## 1.5 LoRA 低秩适配（Hu et al., 2022）

LoRA（Low-Rank Adaptation）\[[Hu et al., 2022](https://arxiv.org/abs/2106.09685)\] 将权重更新分解为低秩矩阵乘：

$$W_\text{eff} = W_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}, \quad \mathbf{A} \in \mathbb{R}^{r \times d_\text{in}},\; \mathbf{B} \in \mathbb{R}^{d_\text{out} \times r}$$

对于可验证推理，直接证明 $Y = X W_\text{eff}^\top$ 等价于证明 $Y = X W_0^\top + \frac{\alpha}{r} X \mathbf{A}^\top \mathbf{B}^\top$，需要额外的两条 Sumcheck 链（$X\mathbf{A}^\top$ 和 $(X\mathbf{A}^\top)\mathbf{B}^\top$），开销约为原层的 $2r/d = 2 \times 32/2048 \approx 3\%$。

---

## 1.6 GQA：分组查询注意力（Ainslie et al., 2023）

GQA（Grouped Query Attention）\[[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)\] 通过让多个 Q-head 共享同一组 KV-head 来减少 KV cache 占用：$n_Q$ 个 Query head 分为 $g$ 组，每组共享 $n_K = n_Q / g$ 个 KV head。对 jina-v4 语言塔：$n_Q = 16$，$n_K = 2$，group size $g = 8$（即 8 个 Q-head 共享 1 个 KV-head），$d_k = 256$（$=2048/8$）。

GQA 的注意力公式与 MHA 一致（$\mathrm{Softmax}(\mathbf{Q}_i\mathbf{K}_{i/g}^\top/\sqrt{d_k})\mathbf{V}_{i/g}$），但 K/V 矩阵尺寸从 $d$ 降至 $d/g=256$，这对 zkAttn 的 NTT 约束产生影响（见 §2.2.2）。

---

# 第二部分：框架设计

> 章节定位：以下内容为本文的设计贡献——将 zkLLM 框架从针对单模态 LLM 扩展至多模态嵌入模型的完整推理链，包含 GQA 适配、LoRA 量化合并、全链路组件覆盖（视觉编码器 + 语言塔 + 融合层）与多项工程优化。

---

## 2.1 解决问题：编码欺骗（B4）

### 2.1.1 威胁模型

Phase 1 和 Phase 2 分别证明了"返回图像来自已承诺语料库"和"相似度分值计算正确、排名未被操控"，但两者均不限制检索服务器使用**哪个模型**生成 embedding。一个理性的攻击者可以：

| 攻击类型（B4）| 攻击者操作 | 危害 |
|:---:|---|---|
| 整模型替换 | 声称使用 jina-v4（3.3B），实际使用低参数廉价模型计算 embedding | 检索质量下降，用户为高性能服务付费但得到低质量结果 |
| 部分层篡改 | 替换 $L$ 层权重矩阵，保持公开承诺文件不变 | 局部操控推理结果，难以直接观测 |
| LoRA 未应用 | 声称使用带 LoRA 微调的专用嵌入模型，实际使用裸基座 | embedding 语义偏移，特定领域查询精度下降 |
| 精度降级 | 声称 FP32 精度，实际使用 INT4/INT8 量化推理 | 低算力成本产生的 embedding 充当高精度结果 |

上述攻击均无法被 Phase 1/2 检测，因为攻击者可以保持图像文件和 embedding 值（embedding.npy）不变——篡改发生在推理过程本身，是"结果正确但过程欺骗"的一类攻击。

### 2.1.2 证明目标

Phase 3 需密码学证明：

$$y = f_W(x) \quad \text{（在 } W \text{ 的公开承诺约束下）}$$

即对给定的输入 $x$（图像像素或文本 token），输出 embedding $y$ 确实由承诺权重集 $\{W_i\}$ 通过正确的多模态编码器计算得到，不存在权重替换或推理捷径。

**信任起点**：原始图像像素（用户提供）→ **信任终点**：$y$ = jina-v4 正确推理结果。

---

## 2.2 内部设计：多模态全链路证明架构

### 2.2.1 完整架构图

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  多模态嵌入推理完整性证明架构                                                   │
│                                                                                 │
│  [输入]                                                                         │
│  图像像素 (T×C×H×W)                                                             │
│       │                                                                         │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│       │  ① Conv3d Patch Embedding                                               │
│       │    im2col展开 → 稀疏矩阵乘（Sumcheck + IPA）                             │
│       │    权重: 1×(1176→1280)，证明: 1 个 IPA proof                            │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│       │  ② ViT 32 Blocks（Window+Full Attention）                               │
│       │    每块：RMSNorm → Q/K/V linear → MHA zkAttn → O proj →                │
│       │           Skip → RMSNorm → SwiGLU FFN → Skip                           │
│       │    证明: 9 个 IPA proof × 32 块 = 288 个 IPA proof                      │
│       │    并行: GPU0(偶数块) ‖ GPU1(奇数块)                                    │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│       │  ③ PatchMerger (视觉→语言桥接层)                                        │
│       │    RMSNorm(1280) → reshape(n/4, 5120) →                                │
│       │    FC1(5120→5120)+GELU → FC2(5120→2048)                                │
│       │    证明: 3 个 IPA proof（RMSNorm + FC1 + FC2）                          │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│       │  ④ LLM 36 Decoder Layers（GQA + SwiGLU）                               │
│       │    每层：RMSNorm → Q/K/V linear(GQA) → GQA zkAttn →                   │
│       │           O proj → Skip → RMSNorm → SwiGLU(gate+up+down) → Skip       │
│       │    在线: Fiat-Shamir 随机抽取 K=6 层证明                                │
│       │    离线: 全部 36 层预计算（每图像 88.8s，总计 3.7h）                    │
│       │    并行: GPU0(层 0,2,...) ‖ GPU1(层 1,3,...)                           │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│       │  ⑤ Pooling Head (MeanPool + L2Norm)                                    │
│       │    MeanPool: Sumcheck 证明 sum(H[mask]) = K × p                        │
│       │    L2Norm: Rescaling 证明（等价单行 RMSNorm）                           │
│  ━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  [输出]                                                                         │
│  embedding y ∈ ℝ²⁰⁴⁸                                                           │
└────────────────────────────────────────────────────────────────────────────────┘

  总 IPA proof 数：1 + 288 + 3 + (≤216) + 0 = 292–508 个（取决于 K）
  全量验证时间（C++ verify-ipa，2-GPU）：~9 min
```

### 2.2.2 GQA 适配：三项工程改造

原始 zkLLM 实现面向 MHA（Multi-Head Attention，每个 Q-head 对应独立 KV-head），jina-v4 语言塔采用 GQA（16 Q-head / 2 KV-head，group size=8），需对 `self-attn.cu` 做三项改造：

| 编号 | 改造点 | 问题根源 | 改造方法 |
|:---:|--------|---------|---------|
| ① | per-head 循环 | 原代码假设 $n_Q = n_K$，对 GQA 产生错误的 Q-KV 广播映射 | 改为显式 per-head 循环：第 $i$ 个 Q-head 对应 KV-head $\lfloor i \cdot n_K / n_Q \rfloor$ |
| ② | KV head transpose | GQA 的 K/V 矩阵内存布局与 MHA 不同，直接读取产生越界或错列 | 在传入 CUDA kernel 前对 K/V 张量做转置，使 $d_k$ 维连续排列 |
| ③ | 动态 Rescaling | GQA 各 Q-head 累积量量级随 $d_k$ 变化，固定因子导致部分 head 整数溢出，zkAttn Sumcheck 自洽性失败 | 改为 per-head 动态计算 Rescaling 因子，保证每 head 整数累积量 $\leq 2^{52}$ |

**GQA zkAttn NTT 约束**：zkAttn 要求 $\text{seq}^2 = 2^{20}$，故 seq_len=1024。同时，tlookup 约束要求 $(\text{seq} \times d_k) \% 65536 = 0$，kv_dim=256 时 $1024 \times 256 = 262144 = 4 \times 65536$，满足。

**正确性验证**（E1 实验，layer-35，seq_len=1024，RTX 4090 D）：

| 指标 | 结果 | 阈值 | 验证改造点 |
|------|:----:|:---:|:--------:|
| $L_\infty$ 误差（CUDA vs float32） | **4.47×10⁻⁷** | $< 10^{-4}$ | ①② Q→KV 映射与内存布局 |
| 余弦相似度（有效 token，CUDA vs float32） | **1.00000000** | $> 0.9999$ | ①② 整体方向 |
| ZK 证明自洽性（returncode） | **0** | $= 0$ | ③ 动态 Rescaling |

GQA 线性扩展验证（E2 实验）：$n_K$ 从 1 增至 2，耗时从 3.54s 增至 6.23s（$\approx 2\times$），验证 per-head 循环无冗余计算。

### 2.2.3 各组件适配要点

**① Conv3d Patch Embedding（视觉输入层）**

3D 卷积等价展开：每个输出 patch 是 $K_T \times K_H \times K_W \times C = 2 \times 14 \times 14 \times 3 = 1176$ 个输入像素的线性组合，展开为 im2col 后得到 $(N_\text{patch}, 1176)$ 矩阵乘 $(1176, 1280)$ 权重，直接用 Sumcheck + IPA 证明。$N_\text{patch} = 32$（smoke test）或实际 patch 数，1 个 IPA proof。

**② ViT 32 Blocks（视觉 Transformer）**

jina-v4 ViT 使用 MHA（$n_Q = n_K = 16$，head\_dim=80），不涉及 GQA，可直接使用原始 zkAttn。每块 8 步证明链：

```
RMSNorm → Attn-linear(1280→1280×3) → zkAttn(seq=1024, MHA) →
O-proj(1280→1280) → Skip → RMSNorm → SwiGLU(1280→3420→1280) → Skip
```

每块生成 9 个 IPA proof（input_layernorm、k/q/v_proj、o_proj、post_attention_layernorm、gate/up/down_proj），共 $32 \times 9 = 288$ 个 IPA proof。

**③ PatchMerger（视觉-语言桥接层）**

PatchMerger 将 $n$ 个 1280 维 ViT patch 合并为 $n/4$ 个 2048 维语言 token。非线性激活为 **GELU**（而非 LLM 中的 SiLU），需离线生成 GELU lookup table（$524288 = 2^{19}$ 条目，SCALE\_IN=$2^{12}$，SCALE\_OUT=$2^{16}$）。

关键工程问题：FC1 输出在 GELU 查表前需要两阶段 Rescaling（将中间尺度从 $2^{16}$ 降至 $2^{12}$），两阶段分别满足 `D_padded % N == 0` 约束（$N_1 = 2^{16}$：$524288 / 65536 = 8$；$N_2 = 2^4$：$524288 / 16 = 32768$），串联缩放因子 $2^{20}$ 精确还原 GELU 输入尺度。

**④ LLM 36 Decoder Layers（GQA 语言塔）**

每层三段式（FFN + Attn-linear + GQA zkAttn），hook 位置：
- Attn 输入：`layers[l].input_layernorm` 的输出（量化后写入 `layer-{l}-attn-input.bin`）
- FFN 输入：`layers[l].post_attention_layernorm` 的输出（写入 `layer-{l}-ffn-input.bin`）

seq_len 统一 pad 到 1024（补零）。Hook 非侵入性已由实验 3.H.2 验证（cos\_sim=1.00000000，overhead=-0.3%）。

**⑤ Pooling Head（MeanPool + L2Norm）**

- **MeanPool**：公开 mask 向量（无可学习参数），用 Sumcheck 证明 $\sum_{t:\text{mask}[t]=1} H_t = K \cdot \mathbf{p}$，Verifier 直接验证求和；
- **L2Norm**：等价于无可学习 $\gamma$ 的单行 RMSNorm，用 Rescaling 证明（Prover 声称 $1/\|p\|_2$，Verifier 验证 $\|p\|_2^2 \cdot \text{inv}^2 \approx 1$）。

---

## 2.3 安全指标：B4 检出率与 Fiat-Shamir 随机层挑战

### 2.3.1 固定层 vs 随机挑战的安全差异

若在线查询固定证明"最后 K 层"，理性攻击者只需保持这 $K$ 层权重不变即可绕过检测，其余 $36-K$ 层可任意替换——即以 $O(K)$ 计算代价维持合法外观。

**Fiat-Shamir 随机层挑战**将层选择变为非交互随机过程：

$$\text{challenge} = \mathrm{SHA256}(\text{query\_text} \,\|\, \text{nonce})$$

以 challenge 为 PRNG 种子，从 $\{0, 1, \ldots, 35\}$ 中无放回均匀抽取 $K$ 层。Verifier 持有相同的 $(\text{query}, \text{nonce})$ 即可独立重现层选择，过程完全非交互。

**安全保证**：若攻击者篡改了 $L$ 层权重，随机挑战选中至少 1 个被篡改层的概率为：

$$P(\text{caught} \mid L \text{ 层被篡改}) = 1 - \frac{\binom{36-L}{K}}{\binom{36}{K}}$$

### 2.3.2 B4 检出率表（$N=36$，$K=6$，公式精确值）

| 被篡改层数 $L$ | $K=3$ | $K=6$（推荐）| $K=12$ |
|:---:|:---:|:---:|:---:|
| 1 | 8.33% | 16.67% | 33.33% |
| 3 | 23.59% | 43.14% | 71.65% |
| **6**（节省 ≥17% 算力）| **43.14%** | **69.52%** | **93.09%** |
| 12 | 71.65% | 93.09% | 99.78% |
| 18（替换半数层）| 88.57% | 99.05% | ≈100% |
| **36**（整模型替换）| **100%** | **100%** | **100%** |

> 注：$L \geq 36 - K$ 时 $\binom{36-L}{K} = 0$，故检出率为 100%（确定性检出）。

整模型替换是 B4 攻击的主要形式（服务商用低参数模型替代高性能模型节省算力），此情形下 $K=6$ 提供 **100% 确定性检出率**。

### 2.3.3 实验验证

**B4 权重篡改实验**（在 layer-35 的 `mlp.gate_proj.weight-int.bin` 加 $\pm 2^{20}$ 随机扰动，承诺文件不变）：

| 步骤 | 权重状态 | FFN binary 返回码 | 验证结果 |
|------|---------|:----------------:|:-------:|
| 基准 | 原始 $W_\text{gate}$ | **0** | ✅ 通过 |
| 篡改后 | $W_\text{gate} + \Delta W$（$\pm 2^{20}$，100% 元素受影响） | **−6** | ✅ 检测到 |

篡改后 $\widetilde{W_\text{tampered}}(\mathbf{u}_W) \neq \widetilde{W_\text{original}}(\mathbf{u}_W)$，IPA 开放值与承诺不匹配，Sumcheck 拒绝。检测率：1/1 = **100%**。

---

## 2.4 关键参数与优化实验

### 2.4.1 K 层性能曲线与安全-代价权衡

**实验设计**：使用 jina-v4 真实权重（含 LoRA retrieval，合并量化），seq_len=1024，GQA kv_dim=256，RTX 4090 D，测量不同 $K$ 值下的证明耗时。

| $K$ | 在线证明墙钟（双 GPU，层间并行）| $P(L=1)$ | $P(L=6)$ | $P(L=18)$ | $P(L=36)$ |
|:---:|:--------------------------:|:-------:|:-------:|:--------:|:--------:|
| 3 | **~31 s** | 8.33% | 43.14% | 88.57% | 100% |
| **6（推荐）** | **~46 s** | **16.67%** | **69.52%** | **99.05%** | **100%** |
| 12 | ~91 s | 33.33% | 93.09% | ≈100% | 100% |
| 36（全量）| ~280 s | 100% | 100% | 100% | 100% |

每层三段式耗时基准（seq_len=1024，实测 / 推算）：

| 阶段 | 耗时 | 来源 |
|------|:----:|:---:|
| FFN proof（gate+up+down） | ~5.2s | K=1 bench 实测 |
| Attn-linear proof（q/k/v） | ~2.5s | K=1 bench 实测 |
| GQA zkAttn proof（Softmax）| ~7.1s | 6 层实测 74s 反推 |
| **单层合计（串行）** | **~14.8s** | — |

**K=6 推荐理由**：整模型替换（B4 最主要攻击形式）100% 检出；$\geq 6$ 层局部篡改检出率 ≥ 70%；在线证明 ~46s 异步后台执行，不阻塞用户答案返回（用户感知延迟仍为 5.6s）。

### 2.4.2 LoRA 量化合并（消除 3% 额外开销）

理论上，LoRA 适配器每层需要额外两条 Sumcheck 链，开销约 3%。本文采用**量化前合并**方案：

$$W_\text{eff} = W_0 + \frac{\alpha}{r}\mathbf{BA} \quad \text{（float32 精确计算）}$$

$$W_\text{int} = \mathrm{round}(W_\text{eff}^\top \times 2^{16}).\text{int32}$$

合并后的 $W_\text{int}$ 已包含 LoRA 贡献，zkLLM Sumcheck 对 $W_\text{eff}$ 的证明隐式覆盖 LoRA delta，**不需要任何额外 CUDA 代码修改**。验证：合并前后 $W_\text{eff}$ 的余弦相似度 = 1.000000，差异仅为量化舍入误差（$\max \text{err} = 6 \times 10^{-4}$）；实际运行时间与合成权重完全一致（$+0\%$）。

### 2.4.3 激活量化（Hook 实验数据）

**实测激活分布**（jina-v4，layer 30–35，5 条文本 query）：

| Hook 点 | max\_abs | p99 | sf=$2^{16}$ 溢出率 | 量化余弦误差 |
|---------|:--------:|:---:|:-----------------:|:----------:|
| pre\_attn（input\_ln 输出）| 19.6 | 4.69 | 0 | ≈ $-5\times10^{-7}$ |
| pre\_ffn（post\_attn\_ln 输出）| 82.0 | 2.58 | 0 | ≈ $-5\times10^{-7}$ |

最大激活值 82.0 时，$82 \times 2^{16} = 5.4 \times 10^6 \ll 2^{31}$，int32 绝对不溢出。选定 sf=$2^{16}$ 与 Phase 2（Sumcheck）及 zkLLM 默认量化统一对齐。

**Hook 非侵入性**：12 个 hook（layers 30–35 各 2 个）对推理延迟的影响为 $-0.3\%$（测量噪声范围内），对 embedding 输出的影响 cos\_sim=1.00000000（精确无差）。

### 2.4.4 并行策略对比

| 策略 | 原理 | K=6 墙钟 | 适用场景 |
|------|------|:--------:|:-------:|
| 单卡串行 | 6 层顺序，每层 14.8s | **88.8s** | 调试 |
| 层内双卡（naive） | FFN ‖ Attn-linear，再 zkAttn | **74s**（实测）| — |
| 层内双卡（修复后）| FFN ‖ (Attn-linear→zkAttn) | **57.6s** | — |
| **层间并行（推荐）** | GPU0:层 0,2,... ‖ GPU1:层 1,3,... | **~45s** | 在线 query |
| 多 worker（语料库）| 各 GPU 独立处理不同图像 | 88.8s/GPU | 离线建库 |

层间并行比层内双卡快的根本原因：hook 预捕获后各层证明无数据依赖，可任意调度；层内双卡受 zkAttn（~7.1s，顺序执行）限制，瓶颈为 $\max(5.2, 2.5+7.1) = 9.6\text{s/层}$；层间并行无此限制，有效利用率接近 100%。

### 2.4.5 Batch Sumcheck 优化

**原理**：gate/up 两个 FFN 投影共享同一输入 $X$，用 Schwartz-Zippel 随机组合将两条独立的 zkip 归约为一条：

$$\alpha \cdot (XW_\text{gate}^\top) + (1-\alpha) \cdot (XW_\text{up}^\top) = X(\alpha W_\text{gate}^\top + (1-\alpha)W_\text{up}^\top)$$

对 Attn-linear 同理，q/k/v 三条归约为一条。

**实测改善**（LLM layer 30，RTX 4090 D）：

| 优化 | 单层耗时 | 改善 | IPA 验证 |
|------|:-------:|:---:|:-------:|
| 基线（verify-ipa 后）| 22.5s | — | 9/9 PASS |
| **+ Batch Sumcheck** | **17.3s** | **−23%** | **9/9 PASS** |

### 2.4.6 C++ GPU verify-ipa 替换 Python IPA 验证器

**瓶颈分析**：Python py_ecc G1 乘法约 7ms/次；ViT 每块 9 个 IPA proof 的 binding check 需 $2 \times 1280 = 2560$ 次 G1 乘法，耗时 $\approx 162\text{s/块}$，导致 ViT 32 块总时间 76 分钟（GPU prover 本身仅 13s/块）。

**解决方案**：新增 `src/zkllm/verify-ipa.cu`，接口 `./verify-ipa <proof_file> <commitment_file>`，输出 JSON `{"fold_ok":true,"binding_ok":true}`。GPU G1 MLE 将每次 G1 乘法从 7ms 降至 $< 0.01\text{ms}$，加速 $> 700\times$。

**实测性能对比**：

| 组件 | Python py_ecc | C++ GPU verify-ipa | 加速比 |
|------|:------------:|:-----------------:|:-----:|
| ViT block（9 proof） | ~288s | **16.7s** | **17×** |
| LLM layer（9 proof） | ~45s | **22.5s** | **2×** |
| LLM 2 层（双 GPU 并行）| ~90s | **24s** | **3.8×** |

> LLM 加速比较低（2×）是因为 GPU prover（~18s/层）已成为新瓶颈；IPA 验证本身已从 ~27s 降至 < 0.5s。

**全量证明耗时对比**（estimate，2-GPU）：

| 组件 | 优化前 | 优化后（verify-ipa + Batch）| 加速比 |
|------|:------:|:-------------------------:|:-----:|
| ViT 32 blocks | 76 min | **~5.3 min** | **14×** |
| LLM 36 layers | ~27 min | **~6.0 min** | **4.5×** |
| PatchMerger | 2.0s | < 0.5s | >4× |
| **总计** | **~103 min** | **~11.3 min** | **~9×** |

---

## 2.5 完整覆盖结果

**IPA 验证总计**（全部组件，fold + binding 双重校验）：

| 组件 | 证明类型 | 权重矩阵数 | 结果 |
|------|---------|:---------:|:---:|
| Conv3d embed | IPA | 1 | **1/1 PASS** |
| ViT 32 blocks | IPA × 9/块 | 288 | **288/288 PASS** |
| PatchMerger | IPA × 3 | 3 | **3/3 PASS** |
| Pooling head | Sumcheck + Rescaling | — | **PASS** |
| LLM 36 layers（离线）| IPA × 6/层 | 216 | **216/216 PASS** |
| **总计** | — | **508** | **全部通过** |

**系统全组件覆盖**（三层证明联合）：

| Phase | 防御内容 | 机制 | 检测率 |
|:-----:|---------|------|:------:|
| Phase 1 | 图像来源 + embedding 绑定 | ZAC（Bloom Filter + Pointproofs）| 100%（B1/B2）|
| Phase 2 | 相似度分值 + 排名 | Global Batch Sumcheck | 100%（B3）|
| Phase 3 | 推理权重完整性 | zkLLM IPA + Fiat-Shamir 随机挑战 | 100%（B4，整模型替换）|

---

## 2.6 设计选项与局限性

### 2.6.1 已实现的优化

| 优化 | 效果 | 节 |
|------|------|:--:|
| LoRA 量化前合并 | 消除 3% LoRA 额外 Sumcheck 开销 | §2.4.2 |
| seq_len padding 到 1024 | 满足 zkAttn NTT 约束，比 seq=512 少 ~13% 耗时 | §2.2.2 |
| C++ GPU verify-ipa | IPA 验证 700× 加速，全量从 103min 降至 ~11min | §2.4.6 |
| Batch Sumcheck（gate+up，q/k/v）| 单层耗时降低 23% | §2.4.5 |
| 层间并行（2-GPU）| K=6 在线证明从 89s 降至 ~45s | §2.4.4 |
| per-block 独立子目录 | 消除并行时临时文件冲突，支持 2-GPU 并发 | §2.2.2 |

### 2.6.2 当前局限性

**① 语料库全量证明时间长**

离线预计算 303 张图像的全量 5 组件证明（Conv3d + ViT 32 块 + PatchMerger + Pooling + LLM 36 层）预计约 56 小时（两 GPU 并行，~22 min/图）。与之相比，LLaMA-2-13B 的论文实现在 A100 上为 803s/次推理，本系统总证明链路更长（多了 ViT 和跨模态桥接层），时间量级合理。实用部署中可通过增加 GPU 数量线性降低建库时间。

**② 在线 K 层证明与全量推理的安全差距**

在线查询仅证明随机选取的 $K=6$ 层，而语料库离线证明为全量 36 层（未来将扩展为全量 5 组件）。$K=6$ 的随机挑战对整模型替换（B4 主要形式）提供 100% 检出率，但对局部单层篡改（$L=1$）检出率仅 16.7%。对安全性要求更高的场景，可调大 $K$（每增加一层约增加 15s）。

**③ ViT 的 Window Attention 未区分处理**

jina-v4 ViT 前 28 块使用 Window Attention（每 window 64 patches），后 4 块使用 Full Attention（1024 patches）。当前实现统一用 seq=1024 证明所有块（前 28 块存在大量 padding），实际上 Window Attention 仅需 seq=64（$64^2 = 4096 = 2^{12}$，同样满足 NTT 约束），可减少约 75% 的 zkAttn 计算量，留作后续优化。

**④ 权重承诺公开性假设**

IPA 方案要求权重承诺 $\{\mathrm{cm}_{W_i}\}$ 事先通过可信渠道公开发布（类似代码签名）。若承诺本身被伪造或建库时权重与公开承诺不一致，协议安全性无法保证。在实际部署中，承诺文件应由可信第三方（如模型原始发布方）签署并独立分发，与服务商的承诺文件相互印证。

**⑤ 不支持 ANN 索引**

当前系统依赖 FAISS IndexFlatIP（精确搜索），Phase 2 Sumcheck 的正确性依赖精确内积计算。在 $N > 10^6$ 的大规模语料库场景下，需要 HNSW 或 IVF-PQ 等近似最近邻索引，与精确 Sumcheck 存在根本矛盾。未来可探索在 ANN 候选集内部做精确重排序阶段的 Sumcheck（"两阶段验证"），或引入支持近似性误差显式建模的可验证 ANN 框架。

---

## 参考文献

\[Sun et al., 2024\] Haochen Sun, Jason Li, and Hongyang Zhang. zkLLM: Zero Knowledge Proofs for Large Language Models. In *Proceedings of ACM CCS 2024*, pp. 4197–4211. [https://dl.acm.org/doi/10.1145/3658644.3670391](https://dl.acm.org/doi/10.1145/3658644.3670391)（长版：[https://arxiv.org/abs/2404.16109](https://arxiv.org/abs/2404.16109)）

\[Bootle et al., 2016\] Jonathan Bootle, Andrea Cerulli, Pyrros Chaidos, Jens Groth, and Christophe Petit. Efficient Zero-Knowledge Arguments for Arithmetic Circuits in the Discrete Log Setting. In *EUROCRYPT 2016*, LNCS 9666:327–357. [https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21](https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21)

\[Bünz et al., 2018\] Benedikt Bünz, Jonathan Bootle, Dan Boneh, Andrew Poelstra, Pieter Wuille, and Greg Maxwell. Bulletproofs: Short Proofs for Confidential Transactions and More. In *IEEE S&P 2018*, pp. 315–334. [https://eprint.iacr.org/2017/1066.pdf](https://eprint.iacr.org/2017/1066.pdf)

\[Hu et al., 2022\] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In *ICLR 2022*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

\[Ainslie et al., 2023\] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In *EMNLP 2023*. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

\[Thaler, 2022\] Justin Thaler. *Proofs, Arguments, and Zero-Knowledge*. Foundations and Trends in Privacy and Security, 2022. [https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

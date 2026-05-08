# Phase 3 — 多模态嵌入推理完整性证明

> 章节定位：本章处理"编码可信性"问题——即检索服务器声称使用某多模态嵌入模型推理，实际上可能使用了低参数替代模型或篡改权重的模型。第一部分（理论基础）按大模型推理的运算类型分类介绍相关 ZKP 证明技术：线性运算（矩阵乘法、卷积展开）对应 Sumcheck 协议与 IPA 权重承诺；非线性激活函数对应 tlookup 查表协议（zkLLM \[Sun et al., CCS 2024\] §4）；注意力机制中的 Softmax 对应 zkAttn 协议（zkLLM §5）；归一化（RMSNorm）对应 Rescaling 代数约束（zkGPT \[Qu et al., USENIX Security 2024\] §5）。LoRA 量化合并与 GQA 适配属工程设计，见第二部分（§2.4.2、§2.2.2）。Sumcheck 与 Fiat-Shamir 已在 Phase 2 中详述，此处不再重复。第二部分（框架设计）描述本文对 zkLLM 从纯语言模型扩展至多模态嵌入模型全链路的适配工作，以及 Fiat-Shamir 随机层挑战、GPU 并行调度、Batch Sumcheck 等工程优化。

---

# 第一部分：理论基础

## 1.1 大模型推理的可验证化：ZKP 技术体系概述

Transformer 模型的前向传播可分解为若干类基本运算，每类运算对应一套特定的 ZKP 证明技术：

| 运算类型 | 代表操作 | ZKP 证明技术 | 文献来源 |
|---------|---------|------------|---------|
| 线性运算（矩阵乘、卷积展开）| $Y = XW^\top$ | Sumcheck（Phase 2 已述）+ IPA 权重承诺（§1.4）| \[Thaler, 2022; Bootle et al., 2016\] |
| 非线性激活 | SiLU、GELU、ReLU | tlookup 查表协议（§1.2）| \[Sun et al., CCS 2024\] §4 |
| 注意力（Softmax）| $\mathrm{Softmax}(QK^\top/\sqrt{d})V$ | zkAttn（§1.3）| \[Sun et al., CCS 2024\] §5 |
| 归一化（RMSNorm）| $h = x/\mathrm{rms}(x) \cdot \gamma$ | Rescaling 代数约束（§1.2.3）| \[Qu et al., 2024\] §5 |

所有计算在整数量化域（int32，scale=$2^{16}$）上进行，对应有限域 $\mathbb{Z}_p$（$p = 2^{61}-1$ 或 BLS12-381 曲线标量域）。

本文实现在整体框架上参考了 zkLLM \[[Sun et al., CCS 2024](https://dl.acm.org/doi/10.1145/3658644.3670391)\]，将上述证明协议组合为覆盖多模态嵌入模型完整推理链路的可验证系统。其中 tlookup（§1.2）与 zkAttn（§1.3）是 zkLLM 专门为大语言模型推理设计的协议；Rescaling 代数约束（§1.2.3）来自 zkGPT \[[Qu et al., USENIX Security 2024]\] §5 的约束融合（Constraint Fusion）技术；IPA 权重承诺（§1.4）是独立于特定框架的通用密码学协议（Bulletproofs 家族）。

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

### 1.2.3 RMSNorm 归一化：Rescaling 代数约束（zkGPT §5）

**zkGPT 约束四分类**

zkGPT \[[Qu et al., USENIX Security 2024]\] §5 将大模型推理中涉及的非算术运算归纳为四类约束：

| 类型 | 包含运算 | 代表场景 |
|:---:|--------|---------|
| **Type-I** | 纯算术（加、乘）| 矩阵乘法、残差连接 |
| **Type-II** | 除法 | 量化 Rescaling（整数缩放） |
| **Type-III** | 平方根 + 除法 | RMSNorm（$1/\sqrt{\Sigma x^2}$）、L2Norm |
| **Type-IV** | 指数 / 查表 | Softmax $\exp$、SiLU、GELU |

约束融合（Constraint Fusion）的可行性取决于相邻约束的类型组合：Type-I↔II 与 Type-II↔III 之间的合并通常**有利（Profitable）**；而与 Type-IV（查表）的合并通常**无利（Unprofitable）**——因为查表输入必须为整数，rounding 误差无法在融合后消除。

**本文各操作的约束类型与处理方式**

| 操作 | 约束类型 | 处理方式 |
|------|:-------:|---------|
| 量化 Rescaling（中间层缩放）| **Type-II** | 相邻两次 Type-II 融合（`prove_chain_with()`，Profitable） |
| RMSNorm（$1/\sqrt{\Sigma x^2}$）| **Type-III** | 代数约束替换 → 降为 Type-I（见下） |
| L2Norm（$1/\|\mathbf{p}\|_2$）| **Type-III** | 同上，`algebraic_constraint_v2` |
| $\exp(S)$（zkAttn Softmax）| **Type-IV** | tlookup 单独处理，**不与相邻除法融合** |
| Softmax 行归一化（$1/\text{rowsum}$）| **Type-II** | 独立 Rescaling，与 exp（Type-IV）分离 |

**Type-III → Type-I 代数降阶**

RMSNorm 和 L2Norm 均属 Type-III 约束，核心难点在于平方根。本文采用变量替换将其降阶为 Type-I 的整数乘法检查：Prover 声称 $\hat{r} = \lfloor S / \sqrt{q} \rceil$（量化整数，$S$ 为缩放因子，$q$ 为整数范数），Verifier 验证：

$$\bigl|\hat{r}^2 \cdot q - S^2\bigr| \leq 2S$$

此约束完全由整数乘法与比较构成（Type-I），无需在 ZK 电路内构造平方根或除法电路。这也意味着该约束可与相邻 Type-I/II 约束进一步融合，而不受 Type-IV 的 rounding 限制。

**为何不与 tlookup（Type-IV）融合**

zkAttn 中 $\exp$ 通过 tlookup（Type-IV）处理，其后续行归一化为 Type-II（除法）。按 zkGPT 的分析，二者融合无利：查表要求输入为精确整数索引，若将除法约束合并进来，rounding 误差会破坏整数性，导致查表失效。本文实现将二者严格分离：tlookup 证明 $\exp$ 的正确性，Rescaling 单独证明行归一化，与 zkGPT 的无利融合结论完全一致。

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

### 1.4.1 IPA 协议概述

**IPA（Inner Product Argument）** \[[Bootle et al., 2016](https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21); [Bünz et al., 2018](https://eprint.iacr.org/2017/1066.pdf)\] 是 Bulletproofs 密码学体系中的向量承诺与内积开放协议，工作在**无可信设置（transparent setup）**的离散对数困难性假设下。IPA 与 Sumcheck 协议互补：Sumcheck 将多元多项式的求值归约为对证人的单点查询，IPA 则提供对承诺值的开放证明，使 Verifier 无需访问原始证人即可验证该查询结果。

本文实现以 IPA 作为模型权重矩阵的承诺方案：对每个权重矩阵 $W \in \mathbb{Z}^{m \times n}$，离线计算其多线性扩展（MLE）在生成元集上的承诺 $\mathrm{cm}_W \in \mathbb{G}_1$（BLS12-381）。

> **注**：IPA 是独立的密码学技术，不特定于某一 ZKP 框架。zkLLM 原文（\[Sun et al., CCS 2024\] §3.4）实际采用 Hyrax 的 Pedersen 承诺变体；本文实现改用 Bulletproofs 家族 IPA，同样基于离散对数困难性，但无需可信设置。

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
│       │    L2Norm: 代数约束 r̂²·sq_norm ≈ S_R²·SCALE²（algebraic_v2）          │
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

**为何使用 Sumcheck 而非 IPA？**

MeanPool 是纯线性聚合——对图像侧，在所有图像 token 位置（mask[t]=1）的隐藏状态求均值：

$$\mathbf{p} = \frac{1}{K} \sum_{t:\,\text{mask}[t]=1} H_t \in \mathbb{R}^{d}$$

mask 向量完全公开（由 Tokenizer 产生，Verifier 可自行获取），**无可学习参数**。IPA 权重承诺的设计动机是"Verifier 无法访问模型权重"；MeanPool 的"权重"（mask/K）公开可知，无需密码学承诺保护。因此直接用 Sumcheck 证明聚合求和正确性：

$$\text{Prover 声称：} \sum_{t=0}^{T-1} \text{mask}[t] \cdot H_{t} = K \cdot \mathbf{p}$$

对整个向量用多线性扩展（MLE）一次性证明全部 $d$ 个维度，proof 大小 $O(\log T)$，Prover 复杂度 $O(T \cdot d)$——与 Phase 2 内积 Sumcheck（证明 $\langle \hat{q}, \hat{w} \rangle = s$）的构造完全对称。

**为何不使用 tLookup？**

tLookup 处理的是非线性激活函数（$\exp$、SiLU、GELU 等量化查表场景）。MeanPool 是纯加权求和，无任何非线性，无需查表。

L2Norm（$\mathbf{e} = \mathbf{p}/\|\mathbf{p}\|_2$）采用代数约束方案（`algebraic_constraint_v2`）：Prover 声称 $\hat{r} = \lfloor S_R \cdot 2^{16} / \|\mathbf{p}_\text{int}\|_2 \rceil$（$S_R = 2^{48}$），Verifier 检查 $|\hat{r}^2 \cdot \mathrm{sq\_norm} - S_R^2 \cdot 2^{32}| \leq 2 S_R \cdot 2^{32}$（误差容限约 $2^{97}$，来自 $\hat{r}$ 的单位量化误差）。全量语料库 303 张 PASS（§9.2），当前实现满足代数可验证性。技术上进一步的扩展是接入完整 Sumcheck 协议（与 §1.2.3 RMSNorm Rescaling 一致），但代数约束已覆盖 Verifier 的独立验证需求，详见 §2.6.2 ⑤。

**"Batch"与 Pooling Sumcheck 的关系辨析**

三种"批量"语义不同，不可混淆：

- **Batch Sumcheck**（§2.4.5）：FFN gate/up 共享输入 $X$，随机线性组合将两条 zkip 归约为一条——针对**同一激活共享多个权重矩阵**的优化。
- **Global Batch Sumcheck**（Phase 2）：将 $N$ 个检索内积 $\{s_i\}$ 通过 Fiat-Shamir 聚合为单次 Sumcheck——针对**多条独立查询**的批量化。
- **Pooling Sumcheck**（本节）：单条 Sumcheck 证明 $T$ 个 token 的 mask 加权求和——针对**单一公开权重线性聚合**的直接 Sumcheck 证明，三者适用场景完全不同。

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

**C3b 随机化全组件篡改检测率实验**（v2，覆盖 ViT 图像侧 + LLM 文字侧）：

实验从全部 32 个 ViT 块中随机抽取 8 块、从全部 36 个 LLM 层中随机抽取 12 层，每个块/层随机选取 3 个权重矩阵（从 gate/up/down/q/k/v 共 6 类中任选），在 3 档篡改比例（0.1%/1%/5%，高斯噪声 $\sigma = \text{std}(W)$）× 3 次重复下测试 IPA binding check 检测率，共 20 个组件 × 9 次 trial。

| 维度 | 统计结果 |
|:-----|:-------:|
| 正常权重通过率 | 60/60（100%）|
| **篡改总检出率** | **540/540（100%）**|
| 图像侧（ViT 块，image） | 216/216（100%）|
| 文字侧（LLM 层，text） | 324/324（100%）|
| ratio = 0.001（0.1% 元素篡改） | 180/180（100%）|
| ratio = 0.01（1% 元素篡改） | 180/180（100%）|
| ratio = 0.05（5% 元素篡改） | 180/180（100%）|
| 总耗时 | 2137 s（含证明生成 + IPA 验证）|

即使仅篡改 **0.1%** 的权重元素，IPA binding check 在 ViT 图像侧和 LLM 文字侧全部 540 个 trial 上均以 100% 概率检出，跨模态检测能力完全一致。

**Fiat-Shamir 随机层挑战统计实验**（10,000 次模拟，$N=36$，$K=6$，Python `random.Random(SHA256)`）：

FS-1 单次检出率实测 vs 理论（$M=10000$ 次随机挑战）：

| 篡改层数 $L$ | 实测检出率 | 理论值 | 误差 |
|:-----------:|:---------:|:-----:|:---:|
| 1 | 16.21% | 16.67% | 0.46% |
| 3 | 43.63% | 43.14% | 0.49% |
| **6** | **68.54%** | **69.52%** | **0.98%** |
| 12 | 93.15% | 93.09% | 0.06% |
| 18 | 99.22% | 99.05% | 0.17% |

FS-2 多轮累积检出率（$L=1$ 层被篡改，连续 $T$ 次查询）：

| 轮次 $T$ | 实测累积检出率 | 理论值 |
|:--------:|:-----------:|:-----:|
| 5 | 59.92% | 59.81% |
| 10 | 84.04% | 83.85% |
| 20 | 97.34% | 97.39% |
| 30 | 99.40% | 99.50% |

FS-3 层选择均匀性：$\chi^2=27.35$，$p=0.82 > 0.05$，各层被选中频次均匀（均值 1667 次，$\sigma=35.6$，CV=2.1%）。

**服务器端 nonce 安全分析**：当前实现 nonce 由服务器生成，恶意服务器可枚举 nonce 直至选出不含篡改层的挑战集合。$L=1$ 时期望枚举次数仅 1.2 次，$L=6$ 时 3.3 次，绕过成本极低。完整防御方案为改用客户端 nonce：客户端提交 `(query, client_nonce)` → 服务器以 `SHA256(query+client_nonce)` 生成挑战 → 客户端可独立验证挑战合法性。当前实现采用**半诚实服务器假设**（honest-but-curious），Fiat-Shamir 构造的层选择不可预测性保持不变；如需完整防御恶意服务器，切换至客户端 nonce 即可，无需修改其余 ZK 协议。

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

LoRA \[[Hu et al., 2022](https://arxiv.org/abs/2106.09685)\]（Low-Rank Adaptation）将权重更新分解为低秩矩阵乘 $W_\text{eff} = W_0 + \frac{\alpha}{r}\mathbf{BA}$，理论上每层需要额外两条 Sumcheck 链（$X\mathbf{A}^\top$ 和 $(X\mathbf{A}^\top)\mathbf{B}^\top$），开销约 $2r/d \approx 3\%$。本文采用**量化前合并**方案彻底消除此开销：

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

**全量证明耗时对比**（Phase 5 阶段估算，2-GPU）：

| 组件 | 优化前 | 优化后（verify-ipa + Batch）| 加速比 |
|------|:------:|:-------------------------:|:-----:|
| ViT 32 blocks | 76 min | **~5.3 min** | **14×** |
| LLM 36 layers | ~27 min | **~6.0 min** | **4.5×** |
| PatchMerger | 2.0s | < 0.5s | >4× |
| **总计** | **~103 min** | **~11.3 min**（估算）| **~9×** |

---

### 2.4.7 Phase 6：跨窗口批量证明与全量端到端验证

Phase 6 在 Phase 5（Batch Sumcheck + C++ verify-ipa）基础上，针对 **ViT 窗口注意力批量化**和 **tLookup 跨层融合**做系统优化，并修复了全局注意力块的 softmax 参数选择 bug，实现首次完整端到端真实语料验证。

**优化 1：区分 Window / Full Attention，WIN_SEQ=64 精确分割**

jina-v4 ViT 前 28 块为窗口注意力（每 window 64 patches），后 4 块为全局注意力（blocks 7, 15, 23, 31，真实 2520 tokens padding 至 seq=4096）。`verify_vit.py` 实现精确区分：

- **窗口块（28 块）**：Python 预切分 1024 patches 为 16 组各 64，分别调用 binary（seq_len=64，seq²=4096=2¹²，满足 NTT 约束）
- **全局块（4 块）**：seq_use=4096，seq²=2²⁴，使用 K=3 full-att softmax 参数（bs={256, 2²⁰, 2²⁰}），同时修复了 Phase 5 中因条件分支错误导致的 tLookup 参数退化 bug（bs 第 3 段 B_k 溢出 2⁵⁶ → sumcheck 断言失败）

**优化 2：跨窗口批量 tLookup 证明**

窗口注意力产生的 softmax tLookup 分段（K=4 个 lookup 表），在 `self-attn.cu` 中实现跨窗口融合：循环内仅累积各头的 X_segs / Y_segs，循环结束后一次性调用 `batch_prove_segs()`，将 n_wins × n_heads × K 次单独 tLookup 合并为 K 次。数据规模：D_merged = next_pow2(n_wins × n_heads × seq²)，比逐头分别证明减少约 40× kernel launch 开销。

**优化 3：跨头批量 rescaling tLookup**

每个注意力头输出需 2 次 Rescaling（rs1, rs2）去量化。Phase 6 在头循环外预分配 `all_rems`，累积所有头的 remainder 张量，循环结束后 1 次批量 tLookup 证明替代每头 2 次独立调用：

| 方案 | tLookup 调用次数 | D（合并后） |
|------|:--------------:|:-----------:|
| Phase 5 逐头独立 | 640 次（40 wins × 16 heads） | 32K/次 |
| **Phase 6 批量融合** | **2 次** | **D_merged ≈ 8M** |

**优化 4：CUB DeviceHistogram 替换 atomicAdd**

tLookup "prep" 阶段频次直方图统计改用 CUB `DeviceHistogram::HistogramEven`，D=8M 时从 ~3.5s 降至 ~0.4s。

**优化 5：verify-ipa 批量模式（9 proofs per binary 调用）**

每块 9 个 IPA proof 原需 9 次独立 verify-ipa 调用（各 ~100ms CUDA 初始化开销，合计 ~900ms）。Phase 6 改为批量接口 `./verify-ipa <com_dir> <proof1> [proof2] ...`，一次 CUDA 初始化处理全部 proof，降至 ~150ms/block（~16ms/proof）。

**全量端到端实测（jina-v4，真实语料，双 GPU RTX 4090 D）**：

| 组件 | Phase 5 估算 | Phase 6 实测 | 说明 |
|------|:-----------:|:-----------:|:---:|
| ViT 32 blocks | ~5.3 min | **~3.7 min** | 含 window/full-att 区分 |
| LLM 36 layers（双 GPU）| ~6.0 min | **~5.5 min** | — |
| PatchMerger + Conv3d + Pooling | < 0.5s | < 0.5s | — |
| **总计** | **~11.3 min** | **683s（11.4 min）** | **首次真实语料全量通过** |
| fold check | — | 288/288 PASS | ViT+LLM 全部通过 |
| binding check | — | 288/288 PASS | — |

---

## 2.5 完整覆盖结果

**IPA 验证总计**（全部组件，fold + binding 双重校验）：

| 组件 | 证明类型 | 权重矩阵数 | 结果 |
|------|---------|:---------:|:---:|
| Conv3d embed | IPA | 1 | **1/1 PASS** |
| ViT 32 blocks | IPA × 9/块 | 288 | **288/288 PASS** |
| PatchMerger | IPA × 3 | 3 | **3/3 PASS** |
| Pooling head | MeanPool Sumcheck + L2Norm 代数约束 | — | **PASS** |
| LLM 36 layers（离线）| IPA × 6/层 | 216 | **216/216 PASS** |
| **总计** | — | **508** | **全部通过** |

> **实测全量证明耗时**（Phase 6，jina-v4 真实语料，双 GPU RTX 4090 D）：**683s（11.4 min）**，32/32 ViT blocks + 36/36 LLM layers 全部 PASS（fold\_ok=True，binding\_ok=True）。

### 2.5.1 端到端语料库全量验证（20 张图像）

为验证系统在真实语料规模下的可靠性，从 303 张图像语料库中对 20 张（含 17 张尼康相机产品文档页面 + 3 张 zkLLM 论文页面）运行完整 5 组件证明，测量单 GPU worker 模式下的逐张耗时与各组件通过率。

**实验环境**：RTX 4090 D，jina-embeddings-v4（真实权重，含 LoRA 合并），`script/build_corpus_full_proof.py`，`--num-workers 1`（单 GPU 串行）。

**逐张耗时分布**（ms）：

| 图像 | 总耗时 | ViT 32 块 | PatchMerger | LLM 36 层 |
|------|:-----:|:---------:|:-----------:|:---------:|
| nikon\_page\_12 | 1337.9s | ~796s | 3.2s | ~530s |
| nikon\_page\_48 | 1307.7s | — | 3.1s | — |
| nikon\_page\_54 | 1315.0s | — | 3.1s | — |
| … | … | … | … | … |
| zkLLM\_paper\_p1 | 1379.8s | — | — | — |
| zkLLM\_paper\_p2 | 1369.2s | — | — | — |
| zkLLM\_paper\_p3 | 1376.8s | — | — | — |
| **均值** | **1330.5s（22.2 min）** | ~796s | ~3.2s | ~528s |
| **总 wall clock** | **26609s（7.4h）** | — | — | — |

> zkLLM 论文页面（755 tokens，K=755）略慢于尼康文档页面���641 tokens，K=641），两者 ViT 输入 patch 数相同（均为 2520 tokens pad 至 4096），耗时差异来自 LLM 侧激活文件读写。

**组件通过率汇总**（20 张 × 全组件）：

| 组件 | 通过率 |
|------|:-----:|
| Conv3d embed（IPA） | **20/20 PASS** |
| ViT 32 blocks（IPA × 9/块）| **20 × 32/32 PASS**（共 5760 个 IPA proof） |
| PatchMerger（IPA × 3）| **20/20 PASS** |
| Pooling head（Sumcheck + 代数约束）| **20/20 PASS** |
| LLM 36 layers（IPA × 6/层）| **20 × 36/36 PASS**（共 4320 个 IPA proof） |
| **端到端（all\_ok=True）** | **20/20（100%）** |

**激活捕获耗时**：第一张图（冷启动）约 3038ms，后续图像约 400–500ms（模型已加载于 GPU）。

**全量语料库外推**：单 GPU worker 均值 1330s/图，303 张分配给 2 个 worker（各约 152 张）→ 预计墙钟约 **56h**，与 §2.6.2 ① 估算一致。

**系统全组件覆盖**（三层证明联合）：

| Phase | 防御内容 | 机制 | 检测率 |
|:-----:|---------|------|:------:|
| Phase 1 | 图像来源 + embedding 绑定 | ZAC（Bloom Filter + Pointproofs）| 100%（B1/B2）|
| Phase 2 | 相似度分值 + 排名 | Global Batch Sumcheck | 100%（B3）|
| Phase 3 | 推理权重完整性 | zkLLM IPA + Fiat-Shamir 随机挑战 | 100%（B4，整模型替换）|

### 2.5.2 端到端在线检索实验 E3（真实查询激活 · Fiat-Shamir 随机层挑战）

为验证三阶段联合证明在真实在线查询场景下的端到端表现，设计实验 E3：使用 5 条真实文本查询（中英文混合），对完整可验证检索流水线进行逐查询延迟与验证通过率测量。

**实验配置**：
- 查询数：5 条（4 条英文 + 1 条中文，涵盖传感器、ISO、续航、AF、防抖等主题）
- Fiat-Shamir 参数：$K=6$（从 36 层随机抽取），挑战 = $\mathrm{SHA256}(\text{query}\|\text{nonce})$
- TOP\_K=5，corpus N=303，D=2048
- GPU：双 RTX 4090 D（jina-v4 模型占用 GPU0 ~7.6GB，GPU1 空闲）
- Phase 3Q：层间双 GPU 并行（GPU0:3 层 ‖ GPU1:3 层，各自串行 FFN+Attn）

**延迟分解（均值 ± 标准差，ms，N=5 条 query）**：

| 阶段 | 均值 (ms) | 标准差 (ms) | 说明 |
|------|:---------:|:----------:|------|
| 向量编码 + hook 激活捕获 | **339** | ±138 | jina-v4 前向 + 12 个 hook（K=6 层 × 2 类型）|
| FAISS 检索（CPU） | **1** | ±0.7 | IndexFlatIP，N=303 |
| Phase 2 Sumcheck | **1022** | ±67 | Global Batch IP，N=303，D=2048 |
| Phase 1 ZAC 成员证明 | **9260** | ±104 | k=5 结果聚合验证（Pointproofs） |
| Phase 3C 语料证明读取 | **4** | ±2 | 预计算 JSON 磁盘读取（303/303 已完成）|
| **同步流水线合计** | **10627** | ±198 | 编码→检索→Sumcheck→ZAC→3C 读取 |
| Phase 3Q query proof | **36543** | ±127 | K=6 层，2-GPU 并行，后台异步 |
| **端到端总计（用户感知）** | **36543** | ±127 | $\max(\text{同步}, \text{Phase3Q})$ |

**验证通过率（5 条 query）**：

| 阶段 | 通过率 | 备注 |
|------|:------:|------|
| Phase 2：Global Sumcheck | **5/5（100%）** | 相似度计算可验证 |
| Phase 1：ZAC 成员证明 | **5/5（100%）** | 图像 embedding 绑定 |
| Phase 3C：语料侧 corpus proof | **5/5（100%）** | 303/303 预计算全覆盖，读取 <5ms |
| Phase 3Q：query 激活 Fiat-Shamir | **5/5（100%）** | K=6 随机层，6/6 层全通过 |
| **四阶段联合（Phase1+2+3C+3Q）** | **5/5（100%）** | ALL\_OK=True |

**各查询的 Fiat-Shamir 挑战层选择**（challenge = SHA256(query‖nonce) → sample(range(36), 6)）：

| 查询（摘要） | nonce | 选中层（$K=6$） |
|------------|-------|--------------|
| "sensor resolution of the Nikon Z8" | `8141803b` | 0, 3, 5, 6, 7, 29 |
| "ISO sensitivity in manual exposure mode" | `d7b8c3ef` | 0, 5, 6, 12, 30, 35 |
| "Battery life and USB-C charging specifications" | `a8940890` | 1, 7, 23, 27, 30, 31 |
| "AF tracking for 4K 60fps video" | `3b04b568` | 0, 1, 10, 13, 19, 22 |
| "尼康Z7电子减震功能不可用场景" | `b0c8b9ca` | 3, 11, 18, 26, 28, 33 |

5 次挑战覆盖 36 层中的 22 个不同层（0–35 全范围），分布均匀。

**已实现端到端关键性质**：
1. **真实激活可证明**：hook 捕获的前向激活（非合成数据）经 Fiat-Shamir 挑战的 K=6 层均通过 FFN + Attn-linear + zkAttn 三段 IPA 验证，被选层分布于 layer 0–35 全范围。
2. **语料侧全量覆盖**：303/303 张语料图像预计算证明已完成（双 GPU 并行，~56h），查询时直接读取，延迟 <5ms。
3. **异步流水线**：Phase 3Q（~36.5s）后台运行，与同步流水线（~10.6s）并行，用户感知延迟由 Phase 3Q 主导；同步流水线与 Phase 3Q 重叠执行比率 = 10.6/36.5 ≈ 29%。
4. **GPU 资源隔离**：FAISS 检索采用 CPU 模式（IndexFlatIP，N=303 时 <2ms），避免 FAISS-GPU `StandardGpuResources` 预分配显存池与 zkLLM C++ binary CUDA 初始化冲突（后者在 GPU0 的 IPA down\_proj 公共参数加载需分配 72MB；若 FAISS-GPU 已占用 ~1.5GB 显存，CUDA malloc 失败，子进程 abort → SIGABRT）。

---

## 2.6 设计选项与局限性

### 2.6.1 系统总体优化与适配汇总

本文在 zkLLM 基础框架之上，针对多模态嵌入模型全链路可验证推理提出十项工程优化与架构适配，涵盖三个证明阶段：

| # | 优化/适配 | 面向问题 | 技术方案 | 量化效果 |
|:-:|---------|---------|---------|---------|
| ① | **GQA 转置适配** | jina-v4 LLM 使用 GQA（$n_{kv}=2$，$n_q=16$），原 MHA 代码 KV 广播映射与内存布局均错误 | per-head 显式循环 + KV 转置 + 动态 Rescaling 因子（防 int32 溢出） | LLM 36 层 GQA zkAttn 全量通过（216/216 IPA PASS） |
| ② | **Window Attention NTT 切分** | ViT 前 28 块为窗口注意力（64 patches/window），全局 seq=1024 不满足 NTT 约束 $\text{seq}^2=2^k$ | Python 预切分 1024 patches → 16 组各 64，分别调用 zkAttn（$\text{seq}^2=2^{12}$） | ViT 32/32 块全量证明通过，NTT 约束严格满足 |
| ③ | **CUB DeviceHistogram 并行** | tLookup prep 直方图统计 D=8M 时串行 atomicAdd 耗时 ~3.5s/批 | 替换为 GPU CUB `DeviceHistogram::HistogramEven` 设备级 kernel | ~0.4s/批（~8.75× 加速） |
| ④ | **Batch Sumcheck（gate+up）** | FFN gate/up 共享输入 $X$，独立两次 zkip 重复计算 $X_\text{reduced}$，浪费 ~40% FFN 时间 | Schwartz-Zippel 随机线性组合，共享 $X_\text{reduced}$，合并为一次 zkip | 单层总耗时 −23%（22.5s→17.3s），IPA 验证结果不变 |
| ⑤ | **C++ GPU verify-ipa 批量** | Python py\_ecc G1 乘法 7ms/次，ViT 288 个 proof 验证需 76 分钟 | CUDA GPU 加速 G1 MLE，9 proofs/二进制调用共享一次 CUDA 初始化 | ViT 单块验证 288s→16.7s（17×），全量 103min→11.4min |
| ⑥ | **2-GPU 层间并行** | 各层证明无数据依赖，串行调度 GPU 利用率低 | GPU0：偶数层，GPU1：奇数层，`ProcessPoolExecutor` 并发调度 | K=6 在线证明 89s→~46s；全量 LLM 36 层 ~5.5min |
| ⑦ | **IPA Embedding 承诺** | Phase 2 Verifier 需持有 embedding.npy（2.4MB）且无密码学绑定，Prover 可维护两套 embedding | `commit-param` 离线建库（{$cm_i$}，43KB），`open-ipa` 在线生成 oracle proof | Verifier 持有量 2.4MB→43KB；soundness 误差 $2^{-53}$→$2^{-247}$ |
| ⑧ | **语料侧全量预计算** | 图像编码全量证明（5 组件）生成需 ~22min/张，无法实时响应 | 离线预计算 `corpus_proof`，查询时直接读取 | 语料侧查询延迟 ~4ms（实测 4±2ms，E3）；端到端 e2e ~36.5s（Phase 3Q 主导，同步流水线 ~10.6s 被掩盖） |
| ⑨ | **Fiat-Shamir 随机层挑战** | 固定 $K$ 层时攻击者仅需保持那 $K$ 层权重不变即可绕过检测 | $\text{challenge}=\mathrm{SHA256}(\text{query}\|\text{nonce})$ 非交互随机选 $K=6$ 层 | 整模型替换 100% 确定性检出；单层篡改累积 $T=20$ 次→97.4% |
| ⑩ | **级联双层 BF + Pointproofs** | 单层 Bloom Filter 误判率 $\varepsilon\approx0.01$；Merkle 成员证明大小 $O(\log N)$ | 二级串联 BF（$\varepsilon^2\approx10^{-4}$）+ Pointproofs 聚合（固定大小） | 虚假成员率 $10^{-4}$，成员证明 48B（$O(1)$，不随 $N$ 增长） |

> 说明：①–⑥ 为 Phase 3 zkLLM 工程优化；⑦ 为 Phase 2 Verifier 轻量化；⑧⑨ 为系统架构层设计；⑩ 为 Phase 1 ZAC 密码学构造。各项详细描述见对应章节（§2.4.2–§2.4.7、Phase 1/2 文档）。

### 2.6.2 当前局限性

**① 语料库全量证明时间长**

离线预计算 303 张图像的全量 5 组件证明（Conv3d + ViT 32 块 + PatchMerger + Pooling + LLM 36 层）预计约 56 小时（两 GPU 并行，~22 min/图）。与之相比，LLaMA-2-13B 的论文实现在 A100 上为 803s/次推理，本系统总证明链路更长（多了 ViT 和跨模态桥接层），时间量级合理。实用部署中可通过增加 GPU 数量线性降低建库时间。

**② 在线 K 层证明与全量推理的安全差距**

在线查询仅证明随机选取的 $K=6$ 层，而语料库离线证明为全量 36 层（未来将扩展为全量 5 组件）。$K=6$ 的随机挑战对整模型替换（B4 主要形式）提供 100% 检出率，但对局部单层篡改（$L=1$）检出率仅 16.7%。对安全性要求更高的场景，可调大 $K$（每增加一层约增加 15s）。

**③ [已解决] ViT Window / Full Attention 区分**

Phase 6 已实现精确区分：窗口块（28/32）使用 seq=64（seq²=2¹²），全局块（4/32）使用 seq=4096（seq²=2²⁴），两者均满足 NTT 约束。同时修复了全局块 softmax 参数选择 bug（bs 第 3 段 B_k 溢出导致 tLookup sumcheck 断言失败）。实测全量 32/32 块 PASS，此项局限性已消除。

**④ 权重承诺公开性假设**

IPA 方案要求权重承诺 $\{\mathrm{cm}_{W_i}\}$ 事先通过可信渠道公开发布（类似代码签名）。若承诺本身被伪造或建库时权重与公开承诺不一致，协议安全性无法保证。在实际部署中，承诺文件应由可信第三方（如模型原始发布方）签署并独立分发，与服务商的承诺文件相互印证。

**⑤ L2Norm 代数约束证明（已实现，algebraic\_constraint\_v2）**

原实现仅执行量化误差界检查（$\|\hat{e} - e\|_\infty \leq \delta$），Verifier 依赖 Prover 诚实性。已于 2026-05-04 替换为代数约束方案（参照 §1.2.3 RMSNorm Rescaling 构造）：

设 MeanPool 输出 $p_{\mathrm{int}} \in \mathbb{Z}^d$，令

$$\mathrm{sq\_norm} = \sum_j p_{\mathrm{int}}[j]^2, \quad S_R = 2^{48}, \quad \hat{r} = \left\lfloor \frac{S_R \cdot 2^{16}}{\sqrt{\mathrm{sq\_norm}}} \right\rceil$$

Verifier 检查代数约束：

$$\bigl|\hat{r}^2 \cdot \mathrm{sq\_norm} - S_R^2 \cdot 2^{32}\bigr| \leq 2 S_R \cdot 2^{32}$$

误差容限 $2 S_R \cdot 2^{32} \approx 2^{97}$，来自 $\hat{r}$ 的单位量化误差传播（≤1 unit → 约束误差 $\leq 2 S_R \cdot 2^{16} \cdot \mathrm{sq\_norm}^{1/2}$，量级等价）。

全量语料库补丁（`script/patch_l2norm_proofs.py`，2-GPU 并行）：GPU0 处理 152 张（69s），GPU1 处理 151 张（69s），**303/303 PASS**，`scheme: "algebraic_constraint_v2"` 写入所有 `corpus_proof_*.json`。当前系统对 L2Norm 步骤的验证满足代数可验证性，无需半诚实假设。

**⑥ GPU 资源隔离约束**

在线证明流水线中，FAISS 检索与 zkLLM C++ binary 子进程必须严格隔离 GPU 资源。实验中发现，若使用 FAISS-GPU（`faiss.index_cpu_to_gpu()`），`StandardGpuResources` 会在 GPU0 预分配约 1.5GB 显存内存池；随后 Phase 3Q 的 FFN 子进程在同一 GPU 上加载 down\_proj IPA 公共参数（72MB，$2^{19}$ 个 G1 点）时 CUDA malloc 失败，触发 `abort()` → SIGABRT，导致 GPU0 的所有层验证全部失败。修复方案：N=303 时使用 CPU FAISS（IndexFlatIP，<2ms），彻底消除显存冲突。这一约束在更大语料库（N > $10^5$）场景下需要重新评估：CPU FAISS 线性扫描将不再可行，届时需考虑 FAISS-GPU 与 zkLLM 运行在不同 GPU 上（如 GPU0 用于 zkLLM，GPU1 用于 FAISS）或其他资源隔离方案。

**⑦ 不支持 ANN 索引**

当前系统依赖 FAISS IndexFlatIP（精确搜索），Phase 2 Sumcheck 的正确性依赖精确内积计算。在 $N > 10^6$ 的大规模语料库场景下，需要 HNSW 或 IVF-PQ 等近似最近邻索引，与精确 Sumcheck 存在根本矛盾。未来可探索在 ANN 候选集内部做精确重排序阶段的 Sumcheck（"两阶段验证"），或引入支持近似性误差显式建模的可验证 ANN 框架。

---

## 参考文献

\[Sun et al., 2024\] Haochen Sun, Jason Li, and Hongyang Zhang. zkLLM: Zero Knowledge Proofs for Large Language Models. In *Proceedings of ACM CCS 2024*, pp. 4197–4211. [https://dl.acm.org/doi/10.1145/3658644.3670391](https://dl.acm.org/doi/10.1145/3658644.3670391)（长版：[https://arxiv.org/abs/2404.16109](https://arxiv.org/abs/2404.16109)）

\[Qu et al., 2024\] Wenjie Qu, Yijun Sun, Xuanming Liu, Tao Lu, Yanpei Guo, Kai Chen, and Jiaheng Zhang. zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference. In *USENIX Security 2024*.

\[Bootle et al., 2016\] Jonathan Bootle, Andrea Cerulli, Pyrros Chaidos, Jens Groth, and Christophe Petit. Efficient Zero-Knowledge Arguments for Arithmetic Circuits in the Discrete Log Setting. In *EUROCRYPT 2016*, LNCS 9666:327–357. [https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21](https://link.springer.com/chapter/10.1007/978-3-662-49896-5_21)

\[Bünz et al., 2018\] Benedikt Bünz, Jonathan Bootle, Dan Boneh, Andrew Poelstra, Pieter Wuille, and Greg Maxwell. Bulletproofs: Short Proofs for Confidential Transactions and More. In *IEEE S&P 2018*, pp. 315–334. [https://eprint.iacr.org/2017/1066.pdf](https://eprint.iacr.org/2017/1066.pdf)

\[Hu et al., 2022\] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In *ICLR 2022*. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

\[Ainslie et al., 2023\] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. In *EMNLP 2023*. [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

\[Thaler, 2022\] Justin Thaler. *Proofs, Arguments, and Zero-Knowledge*. Foundations and Trends in Privacy and Security, 2022. [https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

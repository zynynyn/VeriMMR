# Phase 2 — 检索正确性证明

> 章节定位：本章分两部分。第一部分（理论基础）介绍 Sumcheck 协议、Fiat-Shamir 变换与 Schwartz-Zippel 批次证明三个密码学构建块，这些均为已有文献中的成熟结果，本文直接引用，不作贡献声明。第二部分（框架设计）描述本文针对最近邻检索场景设计的 Global Batch Sumcheck 检索证明方案，包含架构设计、安全性分析、量化方案、实验验证与局限性讨论。

---

# 第一部分：理论基础

## 1.1 Sumcheck 协议

### 1.1.1 多线性扩展

设 $f : \{0,1\}^\ell \to \mathbb{F}$ 为定义在布尔超立方体上的函数，其**多线性扩展（Multilinear Extension，MLE）** $\tilde{f} : \mathbb{F}^\ell \to \mathbb{F}$ 是满足 $\tilde{f}(x) = f(x)$ 对所有 $x \in \{0,1\}^\ell$ 成立的唯一多线性多项式 \[[Thaler, 2022](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf), Definition 3.5\]：

$$\tilde{f}(x_1, \ldots, x_\ell) = \sum_{w \in \{0,1\}^\ell} f(w) \cdot \prod_{i=1}^\ell \left[(1-x_i)(1-w_i) + x_i w_i\right]$$

对于 $d$ 维向量 $\mathbf{v} \in \mathbb{F}^d$（$d = 2^\ell$），将其视为函数 $f_\mathbf{v} : \{0,1\}^\ell \to \mathbb{F}$，其 MLE 满足 $\tilde{f}_\mathbf{v}(r_1,\ldots,r_\ell) = \sum_{j \in \{0,1\}^\ell} v_j \cdot \prod_i \beta(r_i, j_i)$，其中 $\beta(r, b) = rb + (1-r)(1-b)$。这一表示使向量内积可被写作域上的多项式求值，为 Sumcheck 协议提供了代数基础。

### 1.1.2 Sumcheck 协议定义

Sumcheck 协议 \[[Lund et al., 1992](https://dl.acm.org/doi/10.1145/146585.146605)\] 是一个用于证明以下断言的交互式证明协议：

$$H = \sum_{x \in \{0,1\}^\ell} g(x_1, \ldots, x_\ell)$$

其中 $g : \mathbb{F}^\ell \to \mathbb{F}$ 为有界次数多项式，$H \in \mathbb{F}$ 为声称的求和值。

对内积场景，取 $g(x) = \tilde{f}_\mathbf{q}(x) \cdot \tilde{f}_\mathbf{v}(x)$，则：

$$H = \sum_{x \in \{0,1\}^\ell} \tilde{f}_\mathbf{q}(x) \cdot \tilde{f}_\mathbf{v}(x) = \sum_{j=0}^{d-1} q_j \cdot v_j = \mathbf{q} \cdot \mathbf{v}$$

即 Sumcheck 证明两个向量的内积等于 $H$。

### 1.1.3 协议交互过程

协议共进行 $\ell$ 轮，每轮 Prover 发送一个单变量多项式，Verifier 返回随机挑战 \[[Thaler, 2022](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf), Protocol 4.1\]：

**第 1 轮**：Prover 发送 $g_1(X_1) = \sum_{x_2,\ldots,x_\ell \in \{0,1\}} g(X_1, x_2, \ldots, x_\ell)$。Verifier 检查 $g_1(0) + g_1(1) \stackrel{?}{=} H$，发送均匀随机挑战 $r_1 \xleftarrow{\$} \mathbb{F}$。

**第 $i$ 轮（$i = 2, \ldots, \ell$）**：Prover 发送 $g_i(X_i) = \sum_{x_{i+1},\ldots,x_\ell \in \{0,1\}} g(r_1, \ldots, r_{i-1}, X_i, x_{i+1}, \ldots, x_\ell)$。Verifier 检查 $g_i(0) + g_i(1) \stackrel{?}{=} g_{i-1}(r_{i-1})$，发送挑战 $r_i \xleftarrow{\$} \mathbb{F}$。

**最终 Oracle 查询**：Verifier 直接计算 $g(r_1, \ldots, r_\ell)$ 并与 $g_\ell(r_\ell)$ 比对。对内积场景，此查询等价于在随机点 $(r_1, \ldots, r_\ell)$ 处分别求值 $\tilde{f}_\mathbf{q}$ 和 $\tilde{f}_\mathbf{v}$，计算代价 $O(d)$。

每轮 $g_i$ 的次数 $\leq 2$，故可用三点值 $(g_i(0), g_i(1), g_i(2))$ 表示，每轮传输 3 个域元素。

### 1.1.4 证明规模与复杂度

| 参数 | 值（$d = 2048$，$\ell = 11$） |
|------|:---------------------------:|
| 轮数 | $\ell = \lceil \log_2 d \rceil = 11$ |
| 每轮传输 | 3 个域元素（24 字节，$p = 2^{61}-1$） |
| **总证明大小** | $\ell \times 3 \times 8 = \mathbf{264}$ **字节** |
| Prover 时间 | $O(d)$ 次域乘法 |
| Verifier 时间 | $O(d)$ 次域乘法（Oracle 查询） |

与 Bulletproofs \[[Bünz et al., 2018](https://eprint.iacr.org/2017/1066.pdf)\] 相比，Sumcheck 的 Prover 时间为 $O(d)$（而非 $O(d \log d)$）且无群运算，在纯 Python 实现下可达毫秒级，具备实用可行性。

### 1.1.5 可靠性

**定理**（Sumcheck 可靠性，\[[Thaler, 2022](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf), Theorem 4.1\]）：*若 $H' \neq H$（Prover 声称错误的求和值），则对 $g$ 的任意每变量次数 $\leq d_i$ 的多项式，Sumcheck 验证通过的概率至多为：*

$$\Pr[\text{Verifier accepts}] \leq \frac{\sum_{i=1}^\ell d_i}{|\mathbb{F}|}$$

对内积场景，$d_i = 2$，$\ell = 11$，$|\mathbb{F}| = p = 2^{61}-1$：

$$\Pr[\text{cheat}] \leq \frac{22}{2^{61}-1} \approx 2^{-56}$$

可靠性误差可忽略不计。

---

## 1.2 Fiat-Shamir 变换

### 1.2.1 从交互式到非交互式

Fiat-Shamir 变换 \[[Fiat & Shamir, 1987](https://link.springer.com/chapter/10.1007/3-540-47721-7_12)\] 将任意满足**公共硬币**（public-coin）性质的交互式证明协议转化为非交互式证明（NIZK），方法是用密码学哈希函数模拟 Verifier 的随机挑战：

$$r_i \leftarrow \mathrm{SHA256}(\text{transcript}_{i-1})$$

其中 $\text{transcript}_{i-1}$ 为第 $i$ 轮之前所有消息的拼接。在随机预言机（Random Oracle）模型下，哈希函数的输出对 Prover 不可预测，等价于真实随机挑战 \[[Bellare & Rogaway, 1993](https://dl.acm.org/doi/10.1145/168588.168596)\]。

Sumcheck 协议是公共硬币协议（Verifier 仅发送均匀随机挑战），因此可直接应用 Fiat-Shamir 变换。变换后：

- Prover 独立生成所有挑战值 $r_1, \ldots, r_\ell$，无需 Verifier 参与
- Prover 输出完整证明 $\pi = (g_1, g_2, \ldots, g_\ell)$（加上承诺 $H$）
- Verifier 仅需 $O(d)$ 时间验证，无需任何交互

### 1.2.2 Fiat-Shamir 在本方案中的应用

在 Global Batch Sumcheck 方案中，Fiat-Shamir 变换被用于两个层次：

1. **Batch 挑战派生**（§1.3）：从 $N$ 个宣告分值派生随机聚合标量 $\rho$，使单次篡改以极高概率被检出；
2. **Sumcheck 轮内挑战派生**：将 11 轮交互展开为非交互，Prover 与 Verifier 独立重现所有 $r_i$，无需通信。

---

## 1.3 Schwartz-Zippel 引理与批次证明

### 1.3.1 Schwartz-Zippel 引理

**引理**（Schwartz-Zippel，\[[Schwartz, 1980](https://www.sciencedirect.com/science/article/pii/0022000080900141); [Zippel, 1979](https://link.springer.com/chapter/10.1007/3-540-09519-5_73)\]）：*设 $f \in \mathbb{F}[x_1, \ldots, x_n]$ 为不恒等于零的多项式，每变量次数之和（总次数）为 $d$。若 $r_1, \ldots, r_n$ 从 $\mathbb{F}$ 中均匀随机抽取，则：*

$$\Pr[f(r_1, \ldots, r_n) = 0] \leq \frac{d}{|\mathbb{F}|}$$

### 1.3.2 批次证明思想

Schwartz-Zippel 引理直接给出了**批次证明**的基础思路：若需验证 $N$ 个独立等式 $a_i = b_i$（$i = 1, \ldots, N$），取随机标量 $\rho \xleftarrow{\$} \mathbb{F}$，只需验证一条线性组合等式：

$$\sum_{i=1}^N \rho^i \cdot a_i \stackrel{?}{=} \sum_{i=1}^N \rho^i \cdot b_i$$

若存在某个 $j$ 使 $a_j \neq b_j$，则等式 $\sum_i \rho^i (a_i - b_i) = 0$ 等价于多项式 $f(\rho) = \sum_i \rho^i (a_i - b_i)$ 在 $\rho$ 处有根。由于 $f$ 次数至多为 $N$（不恒为零），Schwartz-Zippel 引理保证：

$$\Pr_{\rho \xleftarrow{\$} \mathbb{F}}[f(\rho) = 0] \leq \frac{N}{|\mathbb{F}|}$$

在 $|\mathbb{F}| = 2^{61}-1$、$N = 303$ 下：

$$\Pr[\text{cheat passes}] \leq \frac{303}{2^{61}-1} \approx 2^{-53}$$

可靠性误差约为 $2^{-53}$，可忽略不计。这一技术允许将 $N$ 个独立内积证明**批量化为 1 条**，是 Global Batch Sumcheck 方案的核心机制。

---

# 第二部分：框架设计

> 章节定位：以下内容为本文的设计贡献，包含针对最近邻检索场景的 Global Batch Sumcheck 证明方案设计、安全性分析、量化实验与局限性讨论。

---

## 2.1 解决问题：检索过程可靠性

### 2.1.1 威胁模型与审计盲区

Phase 1（ZAC）已证明返回的图像来自已声明语料库，但语料库成员关系的验证并不能约束**检索排名过程**。服务方在排名阶段仍有三类可操作空间：

| 攻击类型（B3） | 攻击者操作 | 用户可见效果 |
|:---:|---|---|
| 分值伪造 | 将特定图像 $i$ 的相似度报告为虚假值 $s'_i \neq q \cdot v_i$ | 低相关图像进入 top-k |
| 结果隐藏 | 实际计算了全部分值，但只向 Verifier 汇报选出的 $k$ 个，刻意隐藏更高分结果 | 最相关结果被压制 |
| 排名调换 | 正确计算分值但声称错误的排名顺序 | 等效于分值伪造 |

上述三类攻击统称为**排名操控（Ranking Manipulation，B3 攻击）**，其共同特征是：攻击者无需修改图像文件或 embedding 向量（因此绕过 Phase 1 的跨层绑定），仅在内积计算或排名报告环节动手脚。攻击对用户**完全不可见**。

### 2.1.2 需证明的两件事

Phase 2 目标是密码学证明以下两条性质：

1. **内积正确性**：每个宣告的相似度分值 $s_i = q \cdot v_i$ 确实是 query 向量与第 $i$ 个语料 embedding 的内积；
2. **全局最优性**：返回的 top-$k$ 确实是全部 $N$ 个结果中分值最高的 $k$ 个，服务方未隐藏任何高分结果。

---

## 2.2 内部设计

### 2.2.1 方案演进：从交互到非交互，从单条到 Batch

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Local 模式（朴素方案，已淘汰）                                           │
│                                                                          │
│  query q ──┐                                                             │
│            ├──► Sumcheck(q, v_1) ──► proof_1   ┐                        │
│  top-k     ├──► Sumcheck(q, v_2) ──► proof_2   ├─► k 条独立证明         │
│  {v_1..vk} ├──►          ···                   │   大小 ≈ k × 264 B     │
│            └──► Sumcheck(q, v_k) ──► proof_k   ┘                        │
│                                                                          │
│  ⚠️ 漏洞：Prover 可以隐藏更高分结果，只提交低分 k 个通过验证             │
└─────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────┐
        │  Global Batch 模式（本文方案）                   │
        │                                                  │
        │  [建库（离线）]                                  │
        │  embedding.npy ──────────────────────────────► corpus {v_1..v_N}
        │                                                  │
        │  [查询（在线，每次检索）]                        │
        │                                                  │
        │  Step 1: Prover 宣告全部 N 个分值               │
        │  s_i = q·v_i,  i=1..N                           │
        │         │                                        │
        │  Step 2: Fiat-Shamir 派生聚合标量                │
        │  ρ ← SHA256(s_1 ‖ s_2 ‖ ··· ‖ s_N) mod p       │
        │         │                                        │
        │  Step 3: 构造聚合向量与目标值                    │
        │  w = Σ ρⁱ·v_i,   s_batch = Σ ρⁱ·s_i           │
        │         │                                        │
        │  Step 4: 单条非交互 Sumcheck                     │
        │  prove:  q·w = s_batch   ──► π（264 字节）       │
        │         │                                        │
        │  Step 5: Verifier 独立排序，自选 top-k            │
        │  sort(s_1..s_N) ──► top-k indices               │
        │                                                  │
        │  总证明大小 = N×8B（分值）+ 264B（Sumcheck）     │
        │            ≈ 2.6 KB（N=303）                     │
        └─────────────────────────────────────────────────┘
```

### 2.2.2 协议形式化

**Prover 侧**（在线，每次检索）：

1. 计算全量分值：$s_i \leftarrow \langle \hat{\mathbf{q}}, \hat{\mathbf{v}}_i \rangle_{\mathbb{Z}_p}$，$i = 1, \ldots, N$（量化内积，见 §2.4）
2. Fiat-Shamir 聚合挑战：$\rho \leftarrow \mathrm{SHA256}(s_1 \| s_2 \| \cdots \| s_N) \bmod p$
3. 构造聚合向量与目标：$\mathbf{w} \leftarrow \sum_{i=1}^N \rho^i \cdot \hat{\mathbf{v}}_i$，$s_{\text{batch}} \leftarrow \sum_{i=1}^N \rho^i \cdot s_i$
4. 对 $\langle \hat{\mathbf{q}}, \mathbf{w} \rangle = s_{\text{batch}}$ 运行 Fiat-Shamir Sumcheck，生成证明 $\pi$
5. 输出 $\mathrm{proof} = (\mathbf{s}, \pi)$，其中 $\mathbf{s} = (s_1, \ldots, s_N)$

**Verifier 侧**（在线，收到 proof 后）：

1. 重建聚合挑战：$\rho \leftarrow \mathrm{SHA256}(s_1 \| \cdots \| s_N) \bmod p$（与 Prover 完全相同，无需信任）
2. 重建聚合向量：$\mathbf{w} \leftarrow \sum_{i=1}^N \rho^i \cdot \hat{\mathbf{v}}_i$（Verifier 持有 embedding.npy）
3. 重建目标：$s_{\text{batch}} \leftarrow \sum_{i=1}^N \rho^i \cdot s_i$
4. 验证 Sumcheck 证明 $\pi$ 对命题 $\langle \hat{\mathbf{q}}, \mathbf{w} \rangle = s_{\text{batch}}$ 是否成立
5. 独立排序 $(s_1, \ldots, s_N)$，选出 top-$k$：Verifier 不依赖 Prover 声称的任何排名

### 2.2.3 全局最优性消除排名漏洞

Local 模式的核心问题是 Verifier 仅看到 Prover 选出的 $k$ 个向量——即使每条证明都正确，Prover 仍可通过选择性披露来规避全局最优性约束。

Global Batch 模式通过以下机制根本性地消除这一漏洞：

- **Prover 必须宣告全部 $N$ 个分值**：$\rho$ 是所有 $N$ 个 $s_i$ 的哈希，若 Prover 隐藏任何一个或调整宣告顺序，$\rho$ 就会改变，进而 $s_{\text{batch}}$ 不一致，Sumcheck 失败；
- **Verifier 自行排序**：top-$k$ 由 Verifier 从 $\mathbf{s}$ 中独立计算，Prover 对排名无任何控制权；
- **两者合并**：Prover 若要欺骗，必须同时使 $\pi$ 通过验证（Sumcheck 可靠性约束）且使 $\rho$ 与错误 $\mathbf{s}$ 一致（Schwartz-Zippel 约束），两个约束同时满足的概率 $\leq N/p \approx 2^{-53}$。

---

## 2.3 安全性分析：B3 检出率

### 2.3.1 理论检出率

对排名操控（B3）攻击，Global Batch Sumcheck 的理论检出率如下：

| 攻击类型 | 检出机制 | 检出概率下界 |
|----------|----------|:-----------:|
| 分值伪造（$s_i' \neq q \cdot v_i$） | $\rho$ 由所有宣告分值派生，修改任意 $s_i$ 导致 $s_\text{batch}$ 不一致，Sumcheck 拒绝 | $\geq 1 - N/p \approx 1 - 2^{-53}$ |
| 结果隐藏（不报高分结果） | Verifier 持有全部 $N$ 个分值，独立排序，Prover 无法干预 | $1$（确定性检出）|
| 排名调换 | Verifier 自行选 top-k，不依赖 Prover 声称排名 | $1$（确定性检出）|

**综合可靠性（Soundness）**：若任意分值 $s_j$ 被篡改，则 Sumcheck 以 $\geq 1 - 22/p \approx 1 - 2^{-56}$（协议内轮次可靠性）且批次聚合以 $\geq 1 - N/p$ 的概率检出，两重保护下综合错误概率：

$$\Pr[\text{cheat undetected}] \leq \frac{N + 22}{p} \approx 2^{-53} \quad (N=303)$$

### 2.3.2 实验验证：B3 攻击检测率

**实验设计**：在 Nikon Z8 语料库（$N = 303$）上，取 10 条 query，每条从非 top-$k$ 池中随机选 5 个 victim 图像，将其分值改为 $\max_j s_j + 1$（确保进入 top-$k$），共 50 次篡改，对每次篡改运行 Global Batch Sumcheck 验证。

**Nikon Z8 语料库结果（non-IPA 与 IPA 双模式）**：

| 模式 | 域 | 测试次数 | 检测次数 | 检测率 | 总耗时 |
|------|:--:|:------:|:------:|:----:|:----:|
| non-IPA（$\mathbb{F}_{2^{61}-1}$） | Mersenne | 50 | 50 | **100%** | ~2s |
| IPA/GPU（BLS12-381 Fr，`open-ipa`） | BLS12-381 | 50 | 50 | **100%** | 53.3s |

两种模式均实现 100% 检测率；IPA/GPU 模式在更强的密码学安全保障（$2^{-247}$ vs $2^{-53}$）下附加约 5.3s/query 的开销。

**多数据集检测率**：

| 数据集 | 语料规模 | B3 检测次数 | 检测率 |
|--------|:-------:|:---------:|:-----:|
| SlideVQA | 10 | 50/50 | **100%** |
| MP-DocVQA | 10 | 50/50 | **100%** |
| ChartQA | 10 | 49/50 | 98%†  |
| InfoVQA | 10 | 50/50 | **100%** |

† ChartQA 49/50：该 query 的相关语料在干净索引中未进入 top-10（召回本身失败），攻击后 FAISS 与 Sumcheck 独立排序结果相同，属边界情形，非机制失效。

**E4 扩展实验（每数据集 60 query × 5 victim = 300 次）**：

| 数据集 | 语料规模 | B3-Ext 检测次数 | 检测率 |
|--------|:-------:|:-------------:|:-----:|
| Nikon Z8 | 303 | 300/300 | **100%** |
| SlideVQA | 1,284 | 300/300 | **100%** |
| MP-DocVQA | 741 | 300/300 | **100%** |
| ChartQA | 500 | 300/300 | **100%** |
| InfoVQA | 459 | 300/300 | **100%** |
| **合计** | — | **1500/1500** | **100%** |

E4 将每数据集查询规模从 10 条扩展至 60 条，覆盖不同文档类型（幻灯片、扫描文档、图表、信息图）与语料规模（303–1284 张）。ChartQA† 边界情形在大样本（300 次）下未复现，确认属小样本偶发现象，非机制性问题。FPR Phase 2 净基准在 5 个数据集共 300 次干净 query 中均为 **0.0%**（即 Sumcheck 对合法排序的误检率为零）。

---

## 2.4 量化方案

### 2.4.1 量化必要性

Sumcheck 工作在有限域 $\mathbb{Z}_p$（整数运算），而 jina-v4 输出的 embedding 向量为 float32 类型。需将向量量化为整数：

$$\hat{q}_j = \mathrm{round}(q_j \times s), \quad \hat{v}_{ij} = \mathrm{round}(v_{ij} \times s)$$

量化引入两个约束：
1. **溢出约束**：整数内积 $\langle \hat{\mathbf{q}}, \hat{\mathbf{v}}_i \rangle = \sum_j \hat{q}_j \hat{v}_{ij}$ 不能超过域模 $p = 2^{61}-1$；
2. **精度约束**：量化误差不能改变 top-$k$ 排名。

### 2.4.2 Scale 参数选择

**溢出分析**：embedding 向量经 L2 归一化（jina-v4 默认输出），$|v_{ij}| \leq 1$，故 $|\hat{v}_{ij}| \leq s$，内积绝对值 $|\langle \hat{\mathbf{q}}, \hat{\mathbf{v}}_i \rangle| \leq s^2 \times d$。溢出条件：

$$s^2 \times d < p = 2^{61}-1$$

| scale $s$ | 溢出上界 $s^2 \times d$（$d=2048$） | 是否安全 | 精度步长 |
|:---:|:---:|:---:|:---:|
| $2^8 = 256$ | $1.34 \times 10^8 \ll p$ | ✅ | $\approx 3.9 \times 10^{-3}$ |
| $\mathbf{2^{16} = 65536}$ | $\mathbf{8.8 \times 10^{12} \ll p}$ | **✅** | $\mathbf{\approx 1.5 \times 10^{-5}}$ |
| $2^{20}$ | $\approx 10^{18} > p$ | ❌ | — |

取 $s = 2^{16}$ 与 zkLLM 的量化方案（Phase 3）统一对齐，精度步长约 $1.5 \times 10^{-5}$（是 $s=256$ 的 263 倍）。

### 2.4.3 量化误差实验（C2）

**实验设计**：对 $N=303$ 张图像的 corpus embedding（$D=2048$ 维），分别计算 float32 内积和量化整数域内积，比较 top-$k$ 排名一致性与绝对误差，测试三类 query 覆盖典型与极端场景：

**scale=65536 结果**：

| Query 类型 | $L_\infty$ 误差 | top-5 排名一致 | top-10 排名一致 |
|-----------|:--------------:|:----------:|:-----------:|
| corpus[0]（高分值场景） | $2.41\times10^{-5}$ | ✅ | ✅ |
| corpus[100] | $1.37\times10^{-5}$ | ✅ | ✅ |
| 随机单位向量（低分值极端场景） | $1.70\times10^{-5}$ | ✅ | ✅ |

**scale 对比**：

| scale | top-5 一致 | top-10 一致 | $\max L_\infty$ | Prove 耗时 |
|:-----:|:--------:|:---------:|:--------:|:---:|
| $256$ | ✅ | ❌（随机向量场景） | $6.34\times10^{-3}$ | 1.1s |
| $\mathbf{65536}$ | **✅** | **✅（全部场景）** | $\mathbf{2.41\times10^{-5}}$ | **1.0s** |

scale=256 在随机单位向量场景（分值绝对值趋近零）下出现 top-10 排名不一致：分值差 $\Delta s \sim 10^{-4}$，量化步长 $\sim 4\times10^{-3}$，量化噪声超过分值差，导致相邻排名翻转。scale=65536 将精度步长降至 $1.5\times10^{-5}$，覆盖全部场景，溢出上界 $8.8\times10^{12} \ll p$，耗时无变化。

---

## 2.5 性能影响

### 2.5.1 端到端延迟分析

**实验条件**：$N=303$（Nikon Z8 语料库），5 条代表性 query × 10 次重复取中位数，RTX 4090 × 2，Python 3.10，Mersenne 域实现（$p = 2^{61}-1$）。

**Phase 2 在完整检索流水线中的延迟占比**：

| 阶段 | 延迟（中位数） | 类型 |
|------|:----------:|:---:|
| jina-v4 编码 | 130ms | 同步 |
| FAISS 检索 | $<$1ms | 同步 |
| **Phase 2 Sumcheck** | **973ms** | **同步** |
| Phase 1 ZAC（$k=5$） | 4,514ms | 同步 |
| **同步验证合计** | **5,618ms** | **1.5× baseline** |

Phase 2 Sumcheck（973ms）在同步验证阶段中约占 17%，相较 Phase 1 ZAC（4,514ms）代价较低，整体不构成流水线瓶颈。

### 2.5.2 可扩展性：复杂度随语料规模的线性增长

**实验设计**：$N \in \{50, 100, 200, 303, 500, 1000\}$，$N \leq 303$ 使用真实 embedding，$N > 303$ 补随机单位向量（仅用于计时）：

| $N$ | Sumcheck 耗时 | ZAC 耗时 |
|:---:|:-----------:|:-------:|
| 50 | 166ms | 4,378ms |
| 100 | 321ms | 4,379ms |
| 200 | 634ms | 4,369ms |
| 303 | 954ms | 4,368ms |
| 500 | 1,583ms | 4,370ms |
| 1,000 | 3,172ms | 4,387ms |

**复杂度验证（log-log 线性拟合）**：

| 组件 | 理论复杂度 | 实测 log-log 斜率 | 结论 |
|------|:--------:|:---------------:|:---:|
| Sumcheck | $O(N \cdot D)$ | **0.986**（$\approx 1.0$） | 精确 $O(N)$ 线性 |
| ZAC | $O(k)$（与 $N$ 无关） | 方差 $< 0.5\%$ | **完全 $O(k)$ 常数** |

$N$ 从 50 增至 1000（$\times 20$），Sumcheck 延迟从 166ms 增至 3,172ms（$\times 19.1$），与理论预测的 $\times 20$ 高度吻合（斜率 0.986），$O(N \cdot D)$ 复杂度验证成立。

### 2.5.3 排名一致性总结

量化误差对实际语义检索的排名一致性影响如下：

- top-5 排名：全部 query 类型（高分值、中分值、随机向量）**100% 与 float32 一致**
- top-10 排名：语义 query 场景（corpus[0], corpus[100]）**100% 一致**；随机向量极端场景在 scale=256 下有翻转，scale=65536 后**完全消除**
- 实测 $L_\infty$ 误差：$\leq 2.41 \times 10^{-5}$，远低于语义相关分值差（通常 $\sim 10^{-2}$）

---

## 2.6 设计选项与局限性

### 2.6.1 Verifier 需持有语料库副本（已通过 §2.6.3 IPA 承诺解决）

**原始限制**：Verifier 在重建聚合向量 $\mathbf{w} = \sum_i \rho^i \hat{\mathbf{v}}_i$ 时，需本地持有完整的 embedding 矩阵（$N \times D \times 4$ 字节，$N=303$ 时约 2.4 MB，$N=10^4$ 时约 80 MB）。这意味着：

- Verifier 实际上拥有全部语料 embedding，但 Sumcheck 本身不证明这些向量的**来源正确性**（即"$v_i$ 确实由 jina-v4 对 $I_i$ 正确推理得到"），该问题由 Phase 3 zkLLM 预计算证明覆盖；
- 更严重的是，若 Prover 恶意维护"两份 embedding"——一份用于骗过 Phase 2 Sumcheck，另一份实际用于检索——信任链在 Phase 2 与 Phase 3 之间断裂；
- 在 Verifier 资源受限或通信带宽受限的场景下，持有 80MB embedding 是显著的部署障碍。

**已实现的修复（§2.6.3）**：本系统已为每个 embedding 向量生成 IPA 向量承诺 $\mathrm{cm}_i$，Verifier 仅需持有承诺集合（$N \times 144$ 字节 $\approx$ 43KB），无需原始向量，Sumcheck oracle 查询通过密码学绑定验证。详见 §2.6.3。

### 2.6.2 KZG 承诺方案（已被 IPA 方案替代，未单独实现）

KZG 多项式承诺 \[[Kate et al., 2010](https://link.springer.com/chapter/10.1007/978-3-642-17373-8_11)\] 可从根本上消除 Verifier 持有 embedding 副本的假设：

**方案**：为每个 $\hat{\mathbf{v}}_i$ 离线计算 KZG 承诺 $\mathrm{cm}_i \in \mathbb{G}_1$（48 字节），公开发布承诺集合 $\{\mathrm{cm}_i\}_{i=1}^N$。在线验证时，Verifier 利用 $\mathbb{G}_1$ 运算的同态性直接聚合：

$$\mathrm{cm}_{\mathbf{w}} = \sum_{i=1}^N \rho^i \cdot \mathrm{cm}_i \in \mathbb{G}_1 \quad \text{（} N \text{ 次 G1 标量乘法）}$$

无需访问任何 $\hat{\mathbf{v}}_i$ 本身，附加一次配对验证即可完成 Oracle 查询。Verifier 存储从 $O(N \cdot D)$ 降至 $O(N)$（48 字节 $\times N$）。

**未实现原因（工程约束）**：

| 环节 | 当前方案 | KZG 方案 | 差值 |
|------|:-------:|:-------:|:---:|
| 离线承诺建库 | 无额外开销 | $N \times D$ 次 G1 标量乘法 $\approx$ **1,329s**（py_ecc，$N=303$）| 增加 22 分钟 |
| 在线 Verifier | 442ms（域运算） | 649ms（$N$ 次 G1 运算） | **反而更慢** |

BLS12-381 G1 标量乘法在纯 Python（py_ecc）实现下比 Mersenne 域运算慢约 3000×：$N=303$ 时 Verifier 时间 649ms 反而比当前方案 442ms 更慢。KZG 的真实价值体现于 $N \gg 10^4$ 的大规模场景（域运算扩展快于 G1 运算），或有 GPU 加速的配对运算环境。

KZG 与 Pointproofs（Phase 1 所用）共享 BLS12-381 底层基础设施，技术可行性已具备，留作大规模场景（$N \gg 10^4$）的后续优化方向。当前系统已选用 IPA 向量承诺方案解决 Verifier 存储问题，见 §2.6.3。

---

### 2.6.3 IPA Embedding 承诺方案（已实现）

本系统已实现基于 Pedersen 向量承诺的 IPA（Inner Product Argument）方案，彻底消除 Verifier 持有原始 embedding 矩阵的假设。

#### 密码学构建块

**Pedersen 向量承诺**：给定公开参数 $\mathbf{G} = (G_1, \ldots, G_D) \in \mathbb{G}_1^D$（随机生成的 BLS12-381 G1 点），对量化向量 $\hat{\mathbf{v}}_i \in \mathbb{Z}^D$ 的承诺为：

$$\mathrm{cm}_i = \sum_{j=1}^D \hat{v}_{ij} \cdot G_j \in \mathbb{G}_1$$

承诺满足 **Binding 性质**（计算意义下）：在离散对数困难假设下，不存在有效算法使同一 $\mathrm{cm}_i$ 对应两个不同的向量。

**IPA Oracle Opening**：对聚合向量 $\mathbf{w} = \sum_i \rho^i \hat{\mathbf{v}}_i \in \mathbb{Z}_p^D$，IPA 协议在 $\ell = \lceil \log_2 D \rceil$ 轮内证明：

$$\mathbf{w} \text{ 的 MLE 在 Sumcheck 挑战点 } (r_1, \ldots, r_\ell) \text{ 处的求值}$$

与承诺 $\mathrm{cm}_{\mathbf{w}} = \sum_i \rho^i \cdot \mathrm{cm}_i$ 密码学一致。

#### 协议架构

**离线阶段（一次性，建库）**：

1. 生成公开参数：`ppgen D embedding-pp.bin` → $D$ 个 BLS12-381 G1 随机生成器（CPU 模式：`generate_random_pp_python(D, seed=42)`）
2. 对每个 $\hat{\mathbf{v}}_i$：量化为 int32 → `commit-param` → 144 字节 Jacobian G1 点 $\mathrm{cm}_i$
3. 输出 `embedding_commitments.bin`（$N \times 144$ 字节）

**在线 Prove 阶段（每次检索，新增步骤）**：

在原有 Global Batch Sumcheck 证明（§2.2.2 Step 1–4）的基础上，Prover 执行第 5 步：

5. 调用 `open-ipa pp.bin w.bin u.bin oracle_proof.bin D`：将 Sumcheck 挑战向量 $\mathbf{r} = (r_1, \ldots, r_\ell)$ 与聚合向量 $\mathbf{w}$ 传入，生成 IPA Oracle opening proof $\pi_{\mathrm{ipa}}$

**在线 Verify 阶段（Verifier 新增步骤）**：

原来"重建聚合向量 $\mathbf{w}$"的步骤（需要原始 embedding.npy）替换为：

5a. 加载承诺集合 $\{\mathrm{cm}_i\}$（43 KB），在 BLS12-381 G1 上聚合：
$$\mathrm{cm}_{\mathbf{w}} = \sum_{i=1}^N \rho^i \cdot \mathrm{cm}_i \in \mathbb{G}_1$$

5b. 调用 `verify_ipa_embedding(π_ipa, cm_w)` 执行两项检查：
- **Binding check**：$C_{\mathrm{init}} \stackrel{?}{=} \mathrm{cm}_{\mathbf{w}}$（IPA proof 内嵌的初始承诺与 Verifier 聚合结果一致）
- **Fold check**：$\ell$ 轮 IPA 折叠链验证，确认 $w_{\mathrm{final}} = \tilde{\mathbf{w}}(r_1, \ldots, r_\ell)$

5c. 用 $w_{\mathrm{final}}$ 完成 Sumcheck 最终 oracle 验证（替代原来的 `q_final × w_final_py`）

#### 证明规模与安全参数

| 参数 | 值（$D=2048$，$\ell=11$，$N=303$） |
|------|:----------------------------------:|
| IPA proof 结构 | header（12B）+ $C_{\mathrm{init}}$（144B）+ $\ell \times 32$B $u_{\mathrm{in}}$ + $\ell \times 2 \times 144$B 折叠轮 + $g_{\mathrm{final}}$（144B）+ $w_{\mathrm{final}}$（32B）|
| **Oracle proof 大小** | $12 + 144 + 11\times32 + 11\times2\times144 + 144 + 32 = \mathbf{3{,}852}$ **字节** |
| **Verifier 存储** | $\{\mathrm{cm}_i\}$：$303 \times 144 = 43{,}632$ **字节（43 KB）** |
| 非 IPA 模式 Verifier 存储 | `embedding.npy`：$303 \times 2048 \times 4 = 2{,}482{,}176$ **字节（2.4 MB）** |
| 存储压缩比 | **57×** |

#### 安全性分析

IPA 方案将 Sumcheck oracle 查询的安全基础从 Mersenne 域的有限域可靠性提升至 BLS12-381 Fr 域（$|\mathbb{F}| = p_{\mathrm{FR}} \approx 2^{255}$）的离散对数困难假设：

| 安全参数 | 非 IPA 模式（$\mathbb{F}_{2^{61}-1}$） | IPA 模式（BLS12-381 Fr，$p_{\mathrm{FR}} \approx 2^{255}$） |
|---------|:-----------------------------------:|:----------------------------------------------------------:|
| Oracle 欺骗概率（单次） | $\leq N / (2^{61}-1) \approx 2^{-53}$ | $\leq N / p_{\mathrm{FR}} \approx 2^{-247}$ |
| 安全基础 | 有限域 Schwartz-Zippel | BLS12-381 Fr 离散对数困难 |
| Verifier 信任假设 | 信任 embedding.npy 原始矩阵 | 密码学绑定（binding check） |
| Phase 3 绑定 | 无（信任链在 Phase 2–3 间断开） | cm_i 与 Phase 3 承诺结构共享，可扩展绑定 |

**综合 soundness 误差**（IPA 模式）：

$$\Pr[\text{cheat undetected}] \leq \frac{N + \ell_{\mathrm{SC}} + \ell_{\mathrm{IPA}}}{p_{\mathrm{FR}}} \approx \frac{303 + 11 + 11}{2^{255}} \approx 2^{-247}$$

其中 $\ell_{\mathrm{SC}} = 11$（Sumcheck 轮次）、$\ell_{\mathrm{IPA}} = 11$（IPA 折叠轮次）。

#### 性能开销（CPU 实现，py_ecc）

IPA 模式新增如下在线开销：

| 新增步骤 | 操作 | 测量耗时（CPU，$N=303$，$D=2048$） |
|---------|------|:----------------------------------:|
| Prover：oracle proof 生成 | `ipa_prove_python`（$\approx 8192$ 次 G1 scalar mul） | $\approx 108$s |
| Verifier：$\mathrm{cm}_{\mathbf{w}}$ 聚合 | $N$ 次 G1 scalar mul | $\approx 3$s |
| Verifier：IPA fold check | $\ell$ 轮 × 3 次 G1 操作 | $\approx 0.3$s |
| **Verifier 新增合计** | — | **$\approx 3.3$s** |

注：GPU 模式（`open-ipa` C++ binary）oracle proof 生成降至 $\approx 2$s（实测），Verifier cm_w 聚合约 3s（py_ecc Python，若用 zkLLM G1 kernel 可降至 $<0.1$s）。实测 GPU 模式总开销约 $5.3$s/query（10 条 query 合计 53.3s）；CPU 模式总开销约 $111$s/query。Phase 2 非 IPA 基准为 973ms/query（§2.5.1），IPA 模式以额外时间换取 $2^{194}$ 倍安全性提升。

#### 实验验证（C2b：IPA 模式正确性；B3：IPA 攻击检测）

**C2b 测试设计**：取 $N=10$ 条真实 jina-v4 embedding（从 `embedding.npy` 中读取，$D=2048$），CPU-only 模式（`generate_random_pp_python` + `ipa_prove_python`），对 5 条 query 执行完整 IPA 验证流程。

| 测试项目 | 结果 |
|---------|:----:|
| Sumcheck oracle 正确性（`oracle_ok`） | ✅ True |
| IPA fold check 通过 | ✅ True |
| IPA binding check 通过 | ✅ True |
| top-5 结果与非 IPA 模式一致 | ✅ True |
| 篡改检测（修改 $s_i$ → Sumcheck 失败） | ✅ True |

**B3-IPA 全语料实验**：在 Nikon Z8 全语料库（$N=303$）上以 GPU IPA 模式（`open-ipa` binary + `embedding_commitments.bin`）运行 B3 排名操控攻击检测实验（10 条 query × 5 个篡改位置 = 50 次）：50/50 = **100% 检测率**，总耗时 53.3s（$\approx 5.3$s/query）。

---

## 参考文献

\[Lund et al., 1992\] Carsten Lund, Lance Fortnow, Howard Karloff, and Noam Nisan. Algebraic Methods for Interactive Proof Systems. *Journal of the ACM*, 39(4):859–868, 1992. [https://dl.acm.org/doi/10.1145/146585.146605](https://dl.acm.org/doi/10.1145/146585.146605)

\[Thaler, 2022\] Justin Thaler. *Proofs, Arguments, and Zero-Knowledge*. Foundations and Trends in Privacy and Security, 2022. [https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf)

\[Fiat & Shamir, 1987\] Amos Fiat and Adi Shamir. How to Prove Yourself: Practical Solutions to Identification and Signature Problems. In *CRYPTO 1986*, LNCS 263:186–194. [https://link.springer.com/chapter/10.1007/3-540-47721-7_12](https://link.springer.com/chapter/10.1007/3-540-47721-7_12)

\[Bellare & Rogaway, 1993\] Mihir Bellare and Phillip Rogaway. Random Oracles are Practical: A Paradigm for Designing Efficient Protocols. In *ACM CCS 1993*, pp. 62–73. [https://dl.acm.org/doi/10.1145/168588.168596](https://dl.acm.org/doi/10.1145/168588.168596)

\[Schwartz, 1980\] Jacob T. Schwartz. Fast Probabilistic Algorithms for Verification of Polynomial Identities. *Journal of the ACM*, 27(4):701–717, 1980. [https://dl.acm.org/doi/10.1145/322217.322225](https://dl.acm.org/doi/10.1145/322217.322225)

\[Zippel, 1979\] Richard Zippel. Probabilistic Algorithms for Sparse Polynomials. In *EUROSAM 1979*, LNCS 72:216–226. [https://link.springer.com/chapter/10.1007/3-540-09519-5_73](https://link.springer.com/chapter/10.1007/3-540-09519-5_73)

\[Bünz et al., 2018\] Benedikt Bünz, Jonathan Bootle, Dan Boneh, Andrew Poelstra, Pieter Wuille, and Greg Maxwell. Bulletproofs: Short Proofs for Confidential Transactions and More. In *IEEE S&P 2018*, pp. 315–334. [https://eprint.iacr.org/2017/1066.pdf](https://eprint.iacr.org/2017/1066.pdf)

\[Kate et al., 2010\] Aniket Kate, Gregory M. Zaverucha, and Ian Goldberg. Constant-Size Commitments to Polynomials and Their Applications. In *ASIACRYPT 2010*, LNCS 6477:177–194. [https://link.springer.com/chapter/10.1007/978-3-642-17373-8_11](https://link.springer.com/chapter/10.1007/978-3-642-17373-8_11)

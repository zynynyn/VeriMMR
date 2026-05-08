# 第X章 ZAC 理论基础

> 章节定位：本章介绍 Phase 1 所依赖的密码学构建块——Bloom Filter、Pointproofs 向量承诺方案，以及将二者组合形成的 ZAC 方案。理论内容严格参照原始文献，重点阐明各方案的算法构造与安全属性，为第 Y 章的适配设计提供依据。

---

## X.1 密码学累加器概述

密码学累加器（Cryptographic Accumulator）是一类允许证明者对任意集合生成紧凑承诺，并为集合中任意元素生成简短成员证明的方案 \[Benaloh & de Mare, 1993\]。形式上，一个动态通用累加器方案包含九个算法，满足完备性（Completeness）、可靠性（Soundness）和动态更新性质 \[Dang et al., 2022\]。

**动态通用累加器的核心性质：**
- **完备性**：合法成员的证明必然通过验证；
- **可靠性**：不在集合中的元素以可忽略（或有界）概率通过验证；
- **通用性**（Universal）：既支持成员证明，也支持非成员证明；
- **动态性**（Dynamic）：添加或删除元素时承诺与证明以常数代价更新。

在大规模语料库场景下，直接使用 Merkle 树实现成员证明时，单批次 $k$ 个元素的证明大小为 $O(k \log N)$，其中 $N$ 为语料库规模。本文采用 ZAC 方案 \[Dang et al., 2022\]，将成员证明大小降至 $O(1)$，与 $k$ 和 $N$ 均无关。

---

## X.2 Bloom Filter

### X.2.1 数据结构定义

Bloom Filter（BF）\[Bloom, 1970\] 是一种空间高效的概率型集合表示数据结构。给定集合 $\mathcal{S}$，BF 将其编码为长度 $q$ 的二进制向量 $\mathbf{v} \in \{0,1\}^q$，支持高效的近似成员查询。

形式上，一个 Bloom Filter 方案由四个算法组成 \[Dang et al., 2022\]：

$$
\mathrm{BF} = (\mathrm{Init},\, \mathrm{Gen},\, \mathrm{Update},\, \mathrm{Check})
$$

- $H \leftarrow \mathrm{BF.Init}(q, k)$：给定参数 $q$（位数组长度）和 $k$（哈希函数数量），输出 $k$ 个哈希函数 $H := (h_1, \ldots, h_k)$，其中 $h_i : \{0,1\}^* \to [q]$；

- $\mathbf{m} \leftarrow \mathrm{BF.Gen}(\mathcal{S})$：给定集合 $\mathcal{S}$，初始化 $\mathbf{m} = \mathbf{0}^q$，对每个 $s \in \mathcal{S}$ 将位置 $h_1(s), \ldots, h_k(s)$ 置 1，输出位向量 $\mathbf{m}$；

- $\mathbf{m}' \leftarrow \mathrm{BF.Update}(\mathbf{m}, \mathcal{S}, \mathcal{S}')$：对集合差 $\mathcal{D} = \mathcal{S}' \setminus \mathcal{S}$，将 $h_i(d)$ 的值从 $1 - \mathbf{m}[h_i(d)]$ 更新；

- $b \leftarrow \mathrm{BF.Check}(\mathbf{m}, s)$：输出 $b = \bigwedge_{i \in [k]} \mathbf{m}[h_i(s)]$，若 $s \notin \mathcal{S}$ 则 $b = 0$，若 $s \in \mathcal{S}$ 则 $b = 1$。

### X.2.2 假阳性率与最优参数

BF 以严格零假阴性为代价，允许一定概率的假阳性（False Positive）。若集合大小为 $N$，位数组长度为 $q$，哈希函数数量为 $k$，则假阳性率（FPR）为 \[Dang et al., 2022, Eq. (2)\]：

$$
\Pr_{\mathrm{FP}} = \left(1 - \left(1 - \frac{1}{q}\right)^{kN}\right)^k \approx \left(1 - e^{-kN/q}\right)^k
$$

给定集合大小上界 $N$ 和目标假阳性率 $\varepsilon$，最优参数为：

$$
q = \left\lceil \frac{-N \ln \varepsilon}{\ln^2 2} \right\rceil, \qquad k = \left\lceil \frac{q}{N} \ln 2 \right\rceil
$$

**参数示例**（$\varepsilon = 0.01$，$N = 303$）：$q = 2905$，$k = 7$。位数组长度约为每个元素 9.6 位，远小于直接存储元素所需的空间。

### X.2.3 BF 的关键属性

对 ZAC 方案而言，BF 有两个至关重要的属性：

1. **单向性**：哈希函数的单向性使攻击者无法从 $\mathbf{m}$ 中反推集合元素，保证零知识性；
2. **通用性**：BF 支持对任意集合（不限于素数或特殊代数结构）进行编码，不同于基于 RSA 的累加器；
3. **非成员证明的简洁性**：若某元素对应的 $k$ 个位位置中至少有一位为 0，则可确定性地证明其不在集合中（概率为 1）。

---

## X.3 Pointproofs 向量承诺方案

### X.3.1 向量承诺概念

向量承诺（Vector Commitment，VC）\[Catalano & Fiore, 2013\] 是一种允许证明者承诺向量 $\mathbf{m} = (m_1, \ldots, m_q)$，并在之后为任意位置 $i$ 的值 $m_i$ 生成简短证明的方案。VC 必须满足：

- **隐藏性**（Hiding）：承诺不泄露关于 $\mathbf{m}$ 的信息；
- **绑定性**（Binding/Position Binding）：不能将位置 $i$ 同时证明为两个不同的值。

ZAC 使用的 Pointproofs \[Gorbunov et al., CCS 2020\] 是一种具备**聚合**（Aggregation）性质的零知识向量承诺方案，记为 zkAVC。其关键特性是：可将同一承诺的多个位置证明，或不同承诺的多个证明，聚合为**单个 $\mathbb{G}_1$ 点**（48 字节）。

### X.3.2 算法构造

Pointproofs 建立在椭圆曲线 BLS12-381 上，其双线性群满足 $e : \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$。公共参数由可信设置生成：随机选取陷门 $\alpha \xleftarrow{\$} \mathbb{Z}_p$，计算：

$$
\mathcal{P} = \left(g_1^{\alpha}, \ldots, g_1^{\alpha^q},\, g_1^{\alpha^{q+2}}, \ldots, g_1^{\alpha^{2q}}\right), \qquad
\mathcal{V} = \left(g_2^{\alpha}, \ldots, g_2^{\alpha^q}\right)
$$

其中 $g_1^{\alpha^{q+1}}$ 故意不包含在 CRS 中，以保证绑定性。

**Commit**（生成承诺）：对向量 $\mathbf{m} \in \mathcal{M}^{q-1}$ 和随机数 $r \xleftarrow{\$} \mathbb{Z}_p$，承诺为：

$$
\mathrm{cm} = g_1^{r \cdot \alpha^q} \cdot \prod_{i=1}^{q-1} \left(g_1^{\alpha^i}\right)^{m_i} \in \mathbb{G}_1
$$

**Prove**（生成单元素证明）：对位置 $i \in [q-1]$ 生成证明 $\pi_i$：

$$
\pi_i = \prod_{\substack{j=1 \\ j \neq i}}^{q} g_1^{m'_j \cdot \alpha^{q+1-i+j}} \in \mathbb{G}_1
$$

其中 $\mathbf{m}' = (m_1, \ldots, m_{q-1}, r)$。

**Aggregate**（聚合多个证明）：给定承诺 $\mathrm{cm}$、位置集合 $\mathcal{I} \subseteq [q-1]$ 及对应证明 $\{\pi_i : i \in \mathcal{I}\}$，聚合证明为：

$$
\hat{\pi} = \prod_{i \in \mathcal{I}} \pi_i^{t_i}, \qquad t_i = H'\!\left(i,\, \mathrm{cm},\, \mathcal{I},\, \mathbf{m}[\mathcal{I}]\right)
$$

其中 $t_i$ 为 Fiat-Shamir 挑战值，$\hat{\pi} \in \mathbb{G}_1$ 为单个 48 字节 $\mathbb{G}_1$ 点。

**Verify**（验证）：验证以下双线性配对方程是否成立：

$$
e\!\left(\mathrm{cm},\, \sum_{i \in \mathcal{I}} t_i \cdot g_2^{\alpha^{q+1-i}}\right) \stackrel{?}{=} e\!\left(\hat{\pi},\, g_2\right) \cdot g_T^{\alpha^{q+1} \cdot \sum_{i \in \mathcal{I}} m_i t_i}
$$

其中 $g_T^{\alpha^{q+1}} = e(g_2^{\alpha^q},\, g_1^{\alpha})$ 可在公共参数中预计算。

### X.3.3 复杂度分析

| 操作 | 复杂度 |
|------|--------|
| Commit | $O(q)$ 次 $\mathbb{G}_1$ 标量乘法 |
| Prove（单元素） | $O(q)$ 次 $\mathbb{G}_1$ 标量乘法 |
| Aggregate | $O(\lvert\mathcal{I}\rvert)$ 次 $\mathbb{G}_1$ 标量乘法 |
| Verify | $O(\lvert\mathcal{I}\rvert)$ 次 $\mathbb{G}_2$ 标量乘法 + 2 次配对 |
| **证明大小** | **1 个 $\mathbb{G}_1$ 点 = 48 字节（常数）** |

---

## X.4 ZAC 方案

### X.4.1 设计思路

ZAC（Zero-knowledge Dynamic Universal Accumulator）\[Dang et al., TPS-ISA 2022\] 的核心观察是：BF 将任意集合的成员查询转化为对固定长度二进制向量的位置查询；而 zkAVC（Pointproofs）可以在不泄露向量内容的条件下，证明向量中指定位置的值，并将多个位置的证明聚合为常数大小。

将两者结合，即可实现对任意集合的成员证明：
1. 用 BF 将集合 $\mathcal{S}$ 编码为二进制向量 $\mathbf{v}$；
2. 用 Pointproofs 对 $\mathbf{v}$ 生成承诺 $\mathrm{cm}$（即 ZAC Root）；
3. 成员证明即证明"$s$ 的所有 $k$ 个 BF 哈希位置在 $\mathbf{v}$ 中均为 1"。

### X.4.2 算法构造

以下算法描述直接引自 Dang et al. \[2022\]：

**ZAC.Init**$(1^\lambda, N)$：给定安全参数 $\lambda$ 和集合大小上界 $N$，计算 $q, k$（由 $N, \varepsilon$ 确定），调用 $\mathrm{pp}' \leftarrow \mathrm{PR.Init}(1^\lambda, q)$，初始化 BF：$H \leftarrow \mathrm{BF.Init}(q-1, k)$，输出公共参数 $\mathrm{pp} = (\mathrm{pp}', H, q, k) = \mathbf{0}$。

**ZAC.Com**$(\mathcal{S}, r, \mathrm{pp})$：
1. $\mathbf{v} := (v_1, \ldots, v_{q-1}) \leftarrow \mathrm{BF.Gen}(\mathcal{S})$
2. $\mathrm{cm} \leftarrow \mathrm{PR.Commit}(\mathbf{v}, r, \mathrm{pp})$

**ZAC.ProveM**$(\mathrm{cm}, \mathcal{S}, \hat{\mathcal{S}}, r, \mathrm{pp})$，$\hat{\mathcal{S}} \subseteq \mathcal{S}$：
1. $\mathcal{I} \leftarrow \bigcup_{\hat{s} \in \hat{\mathcal{S}}} \{h_1(\hat{s}), \ldots, h_k(\hat{s})\}$（$\hat{\mathcal{S}}$ 中所有元素的 BF 位置集合）
2. $\mathbf{v} \leftarrow \mathrm{BF.Gen}(\mathcal{S})$
3. 对每个 $i \in \mathcal{I}$，计算 $\pi_i \leftarrow \mathrm{PR.Prove}(i, \mathbf{v}, r)$
4. $\hat{\pi} \leftarrow \mathrm{PR.Aggregate}(\mathrm{cm}, \mathcal{I}, \mathbf{v}[\mathcal{I}], \{\pi_i : i \in \mathcal{I}\})$
5. 输出 $(\hat{\pi}, \mathcal{I})$

**ZAC.VerifyM**$(\mathrm{cm}, \hat{\mathcal{S}}, \hat{\pi}, \mathrm{pp})$：
1. $\mathcal{I} \leftarrow \bigcup_{\hat{s} \in \hat{\mathcal{S}}} \{h_1(\hat{s}), \ldots, h_k(\hat{s})\}$
2. $\mathbf{v} := (v_1, \ldots, v_q) = \mathbf{0}^q$，令 $v_i = 1$ 对所有 $i \in \mathcal{I}$（成员假设）
3. 从 $\hat{\pi}$ 中提取，调用 $b \leftarrow \mathrm{PR.Verify}(\mathrm{cm}, \mathcal{I}, \mathbf{v}[\mathcal{I}], \hat{\pi})$
4. 输出 $b$

### X.4.3 安全性

**定理 1**（Dang et al. \[2022\]，Theorem 1）：*ZAC 构成一个具备完备性、成员证明 $\varepsilon$-可靠性和非成员证明 $\varepsilon$-可靠性的零知识动态通用累加器，其安全性在 Bloom Filter 和向量承诺方案的正确性下成立。*

三条核心性质的直觉：

- **完备性**：对任意 $\hat{\mathcal{S}} \subseteq \mathcal{S}$，$\mathrm{BF.Gen}(\mathcal{S})$ 保证 $\mathcal{I}$ 中所有位置在 $\mathbf{v}$ 中均为 1，Pointproofs 的完备性确保 $\mathrm{PR.Verify} = 1$，故 $\Pr[\mathrm{VerifyM} = 1] = 1$。

- **$\varepsilon$-可靠性（成员证明）**：若 $\hat{\mathcal{S}} \not\subseteq \mathcal{S}$，存在某 $\hat{s} \notin \mathcal{S}$，则 $\hat{s}$ 的某个 BF 位置在 $\mathbf{v}$ 中为 0。事件"$\hat{s}$ 的所有 $k$ 个位置均恰好为 1"（即 BF 假阳性）以概率 $\varepsilon$ 发生（由 BF 的 FPR 设计）。当该事件不发生时，$\mathbf{v}[\mathcal{I}]$ 中存在 0，而验证者假设全为 1，Pointproofs 的绑定性保证此时验证失败（概率 1）。故 $\Pr[\mathrm{VerifyM} = 1 \mid \hat{\mathcal{S}} \not\subseteq \mathcal{S}] \leq \varepsilon$。

- **零知识性**：$\mathbf{v}$ 经 Pointproofs 承诺，只有承诺值 $\mathrm{cm}$（一个 $\mathbb{G}_1$ 点）公开；随机数 $r$ 保证 $\mathrm{cm}$ 与 $\mathbf{v}$ 的统计绑定对外不可区分（模拟器论证）。

### X.4.4 复杂度对比

表 X.1 对比 ZAC 与基于 Merkle 树的方案在语料库规模为 $N$、单次证明覆盖 $k$ 个检索结果时的性能（128 位安全级别）：

| 方案 | 承诺大小 | 单批次证明大小 | 验证复杂度 |
|------|---------|-------------|----------|
| Merkle 树 | $32$ B | $k \cdot \lceil\log_2 N\rceil \cdot 32$ B | $O(k \log N)$ 次哈希 |
| ZAC（本文） | $48$ B | $48$ B（**常数**） | $O(k)$ 次乘法 $+ 2$ 次配对 |

具体数字：$N=303$，$k=5$ 时，Merkle 需要 $5 \times 9 \times 32 = 1440$ 字节；ZAC 仅需 48 字节，节省 **96.7%**。

---

# 第Y章 可验证检索框架的语料库证明设计

> 章节定位：本章在第 X 章理论基础上，描述本文针对多模态 RAG 场景对 ZAC 方案的适配设计，包括两项原创扩展——跨层承诺绑定与串联 Bloom Filter。

---

## Y.1 语料库可信性问题与威胁模型

### Y.1.1 问题描述

多模态 RAG 系统的典型架构中，语料库图像由服务方（Prover）离线构建、嵌入并存储于 FAISS 向量索引，用户（Verifier）在查询时接收 top-$k$ 检索结果。这一架构面临以下数据完整性威胁：

- **图像篡改**（B1 攻击）：服务方在磁盘上将某张图像文件替换为另一张，原始 embedding 和 FAISS 索引不变，检索结果指向伪造内容；
- **Embedding 替换**（B2 攻击）：服务方保持图像文件不变，但将 FAISS 索引中对应的 embedding 向量替换为精心构造的向量，从而操控检索排名。

两类攻击对用户完全不可见——用户无法区分返回的图像或对应 embedding 是否来自原始声称的语料库。

### Y.1.2 设计目标

Phase 1 需满足以下密码学可验证目标：

1. **来源可验证**：任何人持有公开 ZAC Root，可独立验证返回图像属于服务方在建库时声称的语料库；
2. **绑定性**：图像内容与 embedding 向量作为整体被承诺，任意一方的孤立替换均可被以高概率检出；
3. **常数证明大小**：聚合证明大小与语料库规模 $N$ 及检索数量 $k$ 无关；
4. **零知识性**：证明过程不泄露未被检索图像的任何信息。

---

## Y.2 承诺集合构造

### Y.2.1 基础元素哈希

系统将每张图像构造为一个 32 字节的承诺元素。最基础的设计是：

$$
s_i = \mathrm{SHA256}(\mathtt{image\_bytes}_i)
$$

此方案可检测 B1（图像替换）攻击，但对 B2（Embedding 替换）无效——攻击者只需保持图像文件不变即可绕过验证。

### Y.2.2 跨层承诺绑定（Cross-layer Commitment Binding）

**创新设计**：将 embedding 字节纳入承诺元素，构造跨层绑定哈希：

$$
s_i = \mathrm{SHA256}\!\left(\mathtt{image\_bytes}_i \;\|\; \mathtt{emb\_bytes}_i\right)
$$

其中 $\mathtt{emb\_bytes}_i$ 为 jina-v4 输出的 float32 embedding（2048 维，8192 字节）按小端序序列化后的字节串，$\|$ 表示字节拼接。

这一设计的安全论证如下：

- 若攻击者替换图像（B1）：$\mathtt{image\_bytes}_i$ 改变，SHA-256 哈希变化，新元素 $s_i'$ 不在 BF 中（以概率 $1 - \varepsilon$ 成立）；
- 若攻击者替换 embedding（B2）：$\mathtt{emb\_bytes}_i$ 改变，SHA-256 哈希同样变化，$s_i'$ 不在 BF 中（以概率 $1 - \varepsilon$ 成立）；
- 若攻击者同时替换图像和 embedding（B1+B2 联合）：两个字节串均变，SHA-256 哈希变化，以概率 $1 - \varepsilon$ 检出。

跨层绑定使 Phase 1（图像来源证明）与 Phase 2（FAISS 内积排名证明）形成"信任衔接"：任何孤立篡改均无法同时绕过两个阶段的验证。

```
图Y.1  跨层承诺绑定示意

     image_bytes_i          emb_bytes_i
          │                      │
          └──────────┬───────────┘
                     │  SHA256(·∥·)
                     ▼
                    s_i  ───► 纳入 BF 集合 S
                                    │
                              ZAC.Com(S, r)
                                    │
                              cm（ZAC Root）
                           48字节，公开发布
```

---

## Y.3 ZAC 在语料库场景的完整流程

**建库阶段**（离线，一次性）：

1. 从 PDF 渲染所有页面图像 $\{\mathtt{img}_i\}_{i=1}^N$；
2. 调用 jina-v4 编码器，生成 embedding 矩阵 $E \in \mathbb{R}^{N \times 2048}$；
3. 构造承诺集合：$\mathcal{S} = \{s_i\}_{i=1}^N$，$s_i = \mathrm{SHA256}(\mathtt{img\_bytes}_i \| E[i])$；
4. 初始化 BF 参数 $(q, k)$，生成 $\mathbf{v} \leftarrow \mathrm{BF.Gen}(\mathcal{S})$；
5. 初始化 Pointproofs CRS（BLS12-381，随机选取陷门 $\alpha$）；
6. 计算承诺：$\mathrm{cm} \leftarrow \mathrm{PR.Commit}(\mathbf{v}, r)$；
7. 公开发布 ZAC Root（$\mathrm{cm\_hex}$，96 字符十六进制字符串 = 48 字节 $\mathbb{G}_1$ 点）。

**查询阶段**（每次检索）：

1. FAISS 返回 top-$k$ 图像路径及对应 embedding；
2. 计算各检索结果的承诺元素：$\hat{s}_j = \mathrm{SHA256}(\mathtt{img\_bytes}_j \| \hat{E}_j)$；
3. 调用 $\hat{\pi} \leftarrow \mathrm{ZAC.ProveM}(\mathrm{cm}, \mathcal{S}, \hat{\mathcal{S}}, r, \mathrm{pp})$，输出 48 字节聚合证明；
4. 调用 $b \leftarrow \mathrm{ZAC.VerifyM}(\mathrm{cm}, \hat{\mathcal{S}}, \hat{\pi}, \mathrm{pp})$，验证者仅需公开 ZAC Root。

---

## Y.4 串联 Bloom Filter 扩展

### Y.4.1 问题分析

ZAC 的 $\varepsilon$-可靠性保证检测失败概率 $\leq \varepsilon$（即 Bloom Filter 假阳性率）。原始方案取 $\varepsilon = 0.01$，大样本实验（2400 次，3 个随机种子）验证实测 FPR = 1.00%，与理论完全吻合。在安全性要求较高的场景下，1% 的漏报概率可能不可接受。

直接减小 $\varepsilon$（如降至 0.001）会使 $q$ 约增大 1.5 倍，CRS 存储和 Prove 时间线性增长，代价较大。

### Y.4.2 串联扩展方案

**设计**：构造 $n$ 个独立的 BF+Pointproofs 层，每层独立生成承诺，验证要求所有层同时通过。

**独立性保证**：第 $i$ 层（$i = 0, 1, \ldots, n-1$）的 BF 使用 MurmurHash3 种子偏移 $[i \cdot k,\, i \cdot k + k - 1]$，其中 $k$ 为哈希函数数量。不同层的哈希函数集合不相交，使得各层的假阳性事件在统计上独立。

**复合假阳性率**：若各层独立，元素 $s \notin \mathcal{S}$ 通过所有 $n$ 层的概率为：

$$
\varepsilon_{\mathrm{cascade}} = \varepsilon^n
$$

取 $\varepsilon = 0.01$，$n = 2$：$\varepsilon_{\mathrm{cascade}} = 10^{-4}$（0.01%），即期望每 10000 次攻击尝试才有 1 次漏报。

**证明格式**：$n$ 个独立的 $\hat{\pi}_i \in \mathbb{G}_1$，串联后总大小为 $n \times 48$ 字节。ZAC Root 为 $n$ 个层承诺的拼接：$\mathrm{cm\_hex} = \bigoplus_{i=0}^{n-1} \mathrm{compress}(\mathrm{cm}_i)$（$n \times 96$ 字符）。

**验证规则**：

$$
b = \bigwedge_{i=0}^{n-1} \mathrm{ZAC.VerifyM}\!\left(\mathrm{cm}_i,\, \hat{\mathcal{S}},\, \hat{\pi}_i,\, \mathrm{pp}_i\right)
$$

所有层均通过才返回 1。

### Y.4.3 与原始方案的对比

| 指标 | 原始 ZAC（$n=1$） | 串联 ZAC（$n=2$） | 变化 |
|------|:---------------:|:---------------:|:----:|
| 每层 $\varepsilon$ | 0.01 | 0.01 | 不变 |
| 复合 FPR | 1% | **0.01%** | $\div 100$ |
| ZAC Root 大小 | 48 B | 96 B | $\times 2$ |
| 聚合证明大小 | 48 B | 96 B | $\times 2$ |
| Prove 时间（$N=50$） | $\approx$ 1.7 s | $\approx$ 3.4 s | $\times 2$ |
| Verify 时间 | $\approx$ 1.7 s | $\approx$ 3.4 s | $\times 2$ |
| CRS 构建（一次性） | $\approx$ 30 s | $\approx$ 60 s | $\times 2$ |
| $O(1)$ 常数性 | ✅（与 $N$ 无关） | ✅（仍与 $N$ 无关） | 保持 |

串联扩展以 2 倍的常数额外开销换取 100 倍的安全裕量提升，且保留了 ZAC 的全部优良性质。

---

## Y.5 设计参数选择

### Y.5.1 变量关系

ZAC 的核心设计变量是 $\varepsilon$（单层目标假阳性率）与 $n$（串联层数）。各指标对变量的敏感性如下：

| 指标 | 与 $\varepsilon$ 的关系 | 与 $n$ 的关系 |
|------|----------------------|--------------|
| BF 位数组长度 $q$ | $q \propto -\ln \varepsilon$（对数反比） | $q$ 不变 |
| BF 哈希函数数 $k$ | $k \propto \ln 2 \cdot q / N$ | $k$ 不变 |
| 复合 FPR | $\varepsilon^n$ | $\varepsilon^n$ |
| CRS 存储 | $O(q)$（线性） | $O(n \cdot q)$ |
| Prove/Verify 时间 | $O(q)$（线性） | $O(n)$ |

**关键结论**：减小 $\varepsilon$ 的代价是 $O(-\ln \varepsilon)$（近似线性于 $\ln(1/\varepsilon)$），而增加 $n$ 的代价是严格 $O(n)$（线性）。对于相同的目标复合 FPR，将 $\varepsilon$ 减半（如从 0.01 到 0.005）使 $q$ 增加约 44%；而增加一层（$n: 1 \to 2$）将复合 FPR 从 $\varepsilon$ 降至 $\varepsilon^2$（等效于将 $\varepsilon$ 开方），开销严格 $\times 2$。

### Y.5.2 本文参数选择

本文取 $\varepsilon = 0.01$，$n = 2$，理由如下：

- $\varepsilon = 0.01$ 是 ZAC 原始论文的实验参数，与理论 FPR 分析直接可比，便于验证实现正确性；
- $n = 2$ 将复合 FPR 降至 $10^{-4}$，远低于实际攻击者在有限尝试次数下的成功概率，满足工程需求；
- $n > 2$ 带来的边际安全收益（$10^{-4} \to 10^{-6}$）在当前原型系统中无法通过实验区分，暂无必要。

**参数配置表**（InfoVQA 实验，$N = 50$；Nikon 建库，$N = 303$）：

| 配置 | $N$ | $\varepsilon$ | $n$ | $q$ | $k$ | 复合 FPR |
|------|:---:|:---:|:---:|:---:|:---:|:-------:|
| InfoVQA 实验 | 50 | 0.01 | 2 | 480 | 7 | 0.01% |
| Nikon 建库 | 303 | 0.01 | 1\* | 2905 | 7 | 1% |

\* Nikon 建库实验以 $n=1$ 为基准测量延迟和扩展性（A1/A2 实验），验证 $O(k)$ 常数性；FPR 实验以 InfoVQA 验证正确性与串联改进效果。

---

## Y.6 实验验证

### Y.6.1 攻击检测率

**实验设计**：构造 B1（图像替换）和 B2（Embedding 替换）两类典型攻击，各 50 对测试样本，统计 ZAC 的检测率。

| 攻击类型 | 测试次数 | 检出次数 | 检出率 |
|---------|---------|---------|-------|
| B1 图像替换（$n=2$） | 50 | 50 | **100%** |
| B2 Embedding 替换（$n=2$） | 50 | 50 | **100%** |

所有测试中均无漏报，漏报率 = 0。

### Y.6.2 误报率大样本验证

**实验设计**：以 InfoVQA 数据集（语料 459 张，ZAC 集合取前 50 张）为测试集，对非成员元素测试 BF 误报率，重复 3 个随机种子，每种子每类测 400 次，共 2400 次独立试验。

**单层结果**（$n=1$，基线）：

| 类型 | 误报 | 实测 FPR | 95% Wilson CI | $\varepsilon=0.01 \in \text{CI}$ |
|------|-----|---------|--------------|:----------------------------:|
| B1-style | 13/1,200 | 1.08% | [0.63%, 1.83%] | ✅ |
| B2-style | 11/1,200 | 0.92% | [0.51%, 1.63%] | ✅ |
| 合计 | **24/2,400** | **1.00%** | [0.67%, 1.48%] | ✅ |
| 合法成员漏报 | 0/150 | 0% | — | ✅ |

**串联层结果**（$n=2$）：

| 类型 | 误报 | 实测 FPR | 95% Wilson CI | $\varepsilon^2=0.0001 \in \text{CI}$ |
|------|-----|---------|--------------|:--------------------------------:|
| B1-style | 0/1,200 | **0.00%** | [0.00%, 0.32%] | ✅ |
| B2-style | 0/1,200 | **0.00%** | [0.00%, 0.32%] | ✅ |
| 合法成员漏报 | 0/150 | 0% | — | ✅ |

单层方案实测 FPR（1.00%）与理论值 $\varepsilon = 0.01$ 完全吻合，验证实现正确性。串联方案（$n=2$）在相同规模测试下零误报，与理论预测（期望 $0.24$ 次）一致。合法成员在两种配置下均零漏报。

### Y.6.3 扩展攻击类型实验（E4）——多数据集多变体验证

**实验背景**：基于三篇文献的攻击分类：跨模态联合注入（Spa-VLM），语义嵌入替换（MedThreatRAG CMCI），以及 ZAC 成员性盲区分析。在 Nikon Z8（N=303）及 4 个 VisRAG-Ret 公开基准（N=459–1284）上系统验证 ZAC 对三类新型攻击的拦截能力，每类每个数据集各 120 个样本，B5-Swap 各 60 对。

**新增攻击变体**：

| 攻击变体 | 描述 | 检验假设 |
|---------|------|---------|
| **B5-Ext**（跨模态联合注入） | 构造外部合成对 $(img_{fake}, emb_{fake})$：全策略（异或 32 字节盐）或部分策略（仅修改尾部 10%），SHA256$(img_f \| emb_f) \notin S$ | ZAC 精确 hash 无容错，外部合成对 100% 检出 |
| **B6-Semantic**（语义相近替换） | 图像不变，embedding 替换为余弦相似度 $\geq 0.50$ 的另一文档 $j$ 的 embedding；SHA256$(img_i \| emb_j) \notin S$ | 相似度无论高低，hash 绑定精确匹配，均应 100% 检出 |
| **B5-Swap**（位置互换，盲区） | 将语料库中位置 $j$ 的真实对 $(img_j, emb_j)$ 移至位置 $i$；SHA256$(img_j \| emb_j) \in S$，ZAC 判定为合法成员 | 揭示 ZAC"成员性证明 ≠ 位置绑定"的设计边界 |

**五数据集实验结果**：

| 数据集（N） | B5-Ext 检出率 | B6-Semantic 检出率 | B5-Swap 结果 |
|------------|:-----------:|:----------------:|:-----------:|
| Nikon Z8（303） | 120/120 = **100%** | 120/120 = **100%** | 0/60（预期盲区） |
| SlideVQA（1284） | 120/120 = **100%** | 120/120 = **100%** | 0/60（预期盲区） |
| MP-DocVQA（741） | 120/120 = **100%** | 119/120 = **99.2%**† | 0/60（预期盲区） |
| ChartQA（500） | 120/120 = **100%** | 120/120 = **100%** | 0/60（预期盲区） |
| InfoVQA（459） | 120/120 = **100%** | 120/120 = **100%** | 0/60（预期盲区） |
| **合计** | **600/600 = 100%** | **599/600 = 99.8%** | **0/300（预期盲区）** |

† MP-DocVQA 1 次漏报（victim=60, donor=581，cosine sim=0.9996）：SHA256$(img_{60} \| emb_{581})$ 触发双层 BF 假阳性（两层 hash 函数独立碰撞）。期望次数 $\lambda = 600 \times \varepsilon^2 = 600 \times 10^{-4} = 0.06$，$P(X \geq 1) \approx 6\%$——统计合理，非代码缺陷。

FPR Phase 1 净基准：5 个数据集共 500 个干净样本，全部 0 误报，**FPR = 0.0%**。

**ZAC 安全边界：成员性证明 ≠ 位置绑定**

B5-Swap 实验揭示了 ZAC 方案的结构性安全边界：ZAC 集合 $S = \{\mathrm{SHA256}(img_i \| emb_i)\}$ 仅证明"此 $(img, emb)$ 对被承诺"，不证明"此对应出现在检索位置 $i$"。攻击者只需将合法对 $(img_j, emb_j)$ 移至位置 $i$，ZAC 验证即通过——验证器无法察觉语料库内部的位置互换。

**修复方案**：在哈希输入中加入位置索引：

$$\text{elem}_i = \mathrm{SHA256}(i \,\|\, img_i \,\|\, emb_i)$$

此后将位置 $j$ 的合法对移至位置 $i$ 时，$\mathrm{SHA256}(i \| img_j \| emb_j) \notin S$，ZAC 将检出。此修复将安全语义从"集合成员性"升级为"带位置索引的集合成员性"，仅需重建 ZAC 承诺（一次性离线操作，与现有 Pointproofs 框架兼容）。

---

## Y.7 与 Merkle 树方案的对比分析

在语料库规模 $N = 303$、每次检索返回 $k = 5$ 张图像的典型场景下，ZAC（$n=2$）与基于 Merkle 树的方案对比如下：

| 指标 | Merkle 树 | ZAC（$n=1$） | ZAC（$n=2$，本文） |
|------|:---------:|:-----------:|:----------------:|
| 承诺大小 | 32 B | 48 B | 96 B |
| 单批次证明大小 | $1,440$ B$^\dagger$ | 48 B | 96 B |
| 验证复杂度 | $O(k \log N)$ 哈希 | $O(k)$ 乘法 + 2e | $O(kn)$ 乘法 + $2n$e |
| 成员证明 FPR | 0（精确） | 1% | **0.01%** |
| 非成员证明 | ✅ | ✅（BF 定义） | ✅ |
| 零知识性 | ❌ | ✅ | ✅ |

$\dagger$ $5 \times \lceil\log_2 303\rceil \times 32 = 5 \times 9 \times 32 = 1440$ 字节。

ZAC（$n=2$）的证明大小（96 字节）仍比 Merkle 树（1440 字节）小 **93.3%**，同时满足零知识性并将 FPR 控制在 0.01% 以下。

---

## 参考文献

\[Bloom, 1970\] Burton H. Bloom. Space/Time Trade-offs in Hash Coding with Allowable Errors. *Communications of the ACM*, 13(7):422–426, 1970.

\[Catalano & Fiore, 2013\] Dario Catalano and Dario Fiore. Vector Commitments and their Applications. In *PKC 2013*, LNCS 7778:55–72.

\[Dang et al., 2022\] Hai-Van Dang, Tran Viet Xuan Phuong, Thuc D. Nguyen, and Thang Hoang. ZAC: Efficient Zero-Knowledge Dynamic Universal Accumulator and Application to Zero-Knowledge Elementary Database. In *IEEE TPS-ISA 2022*, pp. 248–257.

\[Gorbunov et al., 2020\] Sergey Gorbunov, Leonid Reyzin, Hoeteck Wee, and Zhenfei Zhang. Pointproofs: Aggregating Proofs for Multiple Vector Commitments. In *ACM CCS 2020*, pp. 2007–2023.

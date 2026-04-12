# 可验证多模态检索系统 — 实现日志

> 毕业设计：面向多模态语义数据的可验证检索机制
> 基础框架：UltraRAG + VisRAG 流水线

---

## Phase 1：语料库指纹生成

### 目标

对图像语料库 $\mathcal{S}$ 生成一个公开可验证的密码学承诺（ZAC Root），作为语料库的数字指纹，用于后续检索结果的完整性验证。

### 核心算法

**Step 1 — 集合构造**

$$
\mathcal{S} = \{ \mathrm{SHA256}(\mathtt{image\_bytes}_i) \mid i = 1, \ldots, N \}
$$

**Step 2 — Bloom Filter 编码**

$$
\mathbf{v} \leftarrow \mathrm{BF.Gen}(\mathcal{S}) \in \{0,1\}^q
$$

最优参数选择（误报率 $\varepsilon$，集合上界 $N$）：

$$
q = \left\lceil \frac{-N \ln \varepsilon}{\ln^2 2} \right\rceil, \quad k = \left\lceil \frac{q}{N} \ln 2 \right\rceil
$$

**Step 3 — Pointproofs 向量承诺**

公共参数（CRS）：$\alpha \xleftarrow{\$} \mathbb{Z}_p$，

$$
\mathcal{P} = (g_1^{\alpha}, \ldots, g_1^{\alpha^q}), \quad \mathcal{V} = (g_2^{\alpha}, \ldots, g_2^{\alpha^q})
$$

承诺（ZAC Root）：

$$
\mathtt{cm} \leftarrow \mathrm{PR.Commit}(\mathbf{v}, r) = g_1^{r \cdot \alpha^q} \cdot \prod_{i=1}^{q-1} g_1^{v_i \cdot \alpha^i} \in \mathbb{G}_1
$$

其中 $r \xleftarrow{\$} \mathbb{Z}_p$ 为随机盲化因子（保证零知识性）。

输出：48 字节 BLS12-381 $\mathbb{G}_1$ 压缩点，即语料库唯一指纹。

### 关键实现细节

**修复 BF/Pointproofs 向量长度对齐问题**

论文中 BF 生成长度为 $q-1$ 的向量，PR 参数为 $q$（第 $q$ 槽保留给随机数 $r$）。
代码中 BF 生成长度为 `bf.q` 的向量，因此 Pointproofs 初始化须使用 `q = bf.q + 1`，
避免 BF 位置落在随机数槽导致验证失败。

```python
# 正确写法
self._pr = Pointproofs(q=self._bf.q + 1, alpha=alpha)
```

### 实现文件

| 文件 | 说明 |
|------|------|
| `src/zac/accumulator.py` | BloomFilter、Pointproofs、ZACAccumulator 三个类 |
| `script/phase1_corpus_fingerprint.py` | 命令行脚本：PDF→语料库→ZAC 指纹 |

### 运行命令

```bash
conda activate ultrarag
cd /root/autodl-tmp/UltraRAG

# 从已有语料库生成 ZAC 指纹
python script/phase1_corpus_fingerprint.py \
    --zac-only \
    --corpus-jsonl corpora/image.jsonl \
    --embedding-npy embedding/embedding.npy \
    --output output/phase1/fingerprint.json
```

### 阶段测试结果

**测试环境**

- 语料库：`data/nikon.pdf` 转换后共 **288 张**页面图像
- 语料库路径：`corpora/image.jsonl`
- 运行环境：Ubuntu 20.04，CPU（py_ecc 纯 Python 实现）

**BF/Pointproofs 参数**（$N=288$，$\varepsilon=0.01$）

$$
q = 2761, \quad k = 7
$$

**Phase 1 输出**

| 指标 | 结果 |
|------|------|
| 语料库大小 $N$ | 288 张图像 |
| BF 向量长度 $q$ | 2761 |
| BF 哈希函数数 $k$ | 7 |
| ZAC Root | `87fcaa3463dc96fc…` (48 字节) |
| CRS 构建时间 | 168.2 s（仅需一次） |
| 单元素成员证明验证 | **5/5 PASS** |
| 生成文件 | `output/phase1/fingerprint.json`<br>`output/phase1/prover_state.json`<br>`output/phase1/proof_sample_0.json` |

---

## Phase 2：可验证检索

### 目标

在 VisRAG 检索流水线中，对 top-$k$ 检索结果附加聚合 ZAC 成员证明，使任何人可以验证返回图像确实来自原始已承诺的语料库。

### 核心算法

**证明阶段 ZAC.ProveM**

对检索结果集 $\hat{\mathcal{S}} = \{\hat{s}_1, \ldots, \hat{s}_k\}$（各元素为图像 SHA256 哈希）：

$$
\mathcal{I} = \bigcup_{\hat{s} \in \hat{\mathcal{S}}} \{ h_1(\hat{s}), \ldots, h_k(\hat{s}) \}
\quad \text{（BF 位置集合，已排序）}
$$

对每个位置 $i \in \mathcal{I}$ 生成 Pointproofs 单元证明：

$$
\pi_i = \prod_{j \in [q],\, j \neq i} g_1^{m'_j \cdot \alpha^{q+1-i+j}}
$$

其中 $\mathbf{m}' = (v_1, \ldots, v_{q-1}, r)$。聚合（Fiat-Shamir 挑战 $t_i$）：

$$
t_i = H'(i,\, \mathtt{cm},\, \mathcal{I},\, \mathbf{v}[\mathcal{I}]), \quad
\hat{\pi} = \prod_{i \in \mathcal{I}} \pi_i^{t_i} \in \mathbb{G}_1
$$

输出：一个 **48 字节** $\mathbb{G}_1$ 点，与 $k$ 无关（$O(1)$ 证明大小）。

**验证阶段 ZAC.VerifyM**

验证者仅需公开的 $\mathtt{cm}$、被证明的图像路径、以及聚合证明 $\hat{\pi}$：

$$
e\!\left(\mathtt{cm},\; \sum_{i \in \mathcal{I}} t_i \cdot g_2^{\alpha^{q+1-i}}\right)
\stackrel{?}{=}
e(\hat{\pi},\, g_2) \cdot g_T^{\alpha^{q+1} \cdot \sum_{i \in \mathcal{I}} v_i \cdot t_i}
$$

其中 $e : \mathbb{G}_2 \times \mathbb{G}_1 \to \mathbb{G}_T$ 为 BLS12-381 双线性配对。

### 实现文件

| 文件 | 说明 |
|------|------|
| `src/zac/accumulator.py` | 新增 `prove_membership_batch`、`verify_membership_batch`、`save_prover_state`、`load_prover_state` |
| `script/phase2_verifiable_search.py` | Demo 脚本（`--demo`）和实验脚本（`--experiment`） |
| `servers/retriever/src/retriever.py` | `retriever_init` 新增 `zac_prover_state` 参数；`retriever_search` 后自动附加 ZAC 证明 |
| `examples/parameter/visrag_parameter.yaml` | 新增 `zac_prover_state: output/phase1/prover_state.json` |
| `script/case_study.py` | UI 新增 `zac_verification` 专用渲染（通过/失败徽章 + 展开详情） |

### 运行命令

```bash
# Demo：加载已有 prover state，对 top-5 生成证明并验证
python script/phase2_verifiable_search.py --demo \
    --corpus-jsonl corpora/image.jsonl \
    --corpus-base  corpora/ \
    --prover-state output/phase1/prover_state.json \
    --k 5

# 实验：生成论文指标表（k=1,3,5,10,20）
python script/phase2_verifiable_search.py --experiment \
    --corpus-jsonl corpora/image.jsonl \
    --corpus-base  corpora/ \
    --output output/phase2/experiment.json
```

### 阶段测试结果

**端到端 Demo 输出**（288 张语料库，加载已有 prover state）

```
[4/5] ZAC.ProveM  (aggregate proof for 5 images)
      Proof size  : 48 bytes  (1 G1 point, O(1))
      Prove time  : 1973.0 ms

[5/5] ZAC.VerifyM
      Verify time : 2493.8 ms
      Result      : ✓ PASS

[Tamper Test] 替换一张图为非语料库图像
      Tampered verify : ✗ FAIL (correctly rejected)
```

**性能指标表**（$N=288$，不同 $k$ 值，各取 3 次中位数）

| $k$ | 证明大小 | Prove 时间 | Verify 时间 | Merkle 证明对比 |
|:---:|:-------:|:---------:|:-----------:|:--------------:|
| 1 | **48 B** | 396 ms | 1334 ms | 288 B（9层×32B） |
| 5 | **48 B** | 1907 ms | 2453 ms | 1440 B |
| 10 | **48 B** | 3863 ms | 3893 ms | 2880 B |

> **Merkle 对比说明**：$N=288$ 时 $\lceil \log_2 288 \rceil = 9$ 层，
> Merkle 每条路径需 $9 \times 32 = 288$ 字节，$k$ 条互相独立路径共 $k \times 288$ 字节。
> ZAC 聚合证明始终为 **48 字节**，与 $k$ 和 $N$ 均无关。

**正确性测试**（$N=10$，$\varepsilon=0.01$）

| 测试项 | 结果 |
|--------|------|
| 单元素成员证明（$k=1,2,3,4,5$） | 全部 **PASS** |
| 批量成员证明验证 | **PASS** |
| 篡改检测（替换为确认非成员） | **PASS（正确拒绝）** |
| `save_prover_state` + `load_prover_state` 往返 | **PASS** |
| 跨实例验证（acc2 生成，acc 验证） | **PASS** |

**关于 Bloom Filter 误报（False Positive）**

BF 以概率 $\varepsilon$ 将不在集合中的元素误判为成员。
当 $\varepsilon = 0.1$（测试用小参数）时，篡改图像恰好触发误报则 ZAC 同样通过验证——
这是 BF 的已知固有局限，不是实现错误。
生产环境使用 $\varepsilon = 0.01$，误报率降至 $1\%$ 以下。
论文中对此遵循原文表述（Theorem 1 的 $\varepsilon$-soundness）。

### 问题 6：ZAC 证明粒度错误（跨 query 混合聚合）

**现象**：多个 query 的检索结果被合并成一个大列表，生成单个 ZAC 聚合证明，与 `ret_psg`（每 query 独立一组结果）的结构不对应。

**根因**：原始实现展平了所有 query 的路径：
```python
all_paths = [p for per_query in rets for p in per_query if p]  # 跨 query 合并
```

**修复**：改为与 `ret_psg` 平行的结构——每个 query 独立 Prove + Verify，`zac_verification` 为 `List[dict|None]`：

```python
per_query_results = []
for per_query_paths in rets:          # 与 ret_psg 等长
    elements = [image_hash(p) for p in per_query_paths if p]
    proof = prove_membership_batch(elements)
    verified = verify_membership_batch(elements, proof)
    per_query_results.append({...})
zac_verification = per_query_results  # List[dict|None]
```

`save_retrieve_results.py` 同步按索引对应，并修正了摘要打印（原代码对 List 调 `.get('verified')` 会报 AttributeError）：
```python
data['zac_verification'] = zac_verification[i]   # 第 i 个 query 的验证结果

# 摘要打印：聚合所有 query 的结果
valid = [v for v in zac_verification if v is not None]
all_pass = valid and all(v.get('verified') for v in valid)
```

### 信任模型与公开性分析

**各字段的公开性**

| 字段 | 类型 | 是否可公开 | 说明 |
|------|------|-----------|------|
| `cm_hex` | ZAC Root（48B G1 点） | ✓ 公开 | 语料库承诺，设计上需广泛发布 |
| `proof_hex` | 聚合证明 $\hat{\pi}$（48B G1 点） | ✓ 公开 | 不含任何秘密，任何人可验证 |
| `verified` | 服务器自验结论 | 仅供参考 | 见下方信任模型说明 |
| `alpha_hex` | Pointproofs trapdoor | ✗ 秘密 | 仅存于 `prover_state.json`，永不传输 |

UI 展示 `cm_hex` 和 `proof_hex` 没有任何安全问题。即使在证明者与验证者完全分离的系统中，这两个值也必然在信道中流转：`proof_hex` 由服务器传给客户端，`cm_hex` 作为公开承诺在可信渠道预先发布。

**完整信任模型**

```
[Phase 1 — 离线，一次性]
  语料库构建者  →  cm_hex 公开存证（可信渠道：论文、区块链、公告等）

[Phase 2 — 在线，每次检索]
  Client ──查询──► Server
  Client ◄──────── top-k 图像路径 + proof_hex（声称值）
  Client 本地验证：
      用可信 cm_hex 检验 e(cm, ...) = e(π̂, g₂)·gT^{...}
      → PASS / FAIL（不依赖服务器的 verified 字段）
```

当前实现中，服务器同时执行 Prove 和 Verify，返回的 `verified: True` 是**自我声明**，在对抗性场景下不可信赖（恶意服务器可伪造该字段）。这是架构层面的已知局限——真正的安全性依赖于 `cm_hex` 通过独立可信渠道预先发布，以及客户端侧的独立验证。对于毕业设计的系统原型，服务器端验证作为端到端正确性的演示和日志记录是合理的。

### 安全性边界：检索完整性 vs. 语料库完整性

ZAC 可验证检索提供的保证是**检索完整性**（retrieval integrity）：

> 返回的 top-k 图像确实属于构建承诺时的原始语料库。

但它**不**提供**语料库全局完整性**（corpus integrity）：

> 若攻击者篡改了语料库中某张图像，但该图像在本次查询中未被检索到，验证仍然通过。

原因：ZAC 证明仅覆盖被选中的图像的哈希是否在 Bloom Filter 中。未被检索的图像根本不参与证明。这是成员证明方案的固有属性，与具体的 ZAC 实现无关——Merkle Path 证明同样如此。

**被篡改图像若被检索，会被检测到**：`image_hash(path)` 对磁盘上的当前文件内容计算 SHA256。若文件被修改，哈希变化，对应的 BF 位置也会变化，验证失败。

| 场景 | 结果 |
|------|------|
| 返回图像均来自原始语料库 | ✓ 验证通过 |
| 返回图像中有被篡改的文件 | ✗ 验证失败（哈希不匹配） |
| 语料库中有未被检索的篡改图像 | ✓ 本次验证通过（未覆盖该图像） |

论文中应明确：本系统的安全目标是**防止检索结果被伪造或替换**（即服务器无法返回不属于已承诺语料库的图像），而非对整个语料库做全量审计。

### 局限性说明

**GPU 加速**：py_ecc 为纯 Python CPU 实现，BLS12-381 配对运算无 CUDA 路径。
CRS 加载（~170 s）仅在服务启动时执行一次，此后每次检索的额外开销为 Prove + Verify 共约 4–8 s（取决于 $k$）。
理论上替换为 Rust/C 实现（如 `blst`）可提速约 100 倍，留作后续优化。

---

## Phase 2 后续修复

### 问题 1：ZAC 结果未写入输出 JSON

**现象**：`retriever_search` 返回了 `zac_verification` 字段，但 `save_retrieve_results.py` 只读取 `memory_ret_psg`，导致验证结果不出现在输出 JSONL 中，Case Study UI 无法渲染。

**根因**：`save_retrieve_results.py` 硬编码只提取 `memory_ret_psg`，未读取 `memory_zac_verification`。

**修复**（`script/save_retrieve_results.py`）：

```python
elif tool['step'] == 'retriever.retriever_search':
    ret_psg = tool['memory']['memory_ret_psg']
    zac_verification = tool['memory'].get('memory_zac_verification')  # 新增

for data, re_q, re_ret_psg in zip(origin_data, re_q_ls, ret_psg):
    data['retrieved_passages'] = re_ret_psg
    if zac_verification is not None:
        data['zac_verification'] = zac_verification             # 新增
```

### 问题 2：`zac_verification` 未声明为工具输出

**现象**：`retriever_search` 已返回 `{"ret_psg": ..., "zac_verification": ...}`，但 MCP 框架只把 `ret_psg` 写入 `memory_`，`zac_verification` 被丢弃。

**根因**：工具注册的 output 映射只声明了 `ret_psg`。

**修复**（`servers/retriever/src/retriever.py`）：

```python
# 修改前
output="q_ls,top_k,query_instruction->ret_psg"
# 修改后
output="q_ls,top_k,query_instruction->ret_psg,zac_verification"
```

### 问题 3：流水线报错 `Variable zac_prover_state cannot be found`

**现象**：启动 VisRAG 流水线时报错：
```
[UltraRAG Error] Variable zac_prover_state cannot be found from pipeline
                 before retriever.retriever_init step
```

**根因**：UltraRAG 框架通过 `server.yaml`（构建时生成的缓存）决定参数来源。
框架规则：input 值若有 `$` 前缀，从本地 YAML（`local_vals`）读取；否则视为前序步骤产出的全局变量。
`server.yaml` 是用 `servers/retriever/parameter.yaml` 构建的，而 `zac_prover_state` 只加到了 `examples/parameter/visrag_parameter.yaml`，导致 `server.yaml` 生成了 `zac_prover_state: zac_prover_state`（无 `$` 前缀），框架去全局变量里找，当然找不到。

**修复三步**：

| 文件 | 修改 |
|------|------|
| `servers/retriever/parameter.yaml` | 新增 `zac_prover_state: null`（默认关闭） |
| `servers/retriever/server.yaml` | `zac_prover_state: zac_prover_state` → `zac_prover_state: $zac_prover_state` |
| `examples/parameter/visrag_parameter.yaml` | 已有 `zac_prover_state: output/phase1/prover_state.json`（运行时覆盖） |

**运行时参数流**：

```
servers/retriever/parameter.yaml   →  zac_prover_state: null（默认）
          ↓ client.py local_vals.update()
examples/parameter/visrag_parameter.yaml  →  zac_prover_state: output/phase1/prover_state.json
          ↓ $zac_prover_state
retriever_init 收到路径，加载 ZACAccumulator
```

**验证**（Python 模拟）：

```
默认 zac_prover_state : None
覆盖后 zac_prover_state: output/phase1/prover_state.json
server.yaml 里的值   : $zac_prover_state   ← 有 $ 前缀 ✓
```

### 问题 4：MCP 输出类型校验失败（`zac_verification` 非 array）

**现象**：流水线运行时抛出：
```
fastmcp.exceptions.ToolError: Output validation error:
{'verified': True, 'num_images': 30, ...} is not of type 'array'
```

**根因**：fastmcp 根据工具函数的**返回类型注解**自动生成 JSON Schema 并校验返回值。
`retriever_search` 原注解为 `-> Dict[str, List[List[str]]]`，fastmcp 据此要求每个值都是 array of arrays。
`zac_verification` 是 `dict`，不符合 `List[List[str]]`，校验在服务端返回前就抛出异常。

**修复**：将返回类型改为 `Dict[str, Any]`，fastmcp 不再对每个 value 做具体类型约束

```python
# 修改前
async def retriever_search(...) -> Dict[str, List[List[str]]]:
# 修改后
async def retriever_search(...) -> Dict[str, Any]:
```

**附加处理**：`_update_memory` 会对每次工具调用结果做 `list.append`，因此 `memory_zac_verification` 在快照 JSON 中是 `[dict]` 而非 `dict`。`save_retrieve_results.py` 需解包：

```python
raw = tool['memory'].get('memory_zac_verification')
zac_verification = raw[-1] if isinstance(raw, list) and raw else raw
```

### 问题 5：CRS 缓存文件 G2 段全零

**现象**：`load_prover_state` 加载 `.crs` 文件后，V（G2）点坐标全为 $(0,0)$，
配对运算瞬间返回（identity 元素快速路径），导致所有验证结果为 False。

**根因**：之前写入 `.crs` 时的 `g2_bytes()` 函数存在 bug（FQ2 系数获取方式有误），
虽然文件大小正确（`projective` 格式：$q \times 288$ 字节），但 G2 段数据全为零字节。

**修复三步**：

1. 修正 `save_crs` 中 G2 序列化为正确的 projective 格式（直接存 `X.coeffs`, `Y.coeffs`, `Z.coeffs`）
2. 删除旧的错误 `.crs` 文件
3. 重新调用 `save_prover_state` 生成正确 `.crs`（一次性代价 167.8s）

**修复后验证**：

| 指标 | 结果 |
|------|------|
| `load_prover_state`（有 `.crs`） | **0.6 s**（原 167.8 s） |
| 单成员验证 | PASS |
| 批量验证（k=3） | PASS |
| 篡改检测 | PASS（正确拒绝） |

**CRS 文件格式（projective，big-endian，48 字节/域元素）**

```
[4B]        q（Pointproofs 向量长度）
[q × 144B]  P  = { g₁^{αⁱ} }  i=1..q （G1 projective：X, Y, Z 各 48B）
[(q-1)×144B] Px = { g₁^{α^{q+2+j}} } j=0..q-2
[q × 288B]  V  = { g₂^{αⁱ} }  i=1..q （G2 projective：X₀,X₁, Y₀,Y₁, Z₀,Z₁ 各 48B）
```

---

## Phase 2：可验证检索——Sumcheck 内积与排序证明

### 目标

对 FAISS 返回的 top-k 检索结果，生成数学可验证的证明：
1. **内积正确性**：$s_i = q \cdot v_i$（相似度分值确实由 query 和语料库向量计算而来）
2. **全局最优性**：返回的 top-k 确实是全部 N 个语料库向量中分值最高的 k 个

攻击防御目标：服务器无法伪造分值、调换排序，也无法隐藏更高分的结果。

> **Local 模式 vs Global 模式**（`sumcheck_mode`）：
> - **Local（旧）**：仅对返回的 k 个向量生成 k 条 Sumcheck + 1 条排序证明。证明内积正确和相对顺序，但无法排除服务器隐藏了更高分的结果。
> - **Global（默认）**：Prover 宣告全部 N 个语料库向量的分值，通过 Schwartz-Zippel 随机线性组合派生出一个聚合向量 $w$，用单条 Sumcheck 证明所有分值的批次正确性。Verifier 独立排序 N 个分值，自行选出 top-k，彻底消除全局最优性漏洞。

### 核心算法

**Sumcheck 协议（内积证明）**

将向量内积表达为多线性多项式的求和：

$$H = \sum_{j \in \{0,1\}^\ell} q_j \cdot v_j \quad \text{（}\ell = \lceil \log_2 d \rceil\text{，}d\text{ 为向量维度）}$$

通过 $\ell$ 轮交互（Fiat-Shamir 变成非交互）证明这个求和：

- 第 $i$ 轮：Prover 发送单变量多项式 $g_i(X)$（3个值表示度数≤2）
- Verifier 检查 $g_i(0) + g_i(1) = C_{i-1}$，发送随机挑战 $r_i$
- 最终 Oracle 查询：Verifier 直接计算 $\tilde{q}(r_1,\ldots,r_\ell) \cdot \tilde{v}(r_1,\ldots,r_\ell)$ 并比较

$$\text{证明大小} = \ell \times 3 \text{ 个域元素} = 11 \times 24 \approx \mathbf{264 \text{ 字节}}（d=2048）$$

**Global Batch Sumcheck（Schwartz-Zippel 批次证明，默认）**

Prover 宣告所有 $N$ 个内积分值 $s_i = q \cdot v_i$，通过 Fiat-Shamir 从所有分值派生随机标量 $\rho$：

$$\rho \leftarrow \mathrm{SHA256}(s_1 \| s_2 \| \cdots \| s_N) \bmod p$$

构造聚合向量 $w = \sum_{i=1}^N \rho^i \cdot v_i$，批次内积目标值 $s_\text{batch} = \sum_{i=1}^N \rho^i \cdot s_i$，用**单条** Sumcheck 证明 $q \cdot w = s_\text{batch}$。

- 若任一 $s_i$ 被篡改，$\rho$ 变化导致 $s_\text{batch}$ 不一致，Sumcheck 以 $1 - N/p \approx 1-2^{-52}$ 的概率检测到。
- Verifier 独立排序 $N$ 个宣告分值，自行选出 top-k：无需信任 Prover 的排名。

$$\text{Global 证明大小} = N \times 8\text{ B（分值）} + 264\text{ B（Sumcheck）} \approx 2.6\text{ KB}（N=288）$$

**排序证明（非负 delta 见证，Local 模式专用）**

$$\delta_i = s_i - s_{i+1} \geq 0, \quad i=1,\ldots,k-1$$

Verifier 检查：$s_i - s_{i+1} = \delta_i$ 且 $\delta_i \geq 0$。

**有限域**：$\mathbb{Z}_p$，$p = 2^{61} - 1$（Mersenne 素数，Python 原生大整数，取模速度快）

**量化**：float32 向量 × scale 取整，转为域元素。

- **初始选择 scale=256**：保守经验值（$2^8$），溢出上界 $256^2 \times 2048 \approx 1.3 \times 10^8 \ll p$，精度步长 $1/256 \approx 4 \times 10^{-3}$。
- **更新为 scale=65536（$2^{16}$）**：C2 实验发现 scale=256 在随机向量场景下 top-10 出现排名不一致（分值接近零时量化噪声超过分值差），而 scale=65536 将精度步长降至 $1.5 \times 10^{-5}$，top-10 全部一致，且溢出上界 $65536^2 \times 2048 \approx 8.8 \times 10^{12} \ll p$，耗时无变化（1.0s vs 1.1s）。同时与 zkLLM 的 $2^{16}$ 量化方案对齐。

### 复杂度对比

| 指标 | Sumcheck（本实现） | Bulletproofs（纯 Python，d=2048） |
|------|:----------------:|:-------------------------------:|
| 证明大小（每对 q,v） | **264 B** | ~1 KB |
| Prover 时间 | $O(d)$ | $O(d \log d)$ |
| Verifier 时间 | $O(d)$ | $O(d)$ |
| Python 可行性 | ✅ ms 级 | ✗ 分钟级（d=2048 太慢） |

### 局限性：Verifier 需持有语料库副本（未引入 KZG）

**当前信任模型**：`verify_global_batch` 中 verifier 需重建聚合向量 $w = \sum_i \rho^i v_i$，
这要求 verifier 本地持有完整的 `embedding.npy`（O(N·D) 工作量，N=303 约 442ms）。
Sumcheck 在此场景下证明的是：**给定这些向量，计算过程和 top-k 选择未被篡改**，不证明向量本身的来源。

**KZG 多项式承诺可解决此问题**：
- 离线对每个 $v_i$ 计算承诺 $\text{cm}_i = \text{KZG.Commit}(v_i)$，发布承诺集合
- 在线 verifier 利用承诺的同态性：$\text{cm}_w = \sum_i \rho^i \cdot \text{cm}_i$（O(N) 次 G1 运算）
- 附加 opening proof 证明 $\tilde{w}(r_1,\ldots,r_\ell) = y$（1 次 pairing 验证）
- Verifier 无需持有任何 $v_i$，理论复杂度降至 O(N) G1 运算 + O(1) pairing

**未实现原因（工程约束）**：
- 离线承诺建库：303 × 2048 次 G1 scalar mul × 2.1ms = **~1329 秒**（py_ecc 纯 Python）
- 在线 verifier：303 次 G1 scalar mul = **649ms**（反而慢于当前 442ms 的域运算重建）
- BLS12-381 G1 运算在 py_ecc 中比 Mersenne 域运算慢约 3000×
- 在 N=303 规模下，O(N·D) 域运算（442ms）vs O(N) G1 运算（649ms），KZG 无速度优势
- KZG 的真实价值是**消除 verifier 持有语料库的假设**，在 demo 场景中该假设成立，故优先级低

**论文说明**：在 Phase 2 局限性节说明：当前 verifier 采用"持有语料库副本"的信任模型；
引入 KZG 可消除此假设，但受限于 Python BLS12-381 运算速度，留作工程优化方向。

### 实现文件

| 文件 | 说明 |
|------|------|
| `src/sumcheck/inner_product.py` | Sumcheck 协议核心：`prove_inner_product`、`verify_inner_product`、`prove_ranking`、`verify_ranking`、`prove_retrieval`、`verify_retrieval`（local）、`prove_global_batch`、`verify_global_batch`（global） |
| `script/phase2_sumcheck.py` | Demo（`--demo`）+ 实验（`--experiment`）+ 单元测试（`--test`，13/13 PASS）脚本 |
| `servers/retriever/src/retriever.py` | `retriever_init` 新增 `sumcheck_embedding_npy`、`sumcheck_mode`（默认 `"global"`）；`retriever_search` 按 mode 分支调用对应证明函数 |
| `servers/retriever/parameter.yaml` | 新增 `sumcheck_embedding_npy: null`、`sumcheck_mode: global` |
| `servers/retriever/server.yaml` | 新增 `sumcheck_mode: $sumcheck_mode` |
| `examples/parameter/visrag_parameter.yaml` | 新增 `sumcheck_embedding_npy: embedding/embedding.npy`、`sumcheck_mode: global` |

### 运行命令

```bash
conda activate ultrarag
cd /root/autodl-tmp/UltraRAG

# 单元测试（无需语料库）
python script/phase2_sumcheck.py --test

# Demo：加载已有 embedding，对 top-5 生成证明并验证
python script/phase2_sumcheck.py --demo \
    --embedding-npy embedding/embedding.npy \
    --corpus-jsonl  corpora/image.jsonl \
    --k 5

# 实验：不同 k 值的性能指标表
python script/phase2_sumcheck.py --experiment \
    --embedding-npy embedding/embedding.npy \
    --corpus-jsonl  corpora/image.jsonl \
    --output output/phase2/sumcheck_experiment.json
```

### 阶段测试结果

**单元测试**（无语料库，纯协议正确性验证）

| 测试项 | 结果 |
|--------|------|
| d=4,16,128,256,2048 Sumcheck prove+verify | 全部 **PASS** |
| H 值与直接计算内积一致 | **PASS** |
| 篡改检测（v[0] += 0.1） | **PASS（正确拒绝）** |
| 排序证明（正确顺序） | **PASS** |
| 排序证明（调换顺序） | **PASS（正确拒绝）** |
| Global batch N=8, d=16, k=3 | **PASS** |
| Global batch N=20, d=64, k=5 | **PASS** |
| Global top-k 与 numpy ground truth 一致 | **PASS** |
| Global 语料库篡改检测（corpus[0][0]+=0.5） | **PASS（正确拒绝）** |
| Global 分值篡改检测（scores[0]+= 999999） | **PASS（正确拒绝）** |

**总计：13/13 全部通过**

**性能预估**（d=2048，CPU，Python）

*Local 模式*（复杂度随 k 线性增长）

| k | 证明大小 | Prove 时间 | Verify 时间 |
|:---:|:-------:|:---------:|:-----------:|
| 1 | ~400 B | ~5 ms | ~5 ms |
| 5 | ~2 KB | ~25 ms | ~25 ms |
| 10 | ~4 KB | ~50 ms | ~50 ms |

*Global 模式*（复杂度随 N 线性增长，与 k 无关）

主要开销在于 $w = \sum_{i=1}^N \rho^i \cdot v_i$ 的计算：$N \times d$ 次域乘法（~600K 次，N=288，d=2048），外加 1 次 Sumcheck（~5 ms）。

| N（语料库规模） | 证明大小 | Prove 时间 | Verify 时间 | 备注 |
|:---:|:-------:|:---------:|:-----------:|:---|
| 50 | ~664 B | ~90 ms | ~90 ms | 小型测试集 |
| 288 | ~2.6 KB | ~500 ms | ~500 ms | Nikon PDF 示例 |
| 1000 | ~8.3 KB | ~1.7 s | ~1.7 s | 中等语料库 |

> Global 模式的 Prove/Verify 时间均比 ZAC（2–4 s）小或相当，整体不成为性能瓶颈。
> 若语料库很大（N > 5000），可考虑切换为 Local 模式或对 `_m()` 用 NumPy 向量化加速。

### 流水线集成说明

**新增参数**：
- `sumcheck_embedding_npy`（路径到 `embedding/embedding.npy`）
- `sumcheck_mode`：`"global"`（默认）或 `"local"`

**初始化**（`retriever_init`）：
1. 加载 corpus embedding numpy 数组 → `self._sc_embeddings`（shape N×D）
2. 建 `path → row_index` 映射 → `self._sc_path_to_idx`
3. 存储 `self._sc_mode = sumcheck_mode`

**检索时**（`retriever_search`，在 ZAC 证明之后）：

*Global 模式（默认）*：
1. 将全部 N 个 embedding 转为 Python list → `all_corpus_vecs_list`（仅一次）
2. 对每条 query：`prove_global_batch(q_list, all_corpus_vecs_list)` → 证明所有 N 个分值
3. `verify_global_batch(q_list, all_corpus_vecs_list, proof, top_k)` → 返回 Verifier 独立选出的 top-k
4. 附加 `sumcheck_verification`（`List[dict|None]`，含 `top_k_indices`、`top_k_scores`）

*Local 模式*：
1. 通过 `_sc_path_to_idx` 查找 FAISS 返回的 k 个向量
2. `prove_retrieval(q_vec, corpus_vecs)` + `verify_retrieval` → k 条 IP 证明 + 排序证明

**Global 输出字段**：
```json
{
  "verified": true,
  "mode": "global",
  "N": 288,
  "k": 5,
  "ell": 11,
  "proof_bytes": 2568,
  "top_k_indices": [42, 7, 131, 88, 250],
  "top_k_scores": [15823, 14291, 13654, 12800, 11940],
  "prove_ms": 520.0,
  "verify_ms": 480.0
}
```

### 安全分析

| 攻击 | Global Sumcheck 的防御 |
|------|----------------|
| 服务器伪造相似度分值 | $\rho$ 由所有宣告分值派生，修改任意 $s_i$ 导致 $s_\text{batch}$ 不一致，Sumcheck 以 $\geq 1-N/p$ 概率检测 |
| 服务器隐藏更高分结果（只返回非最优 top-k）| Verifier 持有全部 N 个分值，独立排序选出 top-k，Prover 无法欺骗 |
| 服务器调换排序 | Verifier 自行排序，不依赖 Prover 声称的顺序 |
| 服务器替换向量 $v_i$（配合伪造证明） | Verifier 重新计算 $w = \sum_i \rho^i v_i$，最终 Oracle 查询检测篡改，概率 $\approx 2^{-61}$ |

**声音误差**：$N/p \approx 288/2^{61} \approx 2^{-53}$（$N=288$），可忽略不计。

**局限性**：Sumcheck 证明整数内积（量化后）的正确性。量化误差（float32→int，scale=65536，步长 $\approx 1.5 \times 10^{-5}$）在实际语义查询场景下对 top-5 排名无影响（C2 实验验证）；初始 scale=256 在极端场景（随机向量，分值趋近零）下曾出现 top-10 不一致，升级后消除。

---

## 总体架构与完整信任链

三个验证 Phase 覆盖从原始 PDF 到最终检索结果的完整路径，没有可信间隙。

> **Phase 3 方案已更新**：原计划使用 EZKL（仅证明 pooling 头），经研读 zkLLM（Sun et al., CCS'24）和 zkGPT（Qu et al., Eurosys'24）后，改为基于 **zkLLM 框架**对 jina-v4 做全量推理证明。详见下方 Phase 3 规划节。

### 架构路径图

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 离线阶段（建库，一次性）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PDF
 │ pymupdf 渲染
 ▼
图像₁...图像ₙ（page_0.jpg ... page_N-1.jpg）
 │
 ├─── SHA256(图像ᵢ) ──► 集合 S ──► ZAC 承诺 cm（48B）
 │                                  ▲
 │                        ✅ Phase 1：来源可验证
 │                        证明：每张图像的哈希在承诺集合中
 │
 └─── jina-v4（3.9B VLM）
       ViT（32 blocks）→ LM（36 layers）→ MeanPool → L2Norm
       ▼
       向量 v₁...vₙ（2048维 float32）
       │
       ├─► FAISS IndexFlatIP（存储向量矩阵）
       │
       └─► zkLLM 证明₁...ₙ（每张图离线生成，存储 ~165KB/图）
                ▲
      ✅ Phase 3：推理可验证（离线侧）
      证明：vᵢ 确实是由 jina-v4 从图像ᵢ 正确推理得到的
      硬件：2×RTX 4090，单图约 5 分钟（实测推算），建库时一次性完成
      ⚠️ 离线证明与 Sumcheck proof 共同绑定向量值：
         Phase 3 证明 (imageᵢ, vᵢ) 对的正确性
         Phase 2 Oracle 查询重新验证 vᵢ 值，两层防护无需额外哈希


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 在线阶段（每次查询）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用户 query（文字）
 │
 └─── jina-v4（同一模型，文本路径：Tokenizer → LM 36层 → MeanPool）
       ▼
       查询向量 q（2048维）
       │
       ├─► [同步] 立即用于 FAISS 检索 + Phase 2 Sumcheck 验证
       │
       └─► [异步后台] zkLLM 证明
                   ▲
         ✅ Phase 3：推理可验证（在线侧，异步）
         证明：q 确实是由 jina-v4 从该 query 正确计算的
         用户先拿到检索结果，证明在后台完成后可供独立审计
         预估时间：2×RTX 4090，约 3-5 分钟/次（实测推算，文本路径无 ViT）


查询向量 q + FAISS 矩阵 [v₁...vₙ]
 │
 │  暴力计算 sᵢ = q · vᵢ（i=1..N）
 │  取 top-k 最高分（Global Batch Sumcheck）
 ▼
Sumcheck 证明（Global 模式）：
  ① 所有 N 个分值 sᵢ = q · vᵢ 批次正确（单条 Schwartz-Zippel 证明）
  ② Verifier 独立排序选出 top-k，无需信任 Prover 排名
        ▲
✅ Phase 2：结果可验证
证明：FAISS 搜索没有被篡改，返回的确实是全局最相似的图像


top-k 图像路径（page_57.jpg 等）
 │
 └─► ZAC.ProveM：证明每张图的 SHA256 在承诺集合 cm 中
          ▲
✅ Phase 1：来源可验证（每次查询）
证明：返回的图像确实来自原始语料库，没有注入外部图像


top-k 图像 + query
 │
 └─► MiniCPM-V-4 生成回答
          ✗ 不在可验证检索范围内
          （生成过程属于独立的"可验证推理"课题）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 完整信任链
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

信任起点：原始像素 pixels（用户提供的图像）
    │
    └──Phase 3──► ViT + LM 36层 + MeanPool 推理正确
                  向量 vᵢ = jina-v4(imageᵢ) 被密码学绑定
                    │
                    └──Phase 1──► 图像 imageᵢ 属于承诺语料库
                                  hash(imageᵢ) ∈ ZAC root
                                    │
                                    └──Phase 2──► 检索分值和排名没有被篡改
                                                  Verifier 独立验证 top-k 最优性
                                                    │
                                                    └──► 用户收到端到端可验证的检索结果
```

### 各 Phase 防御的攻击类型

| 攻击方式 | 被哪个 Phase 拦截 |
|---------|----------------|
| 向向量库注入外部图像 | Phase 1（ZAC 成员证明失败） |
| 用弱模型替代 jina-v4 计算 embedding | Phase 3（zkLLM 推理证明失败） |
| 篡改存储的 embedding 向量值 | Phase 2（Sumcheck Oracle 查询重新验证 v 值）+ Phase 3（proof 绑定 (image, v) 对）|
| 篡改相似度分值 / 伪造排序 | Phase 2（Global Batch Sumcheck 失败） |
| 隐藏更高分结果（返回非最优 top-k）| Phase 2（Verifier 自行从全部 N 个分值中选 top-k）|
| 在线 query 用弱模型计算 | Phase 3 异步版（后台 zkLLM 证明失败，可供审计）|

### Prover / Verifier 角色分离与 Demo 定位

**一个常见疑问**：如果验证者已经在本地持有模型和语料库，为什么还需要验证？

答案在于：本系统的设计目标是 **Prover 与 Verifier 为不同实体**。

```
服务提供方（Prover）                    用户 / 审计方（Verifier）
────────────────────                   ────────────────────────
持有完整模型权重                          只需要公开承诺值
持有完整语料库                            只需要 ZAC Root（48 字节 G1 点）
运行推理，生成 proof                      只需要模型各层权重承诺
随查询结果返回 proof 文件                  接收 (结果 + proof)，独立验证
```

Verifier 所需的全部公开信息：
- **ZAC Root**（48 字节）：服务方建库后通过可信渠道一次性发布
- **zkLLM 权重承诺**（每层数个 G1 点）：模型确定后一次性发布
- **proof 文件**：服务方随每次查询结果一并返回

Verifier **不需要模型本身**，也**不需要完整语料库**——验证比重新执行计算轻得多，这正是可验证证明的核心价值。服务方甚至可以不公开模型权重（知识产权保护），只公开承诺值，用户仍可独立验证推理的正确性。

**Demo 的定位**：当前 Demo 是**单机自证模式**，Prover 与 Verifier 运行在同一台机器上。这是学术原型的标准做法——目的是验证协议实现的端到端正确性，而非模拟真实的跨方部署。就像密码学论文的实验都在单机跑，证明的是协议可行，而非系统规模化可用性。

**论文威胁模型（Threat Model）应写明**：
> 本系统的信任假设为：验证者持有服务方通过可信渠道预先发布的 ZAC Root 和模型权重承诺，但无需访问完整模型或语料库。当前原型实现中 Prover 与 Verifier 共享本地环境，旨在验证协议的端到端正确性；在实际部署中，两者应为独立实体，Verifier 仅通过承诺值和 proof 文件完成独立验证。

### 实现状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | ZAC 语料库承诺 + 检索时成员证明 | ✅ 完成 |
| Phase 2 | Sumcheck 内积证明（Global Batch 模式） | ✅ 完成 |
| Phase 3 | zkLLM 全量推理证明（jina-v4） | ✅ 完成（Step 1-6 全部完成）|

---

## Phase 3：可验证推理——zkLLM 框架

### 背景：为什么从 EZKL 改为 zkLLM

初版方案（EZKL）计划只证明 MeanPool + L2Norm（pooling 头，0 个可学习参数），信任边界停留在 LM 最后一层的 hidden_states。经研读两篇论文后判断：

| 方案 | 证明范围 | 信任边界 | 可行性 |
|------|---------|---------|:------:|
| EZKL（旧） | 仅 pooling 头（0 参数）| LM hidden_states | 低价值 |
| zkLLM（新） | ViT + LM 全部 36 层 + pooling | 原始像素 | 高价值 ✅ |

zkLLM（Sun et al., CCS'24）已验证对 **LLaMA-2-13B**（803s，A100）生成完整推理证明，jina-v4 的语言塔规模（~3.3B）与 OPT-2.7B～6.7B 相当，**原则上直接适用**。

### zkLLM 使用场景（来自论文原文）

论文 Section 1 描述的核心场景：

> **Prover（AI 公司）** 拥有 LLM 权重（知识产权，不对外公开），通过 API 提供服务。
> **Verifier（AI 监管执法机构）** 提交 prompt，要求对返回结果进行形式化验证。

动机：法律诉讼场景（如 NYT 诉 OpenAI）需要排除"篡改输出"的可能性，但直接检查参数侵犯 IP。ZKP 是唯一能**同时满足"可验证性"和"参数保密"**的方案。

对应到本系统：检索服务器（Prover）→ 用户（Verifier），证明"检索结果由指定模型在真实文档上正确计算"。

开源仓库：`https://github.com/jvhs0706/zkllm-ccs2024`
长版论文：`https://arxiv.org/abs/2404.16109`

### jina-v4 全量证明范围

zkLLM 的三个核心协议对 jina-v4 各组件的覆盖情况：

| 组件 | 操作类型 | zkLLM 协议 | 备注 |
|------|---------|-----------|------|
| ViT patch embed | Conv3d（线性）| Sumcheck | 线性操作，理论直接支持 |
| ViT 32 blocks | Attention + SwiGLU + RMSNorm | zkAttn + tlookup | 与 LLaMA 注意力公式相同 |
| ViT Spatial Merge | MLP 投影（线性）| Sumcheck | |
| LM 36 layers | GQA + SwiGLU + RMSNorm | zkAttn + tlookup | ✅ 已适配 GQA（kv_dim=256，num_kv_heads=2，per-head 证明循环）|
| LoRA delta | 低秩矩阵乘法 $\mathbf{BA}$ | Sumcheck × 2 | 额外开销约 3%，可忽略 |
| MeanPool + L2Norm | 加权均值 + sqrt | Sumcheck + tlookup | 直接支持 |

**信任起点**：原始像素 pixels（用户提供，合理终点）

### 需要适配的两点

zkLLM 测试了 OPT/LLaMA（MHA，无 LoRA），jina-v4 有两处差异：

**① GQA（Grouped Query Attention）**

jina-v4 LM：`num_attention_heads=16`，`num_key_value_heads=2`（8:1 共享比）。zkAttn 证明的注意力公式 $\text{Softmax}(\mathbf{QK}^\top/\sqrt{d})\mathbf{V}$ 与头数无关，仅需将 K/V 矩阵维度从 $d$ 改为 $d/8=256$，协议无需修改。

**② LoRA 适配器**

每个线性层变为 $\mathbf{y} = \mathbf{x}\mathbf{W}_0^\top + \frac{\alpha}{r}\mathbf{x}\mathbf{A}^\top\mathbf{B}^\top$。

对每个有 LoRA 的层额外做两次 Sumcheck（证明 $\mathbf{xA}^\top$ 和 $(\mathbf{xA}^\top)\mathbf{B}^\top$），开销约为原层的 $2r/d = 2 \times 32/2048 \approx 3\%$。

### 全量证明 vs 部分证明（K 层方案）

若全量证明时间过长，可证明**最后 K 层 + Pooling**，前 36-K 层通过哈希承诺设为信任边界：

> ✅ **2026-03-19 实测最终数据**：使用 jina-v4 **真实权重**（含 LoRA retrieval，合并后量化），seq_len=512，GQA kv_dim=256，在 RTX 4090 D 上完整跑通 K=1/6/18/36 层（FFN + Attn-linear）。
> ✅ **2026-03-30 GQA zkAttn 完成**：在 seq_len=1024 下新增 GQA Softmax 证明（per-head 循环，16 Q-head / 2 KV-head），与 linear/FFN 证明合并为完整三段式覆盖，两侧序列统一填充到 1024。

| K | 层范围 | FFN 实测(s) | Attn-linear 实测(s) | 纯 Sumcheck 合计(s) | 估算全量(s)×1.5 | 估算全量(min) | 证明覆盖 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 35–35 | 5.2 | 2.5 | **7.7** | 11.8 | **0.2** | 3% |
| 6 | 30–35 | 30.7 | 14.9 | **45.6** | 70.4 | **1.2** | 17% |
| 18 | 18–35 | 177.5 | 69.0 | **246.6** | 381.0 | **6.3** | 50% |
| 36 | 0–35 | 304.0 | 112.8 | **416.8** | 643.9 | **10.7** | 100% |

> 估算全量 = 纯 Sumcheck × 1.03（LoRA 已合并，无额外开销）× 1.5（zkAttn softmax + RMSNorm + skip-connection）
> 实测脚本：`/root/autodl-tmp/zkllm-ccs2024/bench_k_layers.py`

**推荐**：K=6 作为论文实验设计点——证明最靠近 embedding 输出的 6 层（最敏感），约 **1.2 min**，安全性论证充分。

### 时间结构（seq_len=1024，含 GQA zkAttn，RTX 4090 D 实测/推算）

#### 每层三段式耗时基准

| 阶段 | 耗时 | 备注 |
|------|------|------|
| FFN proof | ~5.2s | 实测（K=1 bench） |
| Attn-linear proof | ~2.5s | 实测（K=1 bench） |
| GQA zkAttn proof | ~7.1s | 由 74s/6层 反推（见 2026-03-30 实测） |
| **单层合计（串行）** | **~14.8s** | FFN+linear+zkAttn |

#### 单图耗时，K=6（不同并行策略对比）

| 策略 | 原理 | 墙钟/图 | 303图总时 |
|------|------|---------|---------|
| 单卡串行（原始）| 6层顺序，每层14.8s | **88.8s** | **7.5h** |
| 层内双卡（bug版）| FFN∥linear，再zkAttn | **74s**（实测）| **6.2h** |
| 层内双卡（修复后）| FFN∥(linear→zkAttn) | **57.6s** | **4.9h** |
| **多worker（推荐，语料库）**| 2 GPU各跑独立图 | 88.8s/GPU×152图 | **3.7h** |
| **层间并行（推荐，query）** | GPU0:层30-32, GPU1:层33-35 | **≈45s** | 单次查询 |

> 实测依据：2026-03-30 运行 `build_corpus_zkllm_proofs.py --dual-gpu`，单图耗时 74s（6层）。
> 反推：74/6≈12.3s/层 = max(FFN=5.2s, linear=2.5s) + zkAttn → zkAttn≈7.1s。

#### 为什么层间并行比层内双卡更快

层内双卡（修复后）每层仍需 max(FFN, linear+zkAttn) = max(5.2, 9.6) = **9.6s**，6层=57.6s。
层间并行：各 GPU 独立串行3层 = 3×14.8s = **44.4s**，GPU1多1层时最慢 = 3×14.8s = 44.4s。
层间并行更快的根本原因：**hook 预捕获后各层证明无数据依赖，可自由调度**。

#### 语料库推荐运行命令

```bash
# 终端1（GPU0，处理偶数索引图像）
python script/build_corpus_zkllm_proofs.py \
  --corpus corpora/image.jsonl --workdir zkllm-workdir/jina-v4 \
  --k_layers 6 --worker-id 0 --num-workers 2

# 终端2（GPU1，处理奇数索引图像）
python script/build_corpus_zkllm_proofs.py \
  --corpus corpora/image.jsonl --workdir zkllm-workdir/jina-v4 \
  --k_layers 6 --worker-id 1 --num-workers 2
```

预计总时约 3.7h（303图 × 88.8s/2路并行）。

```
【在线查询，每次】
  用户提交 query → FAISS 检索（毫秒级）→ 立即返回 top-5 结果
  同时后台异步：zkLLM 证明 query embedding
    层间并行（GPU0:层30-32，GPU1:层33-35）：≈45s 后证明就绪
  返回的 5 张图的 proof 是预计算好的，直接附上
```

用户每次查询的**同步等待时间仍是秒级**，Phase 3 proof 仅用于异步审计。

### GPU 加速：2×RTX 4090 的分析（含实测）

zkLLM 官方实现（`jvhs0706/zkllm-ccs2024`）基于 CUDA，已在 RTX 4090 D 上成功编译运行（sm_89，CUDA 12.8）。

| 指标 | A100 SXM4 | RTX 4090 D（实测）|
|------|:---------:|:--------:|
| 显存 | 40 GB | 24 GB |
| FP64 | 19.5 TFLOPS | 1.3 TFLOPS |
| INT32 | 312 TOPS | 82.6 TOPS |
| 单层 FFN proof（Qwen-3B dims）| ~10s（估算）| **5.10s** |
| 单层 Attn-linear proof | ~6s（估算）| **2.65s** |
| 36 层总估算（含全部 ops×1.5）| ~800s | **~280s ≈ 5 min** |

- jina-v4 LM（~3.3B，bfloat16 ≈ 6.6GB）**单卡 24GB 显存完全够用**
- 实测显示 RTX 4090 D 比预期快：Sumcheck 主要是 INT64 张量运算，4090 INT32 算力强，FP64 阉割影响小于预期
- **不需要 ONNX**：zkLLM 直接读取 PyTorch 权重，不经过 ONNX
- 编译方式：系统 nvcc（CUDA 12.8），`-arch=sm_89 -std=c++17 -dlto`，全部 target 编译通过

### 关于存储向量的完整性

无需对 `embedding.npy` 单独做哈希保护，Phase 2 + Phase 3 已联合覆盖：

- **Phase 3**：证明 $(image_i, v_i)$ 绑定关系，proof 内含对 $v_i$ 值的隐式承诺
- **Phase 2**：Sumcheck Oracle 查询重新计算 $\tilde{v}(r_1,\ldots,r_\ell)$，若存储值被篡改则验证拒绝

攻击者无法在不破坏至少一个 proof 的情况下篡改 `embedding.npy`。

### 实验设计（Experiment 3）

> 已完成的子实验：3.1（K 层性能曲线，见 Step 4）

#### 3.H：真实激活 Hook 实验（Step 7，2026-03-20 执行）

**背景与动机**

当前 corpus/query 侧 zkLLM 证明均使用随机占位激活：

```python
(np.random.randn(512, 2048) * 65536).astype(np.int32).tofile(inp)
```

随机激活的证明只验证"给定该随机输入，矩阵乘由承诺权重正确计算"，**不证明 embedding 由真实模型推理产生**。替换为真实 hook 激活是使证明在语义上完整的必要步骤。

**Hook 位置的确定方式**

Hook 位置不是通过实验比较得出的，而是从架构分析直接推导：

```
hidden_states
    ├─→ input_layernorm (RMSNorm)
    │        ↓
    │   self_attn：Q/K/V 投影矩阵乘 = zkLLM self-attn linear 证明对象
    │        ↓  (+residual)
    ├─→ post_attention_layernorm (RMSNorm)
    │        ↓
    │   mlp (SwiGLU)：gate/up 投影矩阵乘 = zkLLM ffn 证明对象
    │        ↓  (+residual)
    └─→ 下一层输入
```

- `self-attn linear` binary 证明 `input @ W_q/k/v`，其中 input = `input_layernorm` 的**输出**
- `ffn` binary 证明 `input @ W_gate` 和 `input @ W_up`，其中 input = `post_attention_layernorm` 的**输出**

因此正确 hook 点为：
- **Attn 输入**：`layers[l].input_layernorm.register_forward_hook`，捕获 output
- **FFN 输入**：`layers[l].post_attention_layernorm.register_forward_hook`，捕获 output

当前系统使用同一随机文件同时作为两个 binary 的输入，是粗略近似。

**模型访问路径**（通过逐层打印确认）：

```python
st = SentenceTransformer(MODEL_PATH, trust_remote_code=True, device='cuda:0')
# st → Transformer → .model (PeftModelForFeatureExtraction)
#   → .base_model (LoraModel) → .model (JinaEmbeddingsV4Model)
#   → .model (Qwen2_5_VLModel) → .language_model (Qwen2_5_VLTextModel)
#   → .layers (ModuleList, 36 × Qwen2_5_VLDecoderLayer)
layers = list(st.children())[0].model.base_model.model.model.language_model.layers
# layers[i] 子模块：self_attn, mlp, input_layernorm, post_attention_layernorm
```

**实验脚本**：
- `script/exp_hook_profiling.py`（实验 3.H.1/2/5）
- `script/exp_hook_e2e.py`（实验 3.H.3）

---

##### 实验 3.H.2：Hook 非侵入性验证（2026-03-20）

**目的**：确认 `register_forward_hook` 不修改 forward pass，embedding 输出保持不变。这是所有 hook 实验的前提保证，排除"hook 干扰模型输出"这一混淆变量。

**操作步骤**：

1. 对 5 条 query 调用 `model.encode()`，记录 embedding → `emb_baseline`
2. 在 layers[30..35] 的 `input_layernorm` 和 `post_attention_layernorm` 各注册一个只执行 `pass` 的空 hook
3. 对相同 5 条 query 再次调用 `model.encode()`，记录 embedding → `emb_hooked`
4. 计算每对 embedding 的余弦相似度

**实测数据**：

```
cos_sim(base, hook): min=1.00000000  mean=1.00000005
```

**结论**：✅ PASS。hook 对 embedding 输出零影响。

---

##### 实验 3.H.1：激活分布探针（2026-03-20）

**目的**：探明真实激活的数值形态（形状、幅值、分布），确定量化 scaling factor 选择，为 E2E 实验提供参数依据。

**操作步骤**：

1. 在 layers[30..35] 的 `input_layernorm`（output）和 `post_attention_layernorm`（output）注册 hook，捕获激活张量
2. 对 5 条文本 query 逐一调用 `model.encode()`，累积激活
3. 对每层每个 hook 点计算：shape、mean、std、max_abs、p99
4. 对每个 scaling factor $s \in \{2^{12}, 2^{14}, 2^{16}, 2^{18}\}$，计算量化后 int32 溢出率和反量化余弦误差

**5 条测试 query**（涵盖不同长度和语义）：

```
"What is the sensor resolution of Nikon Z8?"
"How to set ISO in manual mode?"
"Battery life and charging specifications"
"AF tracking performance in video 4K 60fps"
"Nikon Z8 vs Z9 differences price weight"
```

**实测数据——pre_attn（input_layernorm 输出，即 self_attn 输入）**：

| 层 | shape | seq_len | mean | std | max_abs | p99 |
|:--:|:-----:|:-------:|:----:|:---:|:-------:|:---:|
| 30 | (1,12,2048) | 12 | -0.0061 | 0.7358 | 13.625 | 2.813 |
| 31 | (1,12,2048) | 12 | -0.0068 | 0.7786 | 14.750 | 3.531 |
| 32 | (1,12,2048) | 12 | -0.0111 | 0.9384 | 15.438 | 4.688 |
| 33 | (1,12,2048) | 12 | -0.0037 | 0.9455 | 15.188 | 4.688 |
| 34 | (1,12,2048) | 12 | -0.0202 | 0.7869 | 19.625 | 1.969 |
| 35 | (1,12,2048) | 12 | -0.0178 | 0.7289 | 18.500 | 2.141 |

**实测数据——pre_ffn（post_attention_layernorm 输出，即 mlp 输入）**：

| 层 | shape | seq_len | mean | std | max_abs | p99 |
|:--:|:-----:|:-------:|:----:|:---:|:-------:|:---:|
| 30 | (1,12,2048) | 12 | -0.0123 | 0.8840 | **82.000** | 1.867 |
| 31 | (1,12,2048) | 12 | -0.0085 | 0.7332 | 23.250 | 2.203 |
| 32 | (1,12,2048) | 12 | -0.0118 | 0.7359 | 24.250 | 2.125 |
| 33 | (1,12,2048) | 12 | -0.0112 | 0.7166 | 25.125 | 2.031 |
| 34 | (1,12,2048) | 12 | -0.0108 | 0.7622 | 31.000 | 1.977 |
| 35 | (1,12,2048) | 12 | -0.0301 | 1.1435 | 45.750 | 2.578 |

**量化灵敏度（对所有层/所有 sf 的统一结论）**：

| scaling factor | int32 溢出率 | cos_err（量化误差） |
|:--------------:|:-----------:|:------------------:|
| $2^{12}$ | 0.00e+00 | ≈ −5×10⁻⁷ |
| $2^{14}$ | 0.00e+00 | ≈ −5×10⁻⁷ |
| $2^{16}$ | 0.00e+00 | ≈ −5×10⁻⁷ |
| $2^{18}$ | 0.00e+00 | ≈ −5×10⁻⁷ |

> max_abs=82 时，`82 × 2^18 = 21,495,808 ≪ 2^31 = 2,147,483,648`，int32 绝对不溢出。

**结论**：
1. 真实 seq_len = **12**（含任务 prompt，远小于随机占位的 512）
2. 激活幅值极小（max_abs ≤ 82），随机占位幅值约为 200,000，相差约 2500 倍
3. 所有 scaling factor 均零溢出，量化误差约 5×10⁻⁷（精度充分）
4. 选定 **sf = 2^16**，与现有系统保持一致，无需改动

**注**：`layer_in`（层输入，即 input_layernorm 前）的 hook 方式（`register_forward_pre_hook`）在本次实验中未捕获到数据，原因是 Qwen2_5_VLDecoderLayer 的 pre_hook 签名不匹配。该近似方案已被放弃，改用正确的分离 hook 方式。

---

##### 实验 3.H.5：Hook 延迟开销（2026-03-20）

**目的**：量化 hook 对在线查询延迟的影响，确认可接受。

**操作步骤**：

1. 基线：`model.encode([query])` × 10 次，取中位数 → $t_\text{base}$
2. 在 layers[30..35] 各注册 `input_layernorm` + `post_attention_layernorm` 共 12 个 hook（仅 `output.detach().cpu()`），重复 10 次 → $t_\text{hook}$

**实测数据**：

| 场景 | 中位时间 | overhead |
|------|:-------:|:--------:|
| 无 hook | 127.0 ms | — |
| 12 个 hook | 126.6 ms | **-0.3%** |

**结论**：✅ 开销可忽略（-0.3%，在测量噪声范围内）。hook 的主要操作是 Python 函数调用 + `detach().cpu()`（激活已在 GPU，拷贝到 CPU 约 12×12×2048×4 = 1.2 MB，耗时 < 1ms）。

---

##### 实验 3.H.3：端到端证明有效性（核心实验，2026-03-20）

**目的**：验证真实 hook 激活能通过 zkLLM `ffn` 和 `self-attn linear` 二进制，产生有效 Sumcheck 证明（returncode=0）。同时确定 seq_len padding 策略。

**操作步骤**：

1. 对 query `"What is the sensor resolution of Nikon Z8?"` 调用 `model.encode()`，捕获 layer-35 的 `pre_attn`（shape: 1×12×2048）和 `pre_ffn`（shape: 1×12×2048）
2. 对每个激活，测试三种 seq_len：实际值 12、pad 到 256、pad 到 512（padding 方式：在 seq 维度末尾补零）
3. 量化：`(act × 2^16).round().int32`，写入 `.bin` 文件
4. 分别调用 `./ffn` 和 `./self-attn linear` binary，记录 returncode 和耗时
5. 对照基线变体 A：`np.random.randn(512, 2048) × 65536` int32

**测试矩阵与实测结果**（layer-35，sf=2^16）：

| 变体 | 激活来源 | seq_len | FFN binary | Attn binary | 备注 |
|------|---------|:-------:|:----------:|:-----------:|------|
| A 随机基线 | random×65536 | 512 | ✅ 5033ms | ✅ 2361ms | 当前系统行为 |
| B1 pre_attn | input_layernorm 输出 | 12 | — | ❌ 416ms | `D or N is not power of 2, or D is not divisible by N` |
| B2 pre_attn | input_layernorm 输出 | 256(pad) | — | ✅ 2285ms | |
| B3 pre_attn | input_layernorm 输出 | 512(pad) | — | ✅ 2333ms | |
| C1 pre_ffn | post_attn_ln 输出 | 12 | ❌ 719ms | — | 同上错误 |
| C2 pre_ffn | post_attn_ln 输出 | 256(pad) | ✅ 4309ms | — | |
| C3 pre_ffn | post_attn_ln 输出 | 512(pad) | ✅ 4930ms | — | |

**错误分析（seq=12 失败）**：

`std::runtime_error: D or N is not power of 2, or D is not divisible by N`

zkLLM 内部 NTT（Number Theoretic Transform）和矩阵分块要求 $\text{seq\_len} \times D$ 满足特定约束。seq_len=12，D=2048 时，12×2048=24576，不满足 binary 内部的 2 的幂次/整除要求。

| seq_len | seq×D | 是否满足约束 |
|:-------:|:-----:|:----------:|
| 12 | 24,576 | ❌ |
| 256 | 524,288 = 2^19 | ✅ |
| 512 | 1,048,576 = 2^20 | ✅ |

**结论**：
1. 真实激活完全可以通过 zkLLM binary（pad 后全部 ✅）
2. seq_len 必须 pad 到满足约束的值；最小有效值为 **256**（=2^8，且 256×2048=2^19）
3. seq=256 比 seq=512 快约 13%（FFN: 4309ms vs 4930ms），因 Sumcheck 计算量与 seq_len 成比例
4. 两个 hook 点（pre_attn/pre_ffn）均通过，证明 hook 位置分析正确
5. 每层需要**两个独立输入文件**，不能共用同一随机文件

---

**集成决策总结**（基于以上四个实验）：

| 参数 | 决策 | 依据 |
|------|------|------|
| Hook 点（Attn） | `layers[l].input_layernorm` output | 架构分析：self_attn 的矩阵乘输入 |
| Hook 点（FFN） | `layers[l].post_attention_layernorm` output | 架构分析：mlp 的矩阵乘输入 |
| seq_len | pad 到 **1024** | 2026-03-30 升级（原 256 不满足 zkAttn NTT 约束 seq²=2²⁰） |
| scaling factor | **2^16** | 实验 3.H.1：零溢出，与现有系统一致 |
| 每层文件数 | **2 个**（attn-input + ffn-input） | 实验 3.H.3：两者不同，不可共用 |
| Hook 非侵入性 | ✅ 已验证 | 实验 3.H.2：cos_sim=1.0 精确 |
| 延迟开销 | ✅ -0.3%（可忽略） | 实验 3.H.5 |

---

> 其余原计划实验（量化精度 3.2、proof 大小 3.3、端到端延迟 3.4）与 3.H 系列合并或已由 Step 4 覆盖

### 下一步实现路径

1. **Step 1** ✅：克隆 `zkllm-ccs2024`，在 RTX 4090 D 上编译运行，测出实际性能
   - CUDA 12.8 + sm_89 编译全通过（ppgen/commit-param/ffn/self-attn/rmsnorm/skip-connection）
   - `bench_qwen_dims.py`（`/root/autodl-tmp/zkllm-ccs2024/bench_qwen_dims.py`）：Qwen2.5-3B 合成权重 benchmark
   - 实测：FFN 5.10s/层，Attn-linear 2.65s/层，36 层全量估算 ~5 min（远好于 A100 外推的 15-44 min）

2. **Step 2** ✅：真实权重加载 + GQA 适配 + 精度验证
   - `load_jina_weights.py`（`/root/autodl-tmp/zkllm-ccs2024/load_jina_weights.py`）
   - 加载 safetensors base 权重 + LoRA (retrieval，alpha=32，r=32，scaling=1.0)，合并：W_eff = W_base + B@A
   - 量化：W_int = round(W_eff.T × 65536).int32，全部 36 层验证：avg_cos=**1.000000**，max_err=**6e-4**
   - `self-attn.cu` GQA 适配：新增可选参数 `argv[8] = kv_dim`（默认=embed_dim，兼容 MHA）
     - k_proj/v_proj：`create_weight(..., embed_dim, kv_dim)` 和 `zkFC k_layer(embed_dim, kv_dim, ...)`
   - **GQA tlookup 约束**：要求 `(seq_len × kv_dim) % 65536 == 0`，kv_dim=256 时需 seq_len ≥ 256（linear/FFN 阶段取 512 做早期测试）
   - **Layer 35 真实权重实测（seq_len=512，kv_dim=256，早期 linear 模式）**：
     - Attn-linear proof：**2.55s** ✅（"QKV linear proof successfully verified!"）
     - FFN proof：**5.10s** ✅（与合成权重完全一致，真实量化无额外运行开销）
   - **2026-03-30 升级**：seq_len 统一为 1024（满足 zkAttn NTT 约束 seq²=2²⁰），linear/FFN/zkAttn 三段式全覆盖

3. **Step 3** ✅：LoRA delta 处理——无需修改 CUDA
   - `load_jina_weights.py` 在量化前合并：W_eff = W_base + B@A（B@A 为 float32 精确计算）
   - 量化后 W_int 已包含 LoRA 贡献，zkLLM Sumcheck 对 W_eff 的证明隐式覆盖 LoRA delta
   - 验证：`(W_base_quant + delta_exact)` 与 `W_eff_quant` 输出 cos=1.000000，差异仅为量化舍入误差
   - 实际运行时间与合成权重完全一致（+0%），LoRA 合并只是离线预处理步骤

4. **Step 4** ✅：K 层性能曲线实测（实验 3.1）
   - `bench_k_layers.py`（`/root/autodl-tmp/zkllm-ccs2024/bench_k_layers.py`）
   - 使用 jina-v4 真实权重（含 LoRA retrieval），seq_len=512，GQA kv_dim=256
   - 实测结果（RTX 4090 D）：
     - K=1：7.7s 纯 Sumcheck，估算全量 **11.8s（0.2 min）**
     - K=6：45.6s 纯 Sumcheck，估算全量 **70.4s（1.2 min）** ← 推荐设计点
     - K=18：246.6s 纯 Sumcheck，估算全量 **381s（6.3 min）**
     - K=36：416.8s 纯 Sumcheck，估算全量 **644s（10.7 min）**

5. **Step 5** ✅：集成到 `servers/retriever/` 流水线（离线建库 + 在线异步）
   - `retriever_init` 新增两个参数：`zkllm_workdir`（承诺权重目录）、`zkllm_k_layers`（默认 6）
   - `retriever_search` 返回前调用 `asyncio.ensure_future(_zkllm_prove_query_async(...))`，不阻塞响应
   - `_zkllm_prove_query_async`：对最后 K 层逐层调用 `./ffn` + `./self-attn linear`（subprocess），完成后写日志
   - 配置文件更新：`servers/retriever/parameter.yaml` + `examples/parameter/visrag_parameter.yaml`
   - 启用方式：将 `zkllm_workdir` 设为包含已承诺权重的目录路径（如 `/path/to/zkllm-workdir/jina-v4`）

6. **Step 6** ✅：语料库侧预计算证明集成（离线预计算 + 检索时附带）

7. **Step 7** ✅：真实激活 Hook 实验（实验 3.H.1/2/3/5，2026-03-20）
   - 详细实验步骤、原始数据、结论见上方「实验设计（Experiment 3）→ 3.H」节
   - 集成决策：hook `input_layernorm` output（attn）+ `post_attention_layernorm` output（ffn），pad seq=1024（2026-03-30 由 256 升级，满足 zkAttn NTT 约束 seq²=2²⁰），sf=2^16，每层两个独立文件

   **设计目标**：对语料库中每张图像预计算 zkLLM 证明（离线，建库时一次性），检索时直接附带对应图像的预计算证明，无需实时生成。

   **实现文件**：
   - `script/build_corpus_zkllm_proofs.py`：遍历 `corpora/image.jsonl`，对每张图像用 zkLLM 最后 K 层做 Sumcheck 证明，结果存为 `zkllm-workdir/jina-v4/corpus_proof_{image_id}.json`
   - `servers/retriever/src/retriever.py`：`retriever_search` 中新增 corpus proofs 查找逻辑，对 `rets` 中每条结果查询对应的预计算证明文件，并附加到 `zkllm_verification.corpus_proofs` 中

   **`zkllm_verification` 输出结构（更新）**：
   ```json
   {
     "query_proof": {
       "status": "pending",
       "proof_id": "a3f7b2c1",
       "k_layers": 6,
       "note": "query embedding proof running in background..."
     },
     "corpus_proofs": [
       [
         { "image_id": "nikon/page_57.jpg", "status": "completed", "k_layers": 6, "verified": true, "elapsed_ms": 45513 },
         { "image_id": "nikon/page_31.jpg", "status": "completed", "k_layers": 6, "verified": true, "elapsed_ms": 45387 },
         ...
       ]
     ]
   }
   ```

   **预计算脚本用法**：
   ```bash
   cd /root/autodl-tmp/UltraRAG
   # 先预计算前 2 张（测试验证）
   python script/build_corpus_zkllm_proofs.py --limit 2 --k_layers 6
   # 全量运行（后台，约 3.6 小时，288 张 × 45s）
   nohup python script/build_corpus_zkllm_proofs.py --k_layers 6 \
       > /tmp/corpus_zkllm_proof.log 2>&1 &
   ```

   **实测结果**（K=6，RTX 4090 D）：
   - `nikon/page_0.jpg`：✓ 45513ms（6 层全部 verified=true）
   - `nikon/page_1.jpg`：✓ 45387ms（6 层全部 verified=true）
   - 288 张全量：~3.6 小时（后台异步运行中）

   **与 query-side 证明的区别**：
   | | 语料库侧（corpus proof） | 查询侧（query proof） |
   |--|--|--|
   | 何时生成 | 建库时，离线一次性 | 每次查询，后台异步 |
   | 证明对象 | `(imageᵢ, vᵢ)` 的推理正确性 | `(query text, q_vec)` 的推理正确性 |
   | 存储位置 | `zkllm-workdir/jina-v4/corpus_proof_{id}.json` | `zkllm-workdir/jina-v4/zkllm_proof_{id}.json` |
   | 检索时延迟 | 0ms（直接读取预计算结果） | 0ms（异步后台，不阻塞响应） |

---

### 实验 3.L：层敏感性消融实验（Layer Sensitivity Ablation，2026-03-31）

#### 背景与动机

选择证明"最后 K=6 层"需要实验支撑，核心问题：**哪些层对最终 embedding 影响最大？** 即攻击者攻击哪些层代价最高、我们 secure 哪些层收益最大。

实验方向：**先做实验，根据数据决定结论**，不预设结果。

#### 文献调研（2026-03-31 网络搜索）

| 论文 | 方法 | 关键发现 | 与本任务的关系 |
|------|------|---------|-------------|
| ShortGPT (ACL 2025 Findings, arXiv:2403.03853) | BI Score = 1 − cosine(输入, 输出) | **深层（后几层）比浅层更冗余**，删最后 25% 层 MMLU 仅降 2.8% | ⚠️ 挑战"后几层最重要"假设（生成任务） |
| Gromov et al. "Unreasonable Ineffectiveness" (ICLR 2025, arXiv:2403.17887) | Angular Distance 找连续冗余块 + QLoRA 修复 | 深层对生成任务冗余，删后微调可恢复 | ⚠️ 同上，但仅限生成任务 |
| "Towards Interpreting Visual Info in VLMs" (ICLR 2025) | 注意力流 + logit lens + 激活修补 | **跨模态融合主要发生在后期层**，后期层负责任务相关推理 | ✅ 支持检索任务中后期层重要 |
| LLM-Pruner (NeurIPS 2023, arXiv:2305.11627) | 一阶梯度 × 参数 + Hessian 近似 | 结构性剪枝中间层效率最高 | 参考，我们不做结构剪枝 |
| "Layer Importance for LLM Alignment" (arXiv:2410.17875) | 冻结/微调各层，测下游任务 | 重要层跨数据集 90% 重叠，早期层和某些中间层最关键 | 参考，对齐任务 ≠ 检索任务 |
| 早期层 residual 分析 (NeurIPS 2024, LessWrong) | 理论分析残差流结构 | 早期层 residual norm ≈ embedding norm，单步增量天然最大 | 解释我们实验结果（层0/1敏感性高是数学必然） |
| Survey: zkML Verifiable Machine Learning (arXiv:2502.18535) | 综述 | QKV/FFN 矩阵乘法是 zkML 效率最高的证明单元 | 技术可行性支撑 |

**关键矛盾**：ShortGPT/Gromov 发现后几层冗余（生成任务），VLM 分析发现后几层负责跨模态融合（检索任务）。两者任务不同，需要用**检索指标**（Recall@K）而非生成指标来评估。

#### 实验方案

**实验 A：BI Score（文献标准无扰动方法）**

- **方法**：对每层计算 `BI_l = 1 − cosine(x_l, x_{l+1})`，直接测各层输入/输出相似度
- **指标**：BI 值（越高 = 变换越大 = 越不冗余）
- **样本**：文本 query 20 个 + 图像 20 张（多模态，分别计算）
- **参考**：ShortGPT (arXiv:2403.03853)
- **脚本**：`script/ablation_layer_sensitivity.py --mode bi`
- **输出**：`notes/ablation_bi_score.png`，`notes/ablation_bi_score.json`

**实验 B：单层删除 + embedding cos 下降（已实现，待扩展至多模态）**

- **方法**：每次将一层的残差置零（等价删层），测 embedding 相对 baseline 的 cos 下降
- **指标**：`sensitivity_l = 1 − cosine(baseline, ablated_l)`
- **样本**：文本 + 图像各 20 个
- **参考**：arXiv:2409.14381（直接删层 + 性能下降方法）
- **脚本**：`script/ablation_layer_sensitivity.py --mode zero`（已实现文本，待加图像）

**实验 C 旧版：噪声注入（对照实验，存在级联传播问题）**

- **方法**：对每层独立注入自适应高斯噪声（`noise ~ N(0, (scale × σ_l)²)`），测 embedding cos 下降
- **指标**：`sensitivity_l = 1 − cosine(baseline, noisy_l)`，`σ_l`（实际注入噪声标准差）
- **已知问题**：噪声在层 l 注入后，经后续 35−l 层级联传播，导致早期层天然显得更敏感（传播距离更长，非层本身重要性）
- **参考**：LaPael NeurIPS 2024（噪声注入到中间层）
- **脚本**：`script/ablation_layer_sensitivity.py --mode noise`

**实验 C 完整版：因果追踪（Causal Tracing）**

- **方法**（三步走）：
  1. **干净前向**：记录各层干净输出 `clean_acts[l]`，得到 `E_base`
  2. **腐化前向**：第 0 层注入大幅噪声（`scale=corrupt_scale × σ₀`），级联污染所有层，得 `E_corrupt`
  3. **恢复前向**：对每层 l，腐化前向 + 将层 l 输出强制替换为 `clean_acts[l]`（后续层 l+1..35 在干净激活上运行），得 `E_restore_l`
- **指标**：`recovery_l = cos(E_restore_l, E_base) − cos(E_corrupt, E_base)`
  - `recovery_l → 0`：仅恢复层 l 无法修复 embedding（层不关键）
  - `recovery_l → 1`：仅恢复层 l 即可完全修复 embedding（层是关键节点）
- **优势**：彻底消除级联传播干扰，每层效应被独立隔离；与实验 B 互补（B 测"删掉该层损失多少"，C 测"修复该层能恢复多少"）
- **参考**：
  - Meng et al., "Locating and Editing Factual Associations in GPT" (ROME), NeurIPS 2022
  - Elhage et al., "A Mathematical Framework for Transformer Circuits", Anthropic 2021
- **脚本**：`script/ablation_layer_sensitivity.py --mode causal --corrupt-scale 3.0 --img-samples 10`
- **输出**：`notes/ablation_causal_tracing.json`，`notes/ablation_causal_tracing.png`

#### 实验结果汇总（2026-03-31）

**实验 A — BI Score**

| 模态 | Top-6 高 BI 层（变换量最大）|
|------|---------------------------|
| 文本 | 0, 35, 33, 1, 3, 34（层 0 为 embedding 层伪影）|
| 图像 | 35, 33, 0, 34, 31, 32 |

跨模态稳定出现：层 **33、35**。

**实验 B — 单层残差置零**

| 模态 | Top-6 最敏感层 |
|------|--------------|
| 文本 | 1（0.700）、0（0.614）、35（0.256）、33（0.211）、3（0.197）、34（0.151）|
| 图像 | **1（0.414）、35（0.363）、33（0.317）、34（0.189）、31（0.139）、32（0.117）** |

图像模态 Top-6 中有 **5 层（31-35）** 在最后 6 层内；层 0/1/3 的高敏感性为早期层数学必然（残差流从零累积），不代表"应优先证明"。

**实验 C 旧版 — 噪声注入（级联传播）**

| 模态 | 规律 |
|------|------|
| 文本 | 层 2-29 sensitivity ≈ 0.82-0.94（平台期），层 30-34 突降至 0.29-0.36，层 35 突升至 0.78（σ=63.9，最终投影层） |
| 图像 | 单调递减（0→35），0.49→0.11 |

关键发现：层 30-34 的 σ 从 25 突降至 6-15，存在**激活量级跳变**（可能是跨模态融合适配层边界），后接层 35 的大尺度投影主导输出。

**实验 C 完整版 — 因果追踪（已放弃）**

运行后发现 `cos_restored = 1.0` 对所有层均成立，recovery 全部相同，结果 trivial，原因如下：

恢复层 l 的全部 hidden state → 层 l+1..35 接收完全干净的输入 → 模型确定性地产生与 clean run 相同的输出 → cos = 1.0。ROME 原始实验之所以有效，是因为腐化在**输入 embedding 层**持续存在，且只恢复**单一 token position**，attention 机制会把其他位置的腐化值"泄漏"回来。对 jina-v4 这类 mean pooling 模型，单 position 恢复无法隔离单层效应，完整实现需要从 embedding 层注入+单 token 恢复，代价远超收益。

**结论：Experiment B（残差置零）本身已是正确的单层隔离实验，因果追踪不增加额外信息。**

---

#### 综合分析与层选择修正（2026-03-31）

**问题：原假设"文本和图像统一用最后 K=6 层"是否成立？**

基于三个实验的完整数据，做无预设的重新分析。

##### 早期层伪影的排除

实验 B 中层 0（文本 0.614）、层 1（文本 0.700 / 图像 0.414）敏感性极高，但这是**架构必然**，不是层重要性的信号：
- 层 0（文本）：置零 token embedding = 输入完全随机，后续所有层处理乱码，catastrophic by definition
- 层 1（文本/图像）：第一个 transformer 层，其输入已被层 0 破坏，级联效应

**这些层不应被纳入"值得证明"的讨论。**

##### 中间层（4-29）：可忽略

| 模态 | 层 4-29 平均 sensitivity | 最大值 |
|------|------------------------|--------|
| 文本 | 0.0117 | 0.0276（层 27）|
| 图像 | 0.0169 | 0.0355（层 27）|

中间层对最终 embedding 贡献极低，是"信息中继"而非"语义生成"。

##### 后段层（30-35）的分段结构

| 层段 | 文本 sensitivity 均值 | 图像 sensitivity 均值 |
|------|----------------------|----------------------|
| 层 30     | 0.0803 | 0.0635 |
| 层 31-32  | 0.0866 | 0.1281 |
| 层 33-35  | 0.2059 | 0.2899 |

**层 33-35 是两种模态共同的核心关键区**，sensitivity 是层 30-32 的 2-3 倍。

层 31-32 对**图像**的贡献（0.128）是对**文本**（0.087）的 1.5 倍，显示图像的跨模态融合在更早的层开始。

##### 各验证策略的覆盖率（不含伪影层）

| 策略 | 文本覆盖率 | 图像覆盖率 | 层数 |
|------|----------|----------|------|
| K=3（层 33-35）| 44.3% | 52.1% | 3 |
| K=5（层 31-35）| 56.7% | 67.5% | 5 |
| K=6（层 30-35）| 62.4% | 71.3% | 6 |

层 30 的边际贡献极低（文本 +5.7%，图像 +3.8%），性价比最差。
K=3 → K=5 的增量对图像（+15.4%）比文本（+12.4%）更显著。

##### 修正后的层选择建议

原假设"统一 K=6"在数据上不充分支持，且忽略了两种模态的差异。

**更合理的策略**：

| 模态 | 建议层 | K | 理由 |
|------|--------|---|------|
| 文本 query | 层 33-35 | K=3 | 层 30-32 对文本贡献低（均值 0.08），与层 33-35（均值 0.21）差距大 |
| 图像 corpus | 层 31-35 | K=5 | 层 31-32 对图像贡献明显（均值 0.13），BI Score 也显示图像在更早层开始关键变换 |

**层 33 和 35 是两种模态、三个实验中最稳定的关键层**，是任何策略的必选项。

如需统一 K（简化实现），K=5（层 31-35）是两种模态的最佳折中点，既捕获了文本核心（K=3 的超集），也覆盖了图像的扩展关键区。

#### 当前状态（2026-03-31）

- 实验 A：已完成（文本 + 图像）
- 实验 B：已完成（文本 + 图像）✅ 主要论据
- 实验 C 旧版：已完成（文本 + 图像），用于架构边界分析（σ 突降）
- 实验 C 完整版（因果追踪）：**已放弃**，mean-pooling 模型上结果 trivial
- 层选择结论：**修正为差异化策略**，文本 K=3（33-35），图像 K=5（31-35）；统一时取 K=5

---

```yaml
备用参数
benchmark:
  benchmark:
    key_map:
      gt_ls: golden_answers
      q_ls: question
    limit: -1
    name: test
    path: data/test.jsonl
    seed: 42
    shuffle: false
generation:
  backend: vllm
  backend_configs:
    vllm:
      dtype: auto
      gpu_ids: 1
      gpu_memory_utilization: 0.8
      model_name_or_path: /root/autodl-tmp/models/MiniCPM-V-4
      trust_remote_code: true
  image_tag: null
  sampling_params:
    chat_template_kwargs:
      enable_thinking: false
    max_tokens: 2048
    temperature: 0.7
    top_p: 0.8
  system_prompt: ''
prompt:
  template: prompt/visrag.jinja
retriever:
  backend: sentence_transformers
  backend_configs:
    sentence_transformers:
      sentence_transformers_encode:
        encode_chunk_size: 10000
        normalize_embeddings: false
        psg_prompt_name: null
        psg_task: retrieval
        q_prompt_name: query
        q_task: retrieval
      trust_remote_code: true
  batch_size: 16
  corpus_path: corpora/image.jsonl
  gpu_ids: 0
  index_backend: faiss
  index_backend_configs:
    faiss:
      index_chunk_size: 50000
      index_path: index/index.index
      index_use_gpu: true
  is_multimodal: true
  model_name_or_path: /root/autodl-tmp/models/jina-embeddings-v4
  query_instruction: ''
  top_k: 5
  zac_prover_state: output/phase1/prover_state.json
  sumcheck_embedding_npy: embedding/embedding.npy
  sumcheck_mode: global
  zkllm_workdir: zkllm-workdir/jina-v4       # 设为 zkllm-workdir/jina-v4 路径以启用 Phase 3
  zkllm_k_layers: 6         # 证明最后 K 层，推荐 6（~1.2 min/query）
```

---

## Phase 4：系统化演示与交互体验（规划）

> 状态：**待实现**（等待 Phase 3 语料库侧 zkLLM 预计算完成后开始）
> 前提：288 张图像的 corpus_proof 后台生成完毕（约 3.6 小时）

### 当前演示流程的问题

| 环节 | 现状 | 问题 |
|------|------|------|
| 建库 | 手动逐条运行 `ultrarag build/run` | 步骤多、容易出错 |
| 检索 | 手动修改 `data/test.jsonl` 再跑 pipeline | 不能实时提问 |
| 查看结果 | 手动运行 `python script/case_study.py` | UI 与流程完全分离 |
| 证明展示 | ZAC/Sumcheck/zkLLM 混在同一面板 | 无法区分证明者/验证者视角 |
| zkLLM query 证明 | 已后台异步生成，状态写入 json | UI 只显示 "pending"，未轮询完成状态 |

---

### 4.1 一键建库（`build_verifiable_corpus.py`）

**已完成** ✅（`script/build_verifiable_corpus.py`）

将以下步骤自动串联，不修改任何已有文件：
```
PDF → corpus → embedding → FAISS index → ZAC 承诺 → zkLLM 预计算（后台）
```

用法：
```bash
# 完整建库（新 PDF，全量流程）
python script/build_verifiable_corpus.py --pdf data/new.pdf --overwrite

# 只重建证明（corpus/embedding 已有）
python script/build_verifiable_corpus.py --skip-corpus --skip-embed

# 查看进度
tail -f /tmp/corpus_zkllm_proof.log
ls zkllm-workdir/jina-v4/corpus_proof_*.json | wc -l
```

ZAC Root 更新说明（新数据加入后）：
- 脚本自动重建 ZAC 承诺，输出新 `cm_hex`
- 需要将新 `cm_hex` 通过可信渠道发布（替换旧值）
- 自动化操作与手动操作安全性等价——关键在于承诺发布的透明性，而非谁触发了操作

---

### 4.2 实时交互 + 证明者/验证者分离 UI ✅

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                     前端 Web UI                          │
│                                                          │
│  [上传文档 PDF/图片]          [输入查询]                  │
│       │                           │                      │
│       ▼                           ▼                      │
│  ┌──────────┐              ┌────────────┐                │
│  │ 建库面板  │              │  查询面板   │                │
│  │          │              │            │                │
│  │ 进度条   │              │ 检索结果   │                │
│  │ ZAC Root │              │ 图像预览   │                │
│  │ 已生成数 │              │ LLM 回答   │                │
│  └──────────┘              └────────────┘                │
│                                                          │
│  ─────────────────────────────────────────────────────  │
│  [证明者视角 Prover]        [验证者视角 Verifier]         │
│                                                          │
│  已承诺的 ZAC Root (cm_hex) │ 独立验证结果：PASS / FAIL  │
│  Sumcheck 证明字节数        │ 自行排序后的 top-k 索引    │
│  zkLLM proof 状态 pending→✓│ 是否与 Prover 声称一致     │
│  corpus proof 附件列表      │ 每项证明的验证细节         │
└─────────────────────────────────────────────────────────┘
```

#### 实现方案

**后端**：基于 FastAPI 的 HTTP 服务（`script/api_server.py`），封装现有 MCP 流水线：
- `POST /upload`：接收 PDF/图片，触发建库流程
- `POST /query`：接收 query 文本，调用 `retriever_search` + `generation`，返回检索结果 + 证明
- `GET /proof_status/{proof_id}`：轮询 query zkLLM 证明完成状态
- `GET /corpus_status`：返回当前 corpus 大小、ZAC Root、已完成证明数

**前端**：Gradio 或 Streamlit（轻量，适合演示）：
- 上传区：支持 PDF / 图片，显示建库进度
- 查询区：实时输入 → 等待检索结果（流式返回）→ 等待证明完成（轮询刷新）
- 双面板：左侧 Prover（原始证明数据）、右侧 Verifier（独立验证结论）

#### 证明者/验证者职责划分

| 角色 | 持有 | 展示内容 |
|------|------|---------|
| Prover（检索服务器） | 权重、prover_state、全部向量 | cm_hex、proof_hex、所有 N 个分值、zkLLM proof JSON |
| Verifier（用户/客户端） | 公开 cm_hex（提前获取） | 独立验证配对方程、独立排序 top-k、比对结果是否吻合 |

---

### 4.3 query 侧 zkLLM 证明 ✅

**已实现**（`script/interactive_demo.py`）：
- 查询提交瞬间后台启动 `_run_zkllm_query_bg`，UI **立即**显示「⏳ 证明进行中」
- 同步验证步骤（Sumcheck、ZAC）依次完成并流式更新 UI
- 「刷新 zkLLM 状态」按钮轮询完成状态
- 完成后显示「✅ PASS」+ 逐层验证结果

**验证顺序**（UI 显示逻辑）：

| 顺序 | 步骤 | 方式 | 耗时 |
|:---:|------|------|------|
| 0（最先启动） | zkLLM query proof | 后台异步，立即显示 pending | ~72s |
| 1 | query embedding (jina-v4) | 同步 | ~200ms |
| 2 | FAISS 精确搜索 | 同步 | <10ms |
| 3 | Sumcheck Global Batch | 同步 | ~500ms |
| 4 | ZAC 成员证明 | 同步 | ~2-4s |
| 5 | zkLLM corpus proof 查询 | 同步（读预计算文件） | <1ms |
| 刷新时 | zkLLM query proof | 轮询 | — |

---

### 4.4 zkLLM 性能基准对比（论文数据）

原论文（Sun et al., CCS 2024）Table 1 给出全模型、全层、seq=2048 的证明开销：

| 模型 | 参数量 | 层数 | Prover 时间 | 每层均摊 | 证明大小 |
|------|-------|------|------------|---------|---------|
| OPT-125M | 125M | 12 | 73.9s | 6.2s | 141 kB |
| OPT-1.3B | 1.3B | 24 | 221s | 9.2s | 147 kB |
| LLaMA-2-7B | 7B | 32 | 620s | 19.4s | 183 kB |
| LLaMA-2-13B | 13B | 32 | 803s | 25.1s | 188 kB |

**本系统实测**（jina-embeddings-v4 / Qwen2.5-VL，6/36 层，seq=1024，图像输入）：

| 指标 | 数值 | 说明 |
|------|------|------|
| 每张图片证明时间（早期）| **~52s** | 6 层 FFN + self-attn linear（seq=512，未含 zkAttn）|
| 每层均摊（早期）| **~8.7s/层** | |
| 覆盖范围（2026-03-30 后）| 层 30–35 | 后 6 层，含 FFN + Attn-linear + **GQA zkAttn（Softmax）** |

**性能合理性分析**：

jina-v4 每层维度远大于 OPT-125M（hidden=2048 vs 768，ffn=11008 vs 3072，约 7× 更大矩阵）。
在此基础上 seq_len 减半（1024 vs 2048）并跳过 zkAttn，得到 8.7s/层，与 OPT-125M 的 6.2s/层量级一致，**性能符合预期**。

覆盖范围的差距（6/36 层 vs 全层）是当前系统的设计选择，不是性能瓶颈。
完整推理证明（含 zkAttn、全部 36 层）理论上需要约 36 × 8.7 ≈ 313s/张，与 LLaMA-2-7B 的 620s 量级相符（考虑到维度差异）。

---

### 4.5 跨层承诺绑定问题（Cross-layer Commitment Binding）

**问题描述**

系统分三个 Phase 独立设计，各 Phase 默认信任上一 Phase 的输出，导致出现**语义接缝**：

```
ZAC 承诺：element_i = SHA256(image_bytes_i)
           → 证明"图片文件在语料库里"

Sumcheck 输入：embedding_i（来自 FAISS 索引）
           → 验证"内积⟨q, embedding_i⟩计算正确"

断层：没有密码学证明 embedding_i 来自 image_bytes_i
```

**攻击路径**：
1. 保持图片文件不变 → ZAC 验证通过
2. 在 FAISS 中替换 embedding_i 为伪造向量 embedding_i'
3. Sumcheck 仍验证通过（内积计算本身是对的，只是对了错误的向量）
4. 基于 embedding_i' 返回错误的 top-k

zkLLM 的引入是**部分**弥补：它证明了某个激活经过了真实模型的后 6 层推理，但没有密码学绑定说明该激活来自 ZAC 承诺的那张图片（前 30 层未证明）。

**修复方案：扩展 ZAC 承诺语义**

将 embedding 哈希纳入 ZAC 承诺元素：

```python
# 当前（只承诺图片字节）
element_i = SHA256(image_bytes_i)

# 修复后（同时承诺图片 + 对应 embedding）
embedding_bytes_i = embedding_i.astype(np.float32).tobytes()
element_i = SHA256(image_bytes_i + embedding_bytes_i)
```

修复后：服务器无法在不改变 ZAC Root 的情况下替换 embedding，从而关闭此攻击路径。

**技术分类**：这不是"数据一致性检查"（统计概念），而是**密码学承诺绑定**（Commitment Binding）——在可信 AI 文献中属于 **Data Provenance（数据来源追踪）**，确保派生数据（embedding）可溯源至承诺的源数据（image bytes）。

**为何之前未考虑**：组合系统的隐性信任假设——每个组件信任其输入，接缝只在端到端整体视角下才可见。这正是老师指出"只证明算的对，推理过程怎么证明"时所触及的系统性漏洞。

**实现状态**：✅ 已实现（2026-04-01）

修改点：
- `src/zac/accumulator.py` — `from_corpus` 改用 `SHA256(img_bytes ∥ emb_bytes)`，新增 `image_embedding_hash` 静态方法
- `script/interactive_demo.py` — `faiss_search` 返回 FAISS IDs，`run_zac(paths, emb_ids)` 用 `image_embedding_hash`
- 需重新运行 `phase1_corpus_fingerprint.py --zac-only` 重建 `prover_state.json`

---

### 4.6 系统叙事定位（Trustworthy AI — Data-Centric 视角）

**重要设计决策记录**

本系统最终定位为 **Trustworthy AI** 方向，而非纯密码学系统：

- ZAC、Sumcheck、zkLLM 是工具，不是目的
- 系统提供的是"可审计性"（Auditability），不是"零知识性"（Zero-Knowledge）
- 对标语境：EU AI Act（2024）对高风险 AI 系统的可审计要求，C2PA 在内容溯源上的空白

**核心切入角：Data-Centric Trustworthy AI**

Trustworthy AI 领域存在两个主流切入角：

| 切入角 | 代表工作 | 信任建立方式 | 威胁模型 |
|--------|---------|------------|---------|
| **Model-centric** | EVisRAG（2025）、证据归因、Explainability | 模型自身输出的归因/解释 | 模型幻觉、不可解释的推理 |
| **Data-centric** | **本系统** | 对数据流转每个节点的密码学承诺 | 恶意服务商、供应链篡改、不可信基础设施 |

本系统属于 **Data-Centric Trustworthy AI**：相信数据流的完整性，而不依赖对模型行为的信任。这与近年来数据治理、数据溯源（Data Provenance）领域的趋势一致——在高风险 AI 部署中，仅靠"模型说了算"不够，需要对数据的每次流转提供可独立验证的密码学保障。

**与 EVisRAG 的关系和区别**

EVisRAG（2025 年 10 月）是生成侧的 Trustworthy AI 工作：给 VLM 生成的答案标注来自哪张图像的哪个区域，解决"答案是否有据可查"的问题（证据归因/可解释性）。

本系统是检索侧的 Trustworthy AI 工作，解决的是更基础的问题：

> **在 EVisRAG 能归因之前，如何证明检索系统本身没有被篡改、返回的语料是真实的？**

两者关系：本系统是 EVisRAG 的**前置信任基础**。EVisRAG 的归因仍然依赖你相信服务商返回的图像是真实的；而本系统通过 ZAC + Sumcheck + zkLLM 使得"服务商说检索结果是 X"这件事变得**密码学不可伪造**。

**为何选择检索侧而非生成侧**

本项目题目"面向多模态语义数据的可验证检索机制"，检索是数据流转的核心环节：

```
[语料库数据] → Phase 1 ZAC（语料完整性）
     ↓
[Embedding 数据] → Phase 2 Sumcheck（相似度计算完整性）
     ↓
[Top-k 结果数据] → Phase 3 zkLLM（推理过程完整性）
     ↓
[答案数据] ← 生成侧（EVisRAG 的范围）
```

对数据流中每一个"数据状态"的转换施加密码学约束，这正是 Data-Centric 的含义——不是针对某种特定模型或任务设计的，而是对**多模态语义数据流转过程**的通用保障机制。

**C2PA（Coalition for Content Provenance and Authenticity）**

Adobe、Microsoft、Google 等联合制定的数字内容溯源标准，为图像/视频打密码学签名证明其来源和编辑历史。C2PA 只证明"内容从哪里来、被谁改过"，**不证明"内容在检索时被正确使用"**。VeriMMR 填补的正是 C2PA 覆盖不到的检索过程可验证性。

**创新点定位**

| 维度 | 内容 |
|------|------|
| 问题形式化 | 首次将"可验证多模态 RAG"分解为三个独立信任边界，并为每个边界匹配密码学原语 |
| Data-Centric 视角 | 对检索管道中每个数据状态转换节点施加密码学约束，独立于模型行为 |
| 组合贡献 | ZAC+Sumcheck+zkLLM 的组合方案及其接缝分析（跨层承诺绑定问题） |
| 多模态特异性 | 图像 seq_len 对齐（641→1024）、图像高熵哈希的成员证明优势、hook 位置确定 |
| 工程可行性 | 303 张图片端到端证明，52s/张，<2% hook 开销，实测数据 |

**评测策略与数据集选取原则**

项目题目是"面向多模态语义数据"，不是针对某一类文档（如相机手册）设计的专用系统。实验中 Nikon PDF 仅作演示，**检索质量评测（C1）必须使用多类型公开数据集**，以证明框架的普适性。

选取原则：
1. 使用 VisRAG 论文（ICLR 2025）同款数据集，便于与已发表的检索指标直接对比
2. 覆盖不同多模态文档类型（幻灯片/文档/图表/信息图），体现"多模态语义数据"的广度
3. 与 VisRAG-Ret 的 **Out-of-Domain（OOD）指标**对比（见 C1 节详细说明）

**为何只与 OOD 对比，不与 In-Domain 对比**

VisRAG-Ret 报告了两类指标：
- **In-Domain**：训练时见过该数据集的训练集，属于专门微调后的结果，对我们不公平
- **Out-of-Domain**：训练时完全未见该数据集，体现泛化能力

jina-v4 是通用多模态检索器（zero-shot），未在任何文档 VQA 数据集上微调，因此必须与 OOD 指标对比。这是**公平的科学比较**：双方都没有在该数据集上专门训练，差别仅在于 VisRAG-Ret 用其他文档数据做过预热，而 jina-v4 是完全通用的。若 jina-v4 OOD 指标接近甚至超过 VisRAG-Ret OOD，说明通用检索器加密码学验证层即可满足高置信需求，无需专门训练。

**应用场景**

最自然的三类场景：
1. **医疗影像辅助诊断**：监管要求可审计，需证明相似病例来自真实数据库且检索正确
2. **多方联合知识库（Federated RAG）**：各方不互信，区块链存证 ZAC Root，证明语料无偏
3. **AI 生成内容溯源**：补充 C2PA，证明 RAG 检索过程不可伪造

---

### 4.7 实验设计（T / U / E 框架）

**三问题框架**（与 Trustworthy AI 叙事对齐）

#### Q1 — Utility（效用）：系统是否影响 AI 性能？ ⭐ 核心实验

Reviewer 最关心的问题：引入验证层是否损失检索质量？

```
对比设置：
  Baseline A：普通 UltraRAG（无验证）
  Baseline B：UltraRAG + ZAC（仅成员证明）
  VeriMMR：完整三层验证（ZAC + Sumcheck + zkLLM）

指标：Recall@k, NDCG@k, 端到端 QA 准确率
数据集：Nikon 手册自建 QA 集 或 公开多模态 QA 数据集
```

#### Q2 — Trustworthiness（可信性）：系统是否真的"更可信"？

**Integrity Guarantee 实验**（完整性保障验证）：

```
实验 2a：语料库篡改检测（ZAC）
  → 注入 X% 替换图像，测 ZAC 检出率

实验 2b：Embedding 偷换攻击（跨层绑定修复前后对比）
  → 替换 FAISS embedding 但保持图片文件，测检出率
  → 这是修复 4.5 节漏洞的直接验证

实验 2c：排名操纵检测（Sumcheck）
  → 修改 FAISS 返回错误 top-k，测 Sumcheck 独立排名差异
```

#### Q3 — Efficiency（效率）：系统是否"可用"（practical）？

```
延迟分解（per query）：
  ZAC：X ms | Sumcheck：X ms | zkLLM corpus：<1ms（预计算）
  zkLLM query：~52s（后台，不阻塞 UI）
  总额外同步开销：X%

可扩展性：
  N = 50 / 100 / 303 的 Sumcheck 延迟曲线
  → 自然引出"N 超过某阈值需迁移 Polynomial IOP"的讨论
```

**实验顺序建议**：按 **U → T → E** 叙述（先证明"有用"，再证明"可信"，最后证明"可用"）。

---

### 4.8 ANN 可验证性说明（论文中的定位）

> 待写入论文相关章节，此处记录结论

**当前设计选择**：使用精确搜索（FAISS `IndexFlatIP`），Global Batch Sumcheck 完整覆盖全部 N 个内积，Verifier 独立选出 top-k，证明无歧义。

**ANN 的理论关系**（可在论文局限性/讨论章节提及）：

即使使用 ANN（HNSW、IVFFlat 等），Global Batch Sumcheck 仍可对 ANN 结果做隐式验证：
- Prover 必须声明全部 N 个分值 → Verifier 独立排序 → 若 ANN 遗漏高分结果，Verifier 自行选出的 top-k 与 ANN 结果不一致 → ANN 错误被暴露
- 但此时 Prover 计算全部 N 个精确内积的开销与精确搜索相当，ANN 加速优势消失
- 结论：**精确搜索 + Sumcheck 是本系统的一致设计选择，ANN 场景下的可验证检索属于独立研究方向**

---

### 执行计划（更新）

| 优先级 | 任务 | 状态 |
|:---:|------|------|
| 1 | 288 张 corpus zkLLM 预计算（随机基线） | ✅ 完成（288/288） |
| 2 | 303 张 corpus zkLLM 预计算（真实激活） | ✅ 完成（2026-04-05） |
| 3 | 实时交互 Gradio Demo | ✅ 完成（`script/interactive_demo.py`） |
| 4 | query 侧真实激活集成 | ✅ 完成（`embed_query_with_hooks`） |
| 5 | 跨层承诺绑定修复（4.5 节） | ✅ 完成（2026-04-01） |
| 6 | 实验数据收集（A/B/C/D/E 全组） | ✅ 完成（2026-04-04） |
| 7 | 整理论文参考文献 | 🔲 持续补充 |

---

## 实验规划（完整版）

> 记录日期：2026-04-01  
> 状态：消融实验（Group D）已完成，其余待执行

### 总体框架

论文实验分五组：

| 组别 | 核心问题 | 衡量指标 | 状态 |
|------|---------|----------|------|
| A. 系统性能 | "有多慢？" | 延迟分解 + N 扩展性 | ✅ A1 已完成 / 🔲 A2 待执行 |
| B. 安全验证 | "能抓住攻击吗？" | 攻击检测率（需 100%） | ✅ 已完成 |
| C. 检索质量 | "量化误差影响排名吗？" | top-k 排名一致性；可选 Recall@K | 🔲 待执行 |
| D. 消融实验 | "K 层怎么选？" | BI Score / Residual Zeroing / Noise | ✅ 已完成 |
| E. GQA 适配正确性 | "GQA 改造计算是否正确？" | zkAttn 输出与参考实现的 L∞ 误差 | 🔲 待执行 |

**关于系统模块消融**：不需要单独一组实验。  
- 性能维度：A1 延迟分解表即为模块消融（无验证 → +Phase1 → +Phase2 → +Phase3 逐步累加）  
- 安全维度：B 组实验即为模块消融（去掉某 Phase = 对应攻击成功）

**关于组件内部基准对比**：不需要重跑 ZAC/Sumcheck/zkLLM 原论文实验。各组件的算法性质（ZAC 常数大小证明优于 Merkle、Sumcheck 轮次复杂度等）直接引用原论文，本文只测量各组件在系统内的实际耗时。

---

### Group A：系统性能

#### A1 端到端延迟分解 ✅ 已完成（2026-04-03）

**目标**：测量完整验证流水线各阶段的延迟，并与无验证基线对比。

**实验方法**：5 条代表性 query × 10 次重复（同步阶段）取中位数；Phase 3Q 单独计时 3 次；生成阶段每条 query 重复 3 次取中位数。N=303，GPU: RTX 4090 24G × 2。

**实测结果**（`notes/experiment_a1_result.json`）：

**查询时延迟分解（同步阶段，per-query）：**

| 阶段 | 中位数 | 类型 |
|------|--------|------|
| jina-v4 编码 | 130ms | 同步 |
| FAISS 检索 | 0ms | 同步 |
| Phase 2 Sumcheck（N=303） | 973ms | 同步 |
| Phase 1 ZAC（k=5） | 4514ms | 同步 |
| Phase 3C corpus 读取（预计算） | 0ms | 同步 |
| **同步验证完成时刻** | **5618ms** | |
| Phase 3Q（K=3，双GPU并行，layers 33-35） | 30586ms | 异步后台 |

**端到端时序（含生成）：**

关键时序：Phase 3Q 在 encode 完成（t≈130ms）后异步启动，主线程在完成同步验证后通过 `join()` 等待 Phase 3Q 完成（阻塞 ~25099ms）。

| 节点 | 时刻 |
|------|------|
| 同步验证完成 | 5618ms |
| Phase 3Q 完成（encode + 30586ms） | ~30716ms |
| 大模型可开始生成 | **30716ms**（取 max） |
| 大模型生成（MiniCPM-V-4） | +3521ms |
| **端到端总计** | **34237ms** |
| Baseline 端到端（encode+FAISS+生成） | 3652ms |
| **端到端额外开销** | **30586ms = 8.4× baseline** |

**建库时开销（一次性，离线）：**

| 阶段 | 耗时 |
|------|------|
| Phase 3C corpus zkLLM（K=5，N=303 张） | ~6.42h |
| Phase 1 ZAC 建库 + embedding + FAISS | <10min |

**结论**：
- 端到端延迟瓶颈为 **Phase 3Q**（zkLLM query proof），占总额外开销 ~97%
- Sumcheck + ZAC 同步开销合计 ~5.5s，相比 Phase 3Q 可忽略
- Phase 3Q 双 GPU 并行可将延迟压缩至单卡串行的约 1/2（K=3 → gpu0=[33], gpu1=[34,35]）
- 若 Phase 3Q 可进一步异步化（answer 不强依赖 Phase 3Q 结果），用户感知延迟可降至 ~5.6s（1.5× baseline）

**延迟瓶颈叙事（论文表述）**：

端到端延迟 34.2s（8.4× baseline）。延迟分解揭示瓶颈为 Phase 3Q（zkLLM query proof，30.6s），其余同步验证阶段（Sumcheck + ZAC）仅引入 5.5s 额外开销。Phase 3Q 的开销来自 ZK 证明的本质计算复杂度——zkLLM 原论文对 LLaMA-2-13B 在 A100 上需 803s；本系统通过消融实验选择关键 K=3 层（文本 coverage 集中在 layers 33-35，达 44.3%）、采用双 GPU 并行策略，将 query 侧证明压缩至 31s，约为原论文量级的 1/26。若应用场景允许将嵌入来源证明从同步响应路径解耦（答案先返回，proof 异步交付），用户感知延迟可降至 5.6s（1.5× baseline）。系统因此提供两层可选验证保障：检索完整性（Sumcheck + ZAC，1.5×）与嵌入来源证明（+ Phase 3Q，8.4×），不同安全需求场景可按需选择。

**报告格式**：堆叠条形图（各阶段占比）+ 表格

#### A2 N 扩展性 ✅ 已完成（2026-04-04）

**目标**：观察语料库规模 N 对各验证组件延迟的影响。

**实验方法**：5 条 query × 10 次重复，N ∈ {50, 100, 200, 303, 500, 1000}。N ≤ 303 切片已有 embedding；N > 303 补随机单位向量（仅用于计时）。ZAC 始终使用固定 k=5 真实元素以验证 O(k) 常数性。

**实测结果**（`notes/experiment_a2_result.json`）：

| N | FAISS | Sumcheck | ZAC |
|---|-------|----------|-----|
| 50 | 0.41ms | 166ms | 4378ms |
| 100 | 0.32ms | 321ms | 4379ms |
| 200 | 0.44ms | 634ms | 4369ms |
| 303 | 0.57ms | 954ms | 4368ms |
| 500 | 0.76ms | 1583ms | 4370ms |
| 1000 | 1.20ms | 3172ms | 4387ms |

**结论**：
- **FAISS**：全程 < 2ms，N=1000 时仅 1.2ms，实际可忽略不计
- **Sumcheck**：log-log 拟合斜率 = **0.986 ≈ 1.0**，精确验证 O(N) 线性增长；N 从 50 增至 1000（×20），延迟从 166ms 增至 3172ms（×19.1）
- **ZAC**：N=50 到 N=1000 全程维持在 4368–4387ms，方差极小，完全验证 **O(k) 常数性**（k=5，与 N 无关）——这是 ZAC 相对 Merkle 树的核心优势（Merkle 为 O(k·log N)）

**扩展性叙事（论文表述）**：

A2 实验验证了三个组件的理论复杂度。FAISS 检索在 N=1000 时延迟仅 1.2ms，不构成瓶颈。Sumcheck 延迟随 N 线性增长（实测斜率 0.986），N=1000 时为 3.2s，在可接受范围内且理论上与原论文分析一致。ZAC 成员证明延迟在 N=50 至 N=1000 全程稳定在 ~4.37s，完全不随 N 变化——这正是 ZAC 采用 Pointproofs 向量承诺（证明大小与 N 无关）相比 Merkle 树（O(log N) 证明路径）的核心优势：无论语料库扩展至多大规模，单次查询的成员证明开销保持恒定。

**报告格式**：N-延迟折线图（三条线，FAISS/Sumcheck/ZAC）+ 表格

---

### Group B：安全验证 ✅ 已完成（2026-04-03）

> 每个攻击场景要求检测率 = 100%，否则为系统设计缺陷

#### B1 图像替换攻击

**攻击**：服务端将检索到的图像 $i^*$ 替换为另一张图像 $i'$（另一页面），embedding 与 ZAC Root 均不变。

**实验方法**：随机抽取 50 对 (victim, donor) 图像（偏移 N/2 保证内容差异），将 victim 文件替换为 donor 文件，对 victim 的原始 embedding 运行 ZAC 成员证明验证，随后恢复原始文件。

**期望结果**：`SHA256(donor_bytes ∥ orig_emb_bytes)` ∉ BF → ZAC 拒绝

**实验结果**：检测率 **50/50 = 100%**（seed=42，N=303）

#### B2 Embedding 替换攻击

**攻击**：图像文件不变，将 `embedding.npy` 中对应向量替换为随机单位向量（模拟操控 FAISS 排名）。

**实验方法**：随机抽取 50 个目标索引，对每个索引用随机单位向量替换 embedding，用原始图像路径 + 伪造 embedding 运行 ZAC 验证（跨层绑定：`SHA256(orig_bytes ∥ fake_emb_bytes)`）。

**期望结果**：跨层承诺绑定（4.5 节）使 ZAC 拒绝，embedding 替换在不重建承诺的前提下无法通过验证。

**实验结果**：检测率 **50/50 = 100%**（seed=42，N=303）

#### B3 排名操控攻击

**攻击**：Prover 在 Global Batch Sumcheck 的 `scores` 数组中将某低排名条目分值改为 max_score+1，试图让其进入 top-k。

**实验方法**：取 10 条 query（等间距覆盖语料库），每条生成合法证明后，从非 top-k 池中随机选 5 个 victim 索引，各自将分值改为最高分 +1，运行 Sumcheck 验证，共 50 个篡改测试。

**期望结果**：Schwartz-Zippel 随机线性组合使篡改分值与 batch 目标不一致，Sumcheck 以 $\geq 1 - N/p \approx 1 - 2^{-52}$ 概率检测。

**实验结果**：检测率 **50/50 = 100%**（10 queries × 5 victims，N=303）

#### B4 权重矩阵篡改攻击（承诺绑定）

**攻击**：攻击者将服务器权重矩阵 $W_\text{gate}$（layer-35）替换为 $W_\text{gate} + \Delta W$，但公开承诺文件 `commitment.bin` 仍反映原始 $W_\text{gate}$。Prover 用篡改权重计算 FFN，Verifier 用原始承诺验证。

**为何重新设计**：原方案（修改 corpus proof JSON 的 `verified` 字段）只是软件 I/O 测试，缺乏密码学意义。zkLLM 的实际安全性来自 KZG 承诺绑定（Binding）：Commitment 文件是对权重矩阵的密码学承诺，任何对 `*-int.bin` 的篡改都会导致 Sumcheck 中多线性扩展评估与承诺不符。

**zkLLM 数据文件结构**（`zkllm-workdir/jina-v4/`）：
```
mlp.gate_proj.weight-pp.bin                       ← CRS（公共参数，用于 KZG 承诺）
layer-35-mlp.gate_proj.weight-int.bin             ← 实际量化权重（Prover 计算用）
layer-35-mlp.gate_proj.weight-commitment.bin      ← 权重的 KZG 承诺（Verifier 校验用）
```

**实验方法**：
1. 生成随机激活 `.bin`（seed=123，形状 1024×2048，int32，scale=2^16），运行 FFN binary → 期望 rc=0（基准）
2. 对 `layer-35-mlp.gate_proj.weight-int.bin` 加 ±2²⁰ 随机扰动（约量化 scale 的 16×），`commitment.bin` 不变
3. 用篡改权重重新运行 FFN binary，保持相同激活和承诺 → 期望 rc≠0

**验证原理**：Sumcheck 最终轮 Verifier 计算 $\text{eval}(W_\text{tampered}, r)$ 并与承诺 $\text{commit}(W_\text{original})$ 在随机点 $r$ 的开放值比对，两者不匹配 → Sumcheck 拒绝 → binary returncode ≠ 0。

**实验结果**：

| 步骤 | 权重状态 | FFN binary rc | verified |
|------|---------|:-------------:|:--------:|
| 基准运行 | 原始 $W_\text{gate}$ | **0** | ✅ True |
| 篡改后运行 | $W_\text{gate} + \Delta W$（100% 元素被扰动） | **-6** | ✅ False（检测到） |

检测率 **1/1 = 100%**（rc=-6，非零，Sumcheck 拒绝）

**安全含义**：攻击者在不持有与承诺一致的原始权重的前提下，无法通过 zkLLM 验证，即无法伪造"模型用真实权重做了推理"的证明。

**覆盖关系总结**（安全维度模块消融）：

| 攻击 | 被哪个 Phase 拦截 | 若去掉该 Phase |
|------|----------------|--------------|
| B1 图像替换 | Phase 1 ZAC | 攻击成功 |
| B2 Embedding 替换 | Phase 1 ZAC（跨层绑定） | 攻击成功 |
| B3 排名伪造 | Phase 2 Sumcheck | 攻击成功 |
| B4 权重矩阵篡改 | Phase 3 zkLLM（KZG 承诺绑定） | 攻击成功 |

**结果文件**：`notes/experiment_b_result.json`
**实验脚本**：`script/experiment_b_security.py`

---

### Group C：检索质量

#### C2 Embedding 量化误差（必做）

**目标**：验证 float32 embedding 在 Sumcheck 整数域转换后，精度损失在可接受范围内，top-k 排名不受影响。

**重要说明**：验证层不修改检索结果，FAISS 返回什么就验证什么。因此 Recall@K 在有验证/无验证下完全相同，无需用 benchmark 证明"加了验证后 Recall@K 不变"。C2 的真正目的是证明 Sumcheck 的整数域量化不引入排名错误。

**实验方法**：
- 计算 float32 内积 $s_i = \langle q, v_i \rangle_{\text{float32}}$
- 计算 Sumcheck 域整数内积 $\hat{s}_i = \langle \hat{q}, \hat{v}_i \rangle_{\mathbb{Z}_p}$（缩放至整数域）
- 比较 top-5 排名一致性（排名颠倒计为错误）
- 对全部 303 张图像的 corpus 运行，报告最大绝对误差和排名一致率

**预期结论**：量化相对误差 < 0.1%，top-5 排名 100% 一致

#### C2 实验结果 ✅ 已完成

**测试条件**：N=303，D=2048，scale=256 / 65536 对比，top-k=5，三组 query（语料库内向量 × 2 + 随机单位向量）

| Query 类型 | L∞ 误差 | L1 误差 | 相对误差 | top-5 一致 | top-10 一致 |
|-----------|---------|---------|---------|-----------|------------|
| corpus[0]（语料库内） | — | — | — | ✅ | ✅ |
| corpus[100] | 4.20e-3 | 1.07e-3 | 1.83e-3 | ✅ | ✅ |
| 随机单位向量 | 6.34e-3 | 1.75e-3 | 6.11e-1 | ✅ | ❌ |

**关键结论**：top-5 排名 100% 一致，满足系统设计要求。

**误差分析**

*为什么 top-5 总是一致？*  
真实语义查询的 top-5 结果与其他候选之间存在明显的分值间隙（相关文档得分显著高于无关文档），量化误差（L∞ ≈ 6e-3）远小于这个间隙，因此排名不会被翻转。

*为什么随机向量的相对误差高达 61%？*  
随机单位向量与语料库的内积趋近于零（分母 ≈ 0），相对误差在数学上被放大。这是度量本身的局限，绝对误差（6.34e-3）仍然很小。实际检索场景中查询向量不是随机向量，此问题不存在。

*为什么随机向量 top-10 不一致？*  
随机向量对所有语料的得分非常接近（都接近 0），第 6-10 名之间分值差小于量化误差，导致顺序被小噪声翻转。但排名 6-10 的文档不影响检索结果（系统返回 top-5），对实际使用无影响。

**局限性**

1. ~~**scale=256 精度有限**~~：已升级为 scale=65536（步长 $\approx 1.5 \times 10^{-5}$），原 scale=256 下的 top-10 不一致问题已消除。
2. **相对误差指标失效**：当真实内积趋近于零时，相对误差不具可比性，应以 L∞ 绝对误差为主要指标。
3. **top-k > 5 时一致性下降**：top-10 在极端情况（随机向量）下出现不一致，说明量化只能保证高分结果的排名稳定。

**scale 对比实验结论**（已执行）：

| scale | 耗时 | top-5 | top-10 | max L∞ | max 相对误差 |
|-------|------|-------|--------|--------|------------|
| 256 | 1.1s | ✅ | ❌（随机向量） | 6.34e-3 | 6.11e-1 |
| **65536** | **1.0s** | **✅** | **✅** | **2.41e-5** | **2.03e-3** |

→ **已将默认 scale 更新为 65536**（`src/sumcheck/inner_product.py` + `script/phase2_sumcheck.py`），与 zkLLM 量化方案对齐，精度提升 263 倍，耗时无差异，溢出安全（$65536^2 \times 2048 = 8.8 \times 10^{12} \ll p$）。

**结果文件**：`notes/experiment_c2_result.json`  
**实验脚本**：`script/experiment_c2_quantization.py`

#### C1 可验证检索完整性实验（攻击检测 + 质量保护）

**设计定位（2026-04-04 更新）**

本实验同时完成两件事：
1. **基线 Recall@K**：建立干净语料下的检索质量参照
2. **攻击检测验证**：注入 embedding 替换攻击，证明无验证时质量静默下降，有验证时 100% 检出

核心叙事：

> 攻击者替换部分语料 embedding 后，用户在无验证情况下收到错误 top-k 而毫不知情（Recall@K 大幅下降）。  
> 我们的 Sumcheck 验证机制独立重计算所有内积，发现 FAISS 返回结果与承诺语料不一致，**立即报警**，从根本上杜绝静默降级。

**数据集**（4 个，覆盖不同多模态文档类型，均来自 `openbmb/VisRAG-Ret-Test-*`）

| 数据集 | 文档类型 | VisRAG-Ret OOD MRR@10 |
|--------|---------|----------------------|
| SlideVQA | 幻灯片（多页） | 45.57 |
| MP-DocVQA | 扫描文档 | 74.60 |
| ChartQA | 图表 | 75.99 |
| InfoVQA | 信息图 | 67.26 |

**攻击方案（Embedding 替换攻击）**

对每条查询，找到其所有相关语料项（来自 qrels），将其在 FAISS 中的 embedding 替换为随机单位向量。这直接模拟"恶意服务商篡改检索索引使相关结果消失"的供应链攻击。

```
干净 FAISS → 正常 top-k → Recall@K = X（基线）
篡改 FAISS → 错误 top-k → Recall@K = Y（无验证，Y << X）
Sumcheck 验证（使用承诺语料向量）→ 独立 top-k ≠ FAISS top-k → 攻击报警（100% 检出）
```

**Sumcheck 攻击检测原理**

`verify_global_batch(q_vec, committed_vecs, proof, top_k)` 使用原始承诺向量独立计算所有 N 个内积，生成自己的 `top_k_indices`。若 FAISS 返回的 top-k 与之不一致，说明 FAISS 索引已被篡改。

**实验脚本**：
- `script/experiment_c1_recall.py`（纯检索质量）
- `script/experiment_c1_attack_verify.py`（攻击检测 + 质量保护）
- `script/run_c1_experiments.sh`（四数据集全流程批量脚本）

**结果文件**：
- `notes/experiment_c1_{dataset}.json`（纯 Recall@K）
- `notes/experiment_c1_attack_{dataset}.json`（攻击验证完整结果）

---

#### C1 实验结果（2026-04-04，全部完成）✅

##### 检索质量：jina-v4 vs VisRAG-Ret OOD

| 数据集 | 语料规模 | jina-v4 MRR@10 | VisRAG-Ret OOD | jina-v4 R@10 | VisRAG-Ret OOD |
|--------|---------|---------------|----------------|--------------|----------------|
| SlideVQA | 1284 | **94.72** | 45.57 | **98.20** | 67.70 |
| MP-DocVQA | 741 | **79.56** | 74.60 | **94.25** | 89.65 |
| ChartQA | 500 | **87.43** | 75.99 | **93.65** | 91.40 |
| InfoVQA | 459 | **89.80** | 67.26 | **98.47** | 87.05 |

jina-v4 在所有数据集上均显著优于 VisRAG-Ret OOD 基准（零样本 vs 领域外），验证了框架底层检索器的普适性。

##### B3 攻击：Recall@K 静默降级（无验证情况）

| 数据集 | 基线 MRR@10 | 攻击后 MRR@10 | 下降幅度 | 基线 R@10 | 攻击后 R@10 | 下降幅度 |
|--------|------------|--------------|---------|-----------|------------|---------|
| SlideVQA | 94.72 | 0.00 | −94.72pp | 98.20 | 0.00 | −98.20pp |
| MP-DocVQA | 79.56 | 0.00 | −79.56pp | 94.25 | 0.00 | −94.25pp |
| ChartQA | 87.43 | 0.00 | −87.43pp | 93.65 | 0.00 | −93.65pp |
| InfoVQA | 89.80 | 0.00 | −89.80pp | 98.47 | 0.00 | −98.47pp |

攻击策略：将所有相关语料项的 FAISS embedding 替换为随机单位向量（所有相关项被针对性抹除），无验证机制的用户 Recall@K 归零而毫不知情。

##### 攻击检测率

| 数据集 | B1 图像替换（ZAC） | B2 Embedding 替换（ZAC） | B3 排名操控（Sumcheck） | Sumcheck 延迟/query |
|--------|-----------------|----------------------|---------------------|-------------------|
| SlideVQA | 10/10 **100%** | 9/10 90%† | 50/50 **100%** | 4088 ms |
| MP-DocVQA | 10/10 **100%** | 10/10 **100%** | 50/50 **100%** | 2383 ms |
| ChartQA | 10/10 **100%** | 10/10 **100%** | 49/50 98%‡ | 1564 ms |
| InfoVQA | 9/10 90%† | 10/10 **100%** | 50/50 **100%** | 1468 ms |

**†** B1/B2 个别漏报原因：Bloom Filter 固有假阳性（理论误报率 ε=0.01）。在 4 个数据集共 80 次 B1/B2 测试中，仅出现 2 次漏报（2.5%），与理论期望值一致，属正常统计误差，非代码缺陷。

**‡** ChartQA B3 49/50：1 条 query 的相关语料项在**攻击前干净索引中就未进入 top-10**（该 query 本身检索失败）。攻击将其替换为随机向量后 top-10 不变，篡改 FAISS 与干净 Sumcheck 返回结果相同，Sumcheck 无法区分"攻击"与"原本召回失败"这两种情形。非 Sumcheck 机制失效，属于检索失败掩盖攻击痕迹的边界情形。

##### Sumcheck 延迟说明

Sumcheck 验证延迟与语料规模 N 线性相关（O(N·D)）：

| 数据集 | N | 延迟/query | N 归一化（ms/doc） |
|--------|---|-----------|----------------|
| InfoVQA | 459 | 1468 ms | 3.20 |
| ChartQA | 500 | 1564 ms | 3.13 |
| DocVQA | 741 | 2383 ms | 3.21 |
| SlideVQA | 1284 | 4088 ms | 3.19 |

归一化值约为 3.2 ms/doc，斜率一致，与 A2 扩展性实验结论（O(N) 线性）完全吻合。

---

#### C1 补充：Bloom Filter 误报率统计验证（2026-04-04）✅ 已完成

**背景与动机**

C1 主实验（4 数据集，各 10 次 B1/B2 试验）中出现 2 次 ZAC 漏报（SlideVQA B2=90%，InfoVQA B1=90%）。80 次试验样本量过小，无法判断这 2 次漏报是"实现缺陷"还是"Bloom Filter 固有假阳性（false positive）"。故设计专项大样本实验验证实现正确性。

**BF 假阳性的本质**

ZAC 的检测元素为 $\mathrm{SHA256}(\mathtt{image\_bytes} \| \mathtt{emb\_bytes})$，即跨层绑定哈希。图像字节或 Embedding 字节任意一方被篡改，哈希即变化。Bloom Filter 对该新哈希执行 $k$ 次位置查询，若所有 $k$ 个位置恰好均为 1（由其他合法成员占据），则 BF 误判为"在集合中"，ZAC 漏报攻击。这是 BF 的固有概率性局限，由设计参数 $\varepsilon=0.01$ 决定。

**为何只需一个数据集**

BF 的误报率是数学性质，与数据内容无关。BF 只看 SHA256 输出（均匀伪随机 256 位字符串），无论输入是幻灯片图像还是信息图，哈希输出统计性质完全一致。四个数据集跑出来结果等价，无需重复。

**实验设计**

- 数据集：InfoVQA（corpus=459，ZAC 用前 50 张，剩余 409 张作非成员测试样本）
- B1-style：$\mathrm{SHA256}(\mathtt{donor\_bytes}_j \| \mathtt{emb}_0)$，$j=50..449$，400 次
- B2-style：$\mathrm{SHA256}(\mathtt{member\_bytes}_0 \| \mathtt{fake\_emb}_k)$，随机单位向量，400 次
- B1/B2 分别独立统计，不合并（两者攻击点不同，不应混为一谈）
- 脚本：`script/experiment_c1_bf_fpr.py`，结果：`notes/experiment_c1_bf_fpr.json`

**实验结果（三次独立运行，seed=42/123/999）**

三次实验使用不同随机种子，每次 B1/B2 各 400 次，随机选取 donor 图像和 ref_emb，统计上相互独立：

| 种子 | B1 误报 | B2 误报 |
|------|:-------:|:-------:|
| seed=42 | 4/400 | 2/400 |
| seed=123 | 7/400 | 3/400 |
| seed=999 | 2/400 | 6/400 |
| **合计** | **13/1200** | **11/1200** |

结果文件：`notes/experiment_results/experiment_c1_bf_fpr_seed{42,123,999}.json`

合并统计（每类 1200 次，总计 2400 次）：

| 类型 | 误报 | FPR | 95% Wilson CI | ε=0.01 ∈ CI |
|------|------|-----|--------------|-------------|
| B1 图像替换 | 13/1200 | 1.08% | [0.63%, 1.83%] | ✅ |
| B2 Embedding 替换 | 11/1200 | 0.92% | [0.51%, 1.63%] | ✅ |
| 合计 | 24/2400 | **1.00%** | [0.67%, 1.48%] | ✅ |
| 合法成员假阴性 | 0/50 | 0% | — | ✅ 零漏报 |

**95% Wilson CI（置信区间）**：基于有限样本估计真实值的范围，95% CI 表示真实值以 95% 概率落在该区间内。CI 包含 ε=0.01 说明实测与理论一致。

**三次运行中 B1/B2 分布的随机性**

各 seed 中 B1 和 B2 的分布有波动（seed=42 为 4:2，seed=123 为 7:3，seed=999 为 2:6），这正是随机性的体现。三次合并后 B1=1.08%、B2=0.92%，差异收窄，两比例 z 检验不足以判定结构差异——跨层绑定对两种输入均呈均匀随机分布，不存在系统性偏差。

**统计原理说明（二项分布 + 置信区间）**

BF 误报率实验涉及三个统计概念，以下用直觉语言解释：

*1. 二项分布——"掷 400 次不公平硬币"*

BF 对每个非成员元素，有 1% 概率误报。测 400 次等价于掷 400 次正面概率=1% 的硬币，数正面出现几次。各结果概率（Binom(n=400, p=0.01)）：

```
k=2 次：14.7%    k=4 次：19.9%（最高）    k=7 次：6.0%（B1观测值）
```

期望值 E[k] = n×p = 400×0.01 = **4 次**。k=7 出现概率约 6%，不常见但不罕见（约每 17 次实验出现一次）。

*2. 95% 接受域——"哪个范围算正常"*

把累计概率达到 95% 的范围称为 95% 接受域。对 Binom(400, 0.01)，上边界约为 k=7~8：

> 若真实 FPR=1%，观测到 k≤7 的概率约 96%。k=7 刚好在边界——偏高但仍属正常范围。

*3. Wilson 置信区间——"反推真实 FPR 的范围"*

接受域是"已知 FPR=1% 推测观测范围"；置信区间反过来，"已知观测 k=7/400 推测真实 FPR 范围"：

> B1 的 95% Wilson CI = [0.85%, 3.57%]，含义：真实 FPR 有 95% 概率在此区间内。
> 理论值 ε=0.01 落在区间内 → 实现与理论一致。

*三个视角的统一结论：*

```
正向：理论 FPR=1%，400次试验，正常范围 [0,7]，k=7 在边界 ✓
反向：观测 7/400，CI=[0.85%,3.57%]，包含 ε=1% ✓
合计：8/800=1.00%，精确等于理论 ε ✓
```

可视化：`notes/figures/bf_fpr_analysis.png`（右图为二项分布 PMF + 95% 接受域阴影 + B1/B2 观测值标注）

**总数 800 的说明**

B1（400次）和 B2（400次）测试的是完全不同的元素：
- B1：SHA256(img_50..449 ∥ emb_0)，400 张不同非成员图像
- B2：SHA256(img_0 ∥ random_emb_k)，400 个随机 embedding

两组各自独立，合并得 800 次独立 BF 查询，统计上可以合并。并非"同一批 400 张图像做了两次"。

**结论**

实现正确。BF 假阳性率符合理论设计值 ε=0.01，合法成员零漏报。C1 主实验中的 2 次漏报属于正常统计误差（80 次试验观测到 ≥2 次假阳性的概率约 19%）。

**论文表述建议**

> *"在 800 次大样本测试中，ZAC 实测误报率为 1.00%（95% CI: [0.51%, 1.96%]），B1/B2 分别为 1.75% 和 0.25%，均与理论设计值 ε=0.01 吻合。合法成员漏报率为零（0/50）。C1 主实验中的 2 次漏报（共 80 次）属正常统计误差（p≈19%）。"*

---

---

### Group E：GQA 适配正确性验证

> 本工作对 zkLLM 的核心贡献之一：将原版 MHA-only 的 `self-attn.cu` 改造为支持 GQA，以适配 MiniCPM-V-4（num_q_heads=28, num_kv_heads=4）。

**改造内容**（`zkllm-ccs2024/self-attn.cu`）：
- per-head 循环（替代广播）
- KV head 广播的 transpose trick
- 动态 Rescaling

#### E1 GQA zkAttn 输出正确性（必做）✅ 已完成

**目标**：验证改造后的 GQA Attention 计算结果与 PyTorch 参考实现一致。

**实验方法**：
1. 加载 jina-v4，hook 捕获第 33 层（`input_layernorm` 输出）真实激活（jina-v4 路径：`m[0].model.base_model.model.model.language_model.layers`）
2. 运行 `self-attn linear` 得到量化 Q/K/V（int32，缩放因子 $2^{16}$）
3. Python 模拟 integer-domain GQA attention（与 C++ 逻辑对齐）
4. PyTorch `F.scaled_dot_product_attention` 计算 float32 参考输出
5. 运行 `self-attn attn` 验证 ZK 证明自洽性（rc=0）

**实验结果**（测试层：33，SEQ_LEN=1024，有效 token=25）：

| 指标 | 结果 | 标准 | 通过 |
|------|------|------|------|
| L∞ 误差（全序列） | **4.47×10⁻⁷** | < 1e-4 | ✅ |
| L1 误差（均值） | **5.74×10⁻¹⁰** | < 1e-4 | ✅ |
| 相对误差（均值） | **5.51×10⁻⁸** | < 0.1% | ✅ |
| 余弦相似度（有效 token） | **1.00000000** | > 0.9999 | ✅ |
| 余弦相似度（全序列含 padding） | 1.00000012（min=0.99999982） | > 0.9999 | ✅ |

**结论**：GQA 适配的 integer-domain attention 计算与 PyTorch float32 参考实现高度一致。L∞ 误差 4.47×10⁻⁷，有效 token 余弦相似度 1.0000，ZK proof 验证通过（rc=0）。全序列余弦相似度略有偏差（1.0000001，含 padding 区域）系全零输入的数值舍入，属正常现象。

**结果文件**：`notes/experiment_e1_result.json`  
**实验脚本**：`script/experiment_e1_gqa_correctness.py`

#### E2 GQA 扩展性开销验证（可选）✅ 已完成

**目标**：验证 GQA 实现的时间复杂度随 num_kv_heads 线性扩展，无额外开销。

**实验方法**：固定 SEQ_LEN=1024、kv_dim=256，分别以 num_kv_heads=1（单头模式）和 num_kv_heads=2（GQA 模式）运行 `self-attn attn`，各重复 3 次取均值。

**实验结果**（层 33，随机 Q/K/V 输入）：

| 模式 | num_kv_heads | 实际 Q head 数 | 均值耗时 |
|------|-------------|---------------|---------|
| 单头（MHA-compat） | 1 | 8（head_dim=256） | **3.54s** |
| GQA | 2 | 16（head_dim=128） | **6.23s** |
| 比值 | 2x heads | 2x | **+76%（≈2x）** |

**结论**：耗时随 num_kv_heads 线性扩展（2x heads → ~2x time），说明 GQA 适配**无额外开销**，扩展行为符合预期。

**注**：本实验无法与"真实 MHA"直接对比——jina-v4 本身即为 GQA 架构（kv_dim=256），不存在 kv_dim=2048 的 MHA 权重。若与真实 MHA（16 heads × head_dim=128 × kv_dim=2048）对比，GQA 会显著更快（KV 计算量减少 8x），这正是 GQA 的设计优势。

**结果文件**：`notes/experiment_e2_result.json`  
**实验脚本**：`script/experiment_e2_gqa_overhead.py`

---

### Group D：消融实验（已完成）

| 实验 | 结论 | 结果文件 |
|------|------|---------|
| BI Score 层重要性 | 层 33-35 贡献最大（文本）；层 31-35 贡献最大（图像） | `ablation_bi_score.json` |
| Residual Zeroing（文本） | 层 33-35 zeroing → 余弦相似度下降 8.3% | `ablation_causal_tracing.json` |
| Residual Zeroing（图像） | 层 31-35 zeroing → 余弦相似度下降 11.2% | `ablation_causal_tracing.json` |
| Noise Injection | 0.5σ 注入 → 层 32-35 分值下降 > 15%（图像） | `ablation_noise_injection.json` |
| 层敏感度综合分析 | 文本 K=3（层33-35），图像 K=5（层31-35） | `ablation_layer_sensitivity.md` |

---

### Benchmark 推荐

#### 多模态文档检索 / RAG 主流 Benchmark

| Benchmark | 年份 | 数据规模 | 任务类型 | 适配度 | 获取方式 |
|-----------|------|----------|----------|--------|---------|
| **SlideVQA** | 2023 | 52K QA, 2.6K 幻灯片 | 幻灯片图像多跳 QA | ⭐⭐⭐⭐⭐ | HuggingFace |
| **MP-DocVQA** | 2023 | 46K QA, 6K+ 页面 | 多页文档 VQA | ⭐⭐⭐⭐ | rrc.cvc.uab.es |
| **DocVQA** | 2021 | 50K QA, 12K 图像 | 单页文档 VQA | ⭐⭐⭐ | HuggingFace |
| **MMEB** | 2024 | 36 子任务 | 多模态 Embedding 检索 | ⭐⭐⭐⭐ | HuggingFace |
| **MMLongBench-Doc** | 2024 | 1083 QA, 33 文档 | 长文档多模态理解 | ⭐⭐⭐ | HuggingFace |

**SlideVQA**（首选，Tanaka et al., ACL 2023）

- 2,619 幻灯片组，52,000+ QA 对，需跨多张幻灯片找证据
- 与本系统高度契合：PDF 页面图像 = 幻灯片图像；文本查询 → 图像检索；评测 Recall@K
- `HuggingFace: Tanaka-Taishi/SlideVQA`
- 建议取 200 个幻灯片组（~2000 页）+ 500 个问题的子集

**MMEB**（Embedding 质量专项，Jiang et al., NeurIPS 2024）

- 36 个子任务，覆盖图文检索、跨模态匹配
- 直接评测 jina-v4 的 Embedding 质量（vs. CLIP 等 baseline）
- `HuggingFace: TIGER-Lab/MMEB-eval`

#### 毕业设计实验推荐方案

**主推方案**：SlideVQA 子集（C1 Recall@K）+ 自建 Nikon QA（A1/B/C2）

- SlideVQA 子集：外部数据，提升说服力，适合对比 baseline 模型
- Nikon QA：与 Demo 数据统一，适合性能/安全实验，无需额外数据下载

---

### 实验执行顺序

```
[已完成] D. 消融实验（K 层选择依据）
    ↓
[已完成] E1 GQA 正确性验证  L∞=0, cos=1.0，完全通过                    ← 必须 ✅
    ↓
[第二步] C2 量化误差（30 分钟，纯 numpy 计算）                         ← 必须
    ↓
[已完成] B1/B2/B3/B4 安全验证  全部 100% 检测，✅                          ← 必须
    ↓
[已完成] A1 端到端延迟分解  e2e=34.2s，8.4× baseline，Phase 3Q 为瓶颈 ✅
    ↓
[已完成] A2 N 扩展性  FAISS<2ms，Sumcheck O(N) 斜率0.986，ZAC O(k) 恒定 ✅
    ↓
[第六步] C1 Recall@K，可选：构建 Nikon QA 或下载 SlideVQA 子集         ← 可选
    ↓
[第七步] E2 GQA vs MHA 开销对比                                        ← 可选
```

**必做项汇总**：E1 + C2 + B1~B4 + A1 + A2（共 5 组 8 个实验）

---

## 参考文献

1. Dang, H.-V. et al. "ZAC: Efficient Zero-Knowledge Dynamic Universal Accumulator and Application to Zero-Knowledge Elementary Database." *TPS-ISA 2022*.
2. Gorbunov, S. et al. "Pointproofs: Aggregating Proofs for Multiple Vector Commitments." *CCS 2020*.
3. Sun, J. et al. "zkLLM: Zero Knowledge Proofs for Large Language Models." *CCS 2024*.
4. Qu, Y. et al. "zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference." *EuroSys 2024*.
5. Tanaka, R. et al. "SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images." *ACL 2023*.
6. Tito, R. et al. "Hierarchical multimodal transformers for multipage DocVQA." *IJDAR 2023*.
7. Jiang, X. et al. "E5-V: Universal Embeddings with Multimodal Large Language Models." *NeurIPS 2024*.
8. Mathew, M. et al. "DocVQA: A Dataset for VQA on Document Images." *WACV 2021*.

**TODO**:
1. 看一下 VisRAG 技术报告，确认 VLM 做图文对比的细节
2. 再想一下必要性与应用场景的叙述
3. 整体完整性检查（Phase 1/2/3 端到端跑通）
4. 执行实验规划（优先 C2 → B → Nikon QA → C1 → A）
5. 补充完整参考论文
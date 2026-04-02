# 可验证多模态检索系统

基于 VisRAG 架构的端到端可验证检索增强生成系统，覆盖从语料库来源到推理结果的完整密码学证明链。

---

## 系统设计

### 核心问题

传统 RAG 系统的可信性依赖对服务器的信任：检索到的文档是否来自声称的语料库？相似度排名是否被篡改？Embedding 是否由声称的模型计算？本系统对上述问题逐一提供密码学证明。

### 三层验证架构

```
用户查询
  │
  ▼
[Phase 3 在线] zkLLM 推理证明
  jina-v4 后 K=3 层（层33-35）Sumcheck 证明
  FFN (SwiGLU) + Self-Attn linear + GQA zkAttn
  证明：query embedding 确实由 jina-v4 计算
  │
  ▼
[Phase 2] Sumcheck 内积证明
  Global Batch 模式：对全部 N 个语料向量做随机线性组合
  验证者独立从经证明的 N 个分值中选 top-k
  证明：检索分值正确，top-k 选择未被篡改
  │
  ▼
[Phase 1] ZAC 成员证明
  Bloom Filter + Pointproofs (BLS12-381)
  承诺元素：SHA256(image_bytes ∥ embedding_bytes)
  O(1) 批量成员证明（聚合 G1 点）
  证明：检索到的图像属于承诺语料库，且 embedding 与图像绑定
  │
  ▼
[Phase 3 离线] 语料库侧 zkLLM 证明（预计算）
  jina-v4 后 K=5 层（层31-35）
  证明：每张图像的 corpus embedding 由 jina-v4 正确计算
```

### 技术选型说明

| 组件 | 方案 | 原因 |
|------|------|------|
| 成员证明 | ZAC (Pointproofs on BLS12-381) | O(1) 批量证明，优于 Merkle 的 O(k·log N) |
| 检索正确性 | Global Batch Sumcheck | 一次证明覆盖全部 N 个内积，verifier 独立选 top-k |
| 推理证明 | zkLLM (Sun et al., CCS'24) | 支持 SwiGLU FFN + GQA Attention 的 Sumcheck |
| 检索模型 | jina-embeddings-v4 | 原生多模态，同时处理文本 query 和图像 corpus |
| 生成模型 | MiniCPM-V-4 | 轻量多模态 VLM，适合端侧部署 |
| 向量索引 | FAISS IndexFlatIP | 精确搜索（非 ANN），与 Sumcheck 证明的 top-k 可比对 |

### 差异化 K 层设计

基于消融实验（BI Score + Residual Zeroing），文本和图像模态的最优证明层数不同：

- **文本 query**：K=3，证明层 33-35（层 30-32 对文本贡献边际）
- **图像 corpus**：K=5，证明层 31-35（图像在更深层有更强的局部特征依赖）

---

## 目录结构

```
UltraRAG/
├── script/
│   ├── interactive_demo.py          # 主 Demo（Gradio UI + 完整验证流水线）
│   ├── build_verifiable_corpus.py   # 一键建库（PDF → 语料 → embedding → 索引 → ZAC → zkLLM）
│   ├── build_corpus_zkllm_proofs.py # 语料库侧 zkLLM 证明预计算
│   ├── phase1_corpus_fingerprint.py # ZAC 语料库指纹（单独运行）
│   ├── phase2_sumcheck.py           # Sumcheck 协议测试与实验
│   └── ablation_layer_sensitivity.py# 层敏感度消融实验
├── src/
│   ├── zac/accumulator.py           # ZAC 实现（BloomFilter + Pointproofs）
│   ├── sumcheck/inner_product.py    # Sumcheck 协议（Global Batch 模式）
│   └── zkllm/
│       ├── *.cu / *.cuh             # zkLLM CUDA 源码（含 GQA 适配后的 self-attn.cu）
│       ├── Makefile                 # 编译入口（nvcc, sm_89）
│       ├── llama-*.py               # 权重量化 / 公共参数生成脚本
│       ├── fileio_utils.py          # 二进制 IO 工具
│       ├── load_jina_weights.py     # jina-v4 权重提取
│       ├── swiglu-table.bin         # SwiGLU 查找表（FFN 证明所需，预计算，16MB）
│       └── bin/                     # 编译产物（gitignore，需本地 make 生成）
├── corpora/image.jsonl              # 图像语料库索引
├── embedding/embedding.npy          # 语料库 embedding（float32, N×2048）
├── index/index.index                # FAISS IndexFlatIP
├── output/phase1/
│   ├── prover_state.json            # ZAC prover state（BF + Pointproofs CRS）
│   └── fingerprint.json             # 语料库指纹摘要
├── zkllm-workdir/jina-v4/
│   ├── corpus_proof_*.json          # 每张图像的 zkLLM 预计算证明
│   └── *.bin                        # jina-v4 权重承诺文件
├── data/nikon.pdf                   # 示例 PDF（尼康相机手册）
├── docs/                            # 参考论文（ZAC, zkGPT, zkLLM）
├── notes/implementation_log.md      # 完整实现记录与实验分析
└── _archive/                        # 归档（旧脚本、原始 UltraRAG 文档等）
```

---

## 环境依赖

**模型路径**（默认）：
- 检索模型：`/root/autodl-tmp/models/jina-embeddings-v4`
- 生成模型：`/root/autodl-tmp/models/MiniCPM-V-4`

**Python 环境**：
```bash
conda activate ultrarag
```

主要依赖：`torch`、`faiss-gpu`、`sentence-transformers`、`pymupdf`、`py_ecc`、`mmh3`、`gradio`、`Pillow`

---

## zkLLM 编译

zkLLM 的 CUDA 源码位于 `src/zkllm/`，编译后的二进制不包含在 git 中，**首次使用前需本地编译**。

### 前置要求

- CUDA Toolkit ≥ 12.0，`nvcc` 位于 `/usr/local/cuda/bin/nvcc`
- GPU 计算能力 ≥ sm_89（RTX 4090 / A100 等，如有需要修改 Makefile 中 `ARCH`）
- C++17 支持

### 编译步骤

```bash
cd /root/autodl-tmp/UltraRAG/src/zkllm

# 编译全部目标（ppgen, commit-param, self-attn, ffn, rmsnorm, skip-connection）
make all -j$(nproc)

# 将编译产物复制到 bin/
mkdir -p bin
cp ppgen commit-param self-attn ffn rmsnorm skip-connection bin/
```

### GQA 适配说明

原版 zkLLM (`self-attn.cu`) 仅支持标准 MHA。本项目将其扩展以支持 **Grouped Query Attention (GQA)**，用于适配 jina-embeddings-v4（num_q_heads=16, num_kv_heads=2）和 MiniCPM-V-4（num_q_heads=28, num_kv_heads=4）。

关键改动：
- `argv[8]` = `kv_dim`：允许 KV 投影维度不等于 embed_dim
- `argv[9]` = `num_kv_heads`：指定 KV head 数量（默认 1，与原版 MHA 兼容）
- Transpose trick：无需新增 CUDA kernel，通过 `trunc()` 切分各 head
- 动态 Rescaling：根据 `head_out_size = seq_len × head_dim` 自动计算缩放因子

### 生成公共参数（ppgen）

权重承诺所需的 SRS（结构化参考字符串）需提前生成：

```bash
# 在 zkllm-workdir/ 下生成 jina-v4 的公共参数
cd /root/autodl-tmp/UltraRAG
./src/zkllm/bin/ppgen zkllm-workdir/jina-v4 <embed_dim>
```

> **注**：`swiglu-table.bin`（16MB，SwiGLU FFN 查找表）已包含在仓库中，无需重新生成。

---

## 从零建库

### 一键建库（推荐）

从 PDF 出发，自动完成所有阶段：

```bash
cd /root/autodl-tmp/UltraRAG
python script/build_verifiable_corpus.py --pdf data/nikon.pdf
```

流程：PDF 切页图像 → jina-v4 embedding → FAISS 索引 → ZAC 指纹 → zkLLM 语料库证明预计算

完整参数：
```bash
python script/build_verifiable_corpus.py \
    --pdf           data/nikon.pdf \
    --corpus        corpora/image.jsonl \
    --embedding     embedding/embedding.npy \
    --index         index/index.index \
    --zac-output    output/phase1/fingerprint.json \
    --zkllm-workdir zkllm-workdir/jina-v4 \
    --k-layers      5
```

跳过某阶段（已有数据时）：
```bash
# 已有语料和 embedding，只重建索引和 ZAC
python script/build_verifiable_corpus.py --pdf data/nikon.pdf \
    --skip-corpus --skip-embed

# 跳过 zkLLM 预计算（耗时较长，可后续单独运行）
python script/build_verifiable_corpus.py --pdf data/nikon.pdf --skip-zkllm
```

### 追加新 PDF（增量模式）

```bash
python script/build_verifiable_corpus.py --pdf data/new_doc.pdf --incremental
```

增量模式仅处理新 PDF，自动重建 ZAC 承诺（包含所有历史图像）。

---

## 分阶段单独运行

### Phase 1：ZAC 语料库指纹

```bash
# 从已有 corpus + embedding 重建 ZAC prover_state
python script/phase1_corpus_fingerprint.py \
    --zac-only \
    --corpus-jsonl  corpora/image.jsonl \
    --embedding-npy embedding/embedding.npy \
    --output        output/phase1/fingerprint.json
```

### Phase 3：zkLLM 语料库证明预计算

```bash
# 单卡全量
python script/build_corpus_zkllm_proofs.py \
    --corpus  corpora/image.jsonl \
    --workdir zkllm-workdir/jina-v4 \
    --k_layers 5

# 双卡并行（每张图像内部 FFN∥Attn）
python script/build_corpus_zkllm_proofs.py \
    --corpus  corpora/image.jsonl \
    --workdir zkllm-workdir/jina-v4 \
    --k_layers 5 \
    --dual-gpu

# 多卡分片（两张卡各处理一半语料，后台并行）
python script/build_corpus_zkllm_proofs.py \
    --num-workers 2 --worker-id 0 &
python script/build_corpus_zkllm_proofs.py \
    --num-workers 2 --worker-id 1 &
```

---

## 启动 Demo

```bash
cd /root/autodl-tmp/UltraRAG
conda activate ultrarag
python script/interactive_demo.py
```

浏览器访问 `http://127.0.0.1:7860`

Demo 提供两个功能：
1. **查询**：输入自然语言问题，展示检索结果及完整验证状态（ZAC / Sumcheck / zkLLM corpus / zkLLM query）
2. **新建/追加知识库**：上传 PDF，自动完成全流程建库

每次查询的验证流水线（终端日志可观察）：
```
[Query]   新查询 proof_id=xxxxxxxx
[Step 1]  向量编码 + hook 捕获（layers 33-35）
[Step 2]  FAISS 检索 top-5
[Step 3]  读取语料库侧 zkLLM 预计算证明
[Step 4]  Sumcheck 内积证明（N 个向量，11 轮）
[Step 5]  ZAC 成员证明（image+embedding hash）
[zkLLM-Q] 后台双卡证明 query 推理（layers 33-35）
[Step 6]  汇总：ZAC ✅ / Sumcheck ✅ / zkLLM corpus ✅ / zkLLM query ✅
[Step 7]  MiniCPM-V-4 生成回答
```

---

## 消融实验

层敏感度分析（BI Score / Residual Zeroing / Noise Injection）：

```bash
# BI Score（基础层重要性）
python script/ablation_layer_sensitivity.py --mode bi

# Residual Zeroing（文本+图像）
python script/ablation_layer_sensitivity.py --mode zero

# Noise Injection
python script/ablation_layer_sensitivity.py --mode noise --noise-scale 0.5

# 全部实验
python script/ablation_layer_sensitivity.py --mode all
```

结果保存至 `notes/ablation_*.json` 和 `notes/ablation_*.png`。

---

## 密码学说明

### ZAC（Phase 1）

- 论文：Dang et al., TPS-ISA 2022
- 构造：Bloom Filter → 二进制向量 v → Pointproofs.Commit(v) → 48 字节 G1 点（承诺根）
- 承诺元素：`SHA256(image_bytes ∥ embedding_bytes)`（跨层绑定，防止 embedding 替换攻击）
- 批量成员证明：O(1) 大小，单次 pairing 验证

### Sumcheck（Phase 2）

- 协议：Thaler (2022) Chapter 4，Fiat-Shamir 非交互化
- 域：$\mathbb{Z}_p$，$p = 2^{61} - 1$（Mersenne 素数）
- 11 轮（$\ell = \lceil \log_2 2048 \rceil$），证明大小 264 字节
- Global Batch 模式：Schwartz-Zippel 随机线性组合，单证明覆盖全部 N 个内积
- 局限：verifier 需持有 embedding.npy 副本（未引入 KZG，见 `notes/implementation_log.md`）

### zkLLM（Phase 3）

- 论文：Sun et al., CCS 2024
- 扩展：GQA zkAttn（16 Q 头 / 2 KV 头），SEQ_LEN=1024（满足 NTT 约束 seq²=2²⁰）
- 覆盖：FFN (SwiGLU) + Self-Attn linear (Q/K/V 投影) + zkAttn (Softmax)
- 文本 K=3（层 33-35）；图像 K=5（层 31-35）

---

## 参考文献

- Dang et al., *ZAC: Efficient Zero-Knowledge Dynamic Universal Accumulator*, TPS-ISA 2022
- Sun et al., *zkLLM: Zero Knowledge Proofs for Large Language Models*, CCS 2024
- Qu et al., *zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference*, EuroSys 2024
- Thaler, *Proofs, Arguments, and Zero-Knowledge*, 2022

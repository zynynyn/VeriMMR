# VeriMMR：面向多模态语义检索的可验证框架

> 在不信任服务提供方的前提下，对"编码—嵌入—匹配—排序"全链路施加密码学约束，使任意第三方可独立验证多模态语义数据检索结果的完整性和可靠性。

---

## 为什么需要可验证检索？

多模态 RAG 系统正在被部署于医疗影像辅助诊断、法律文书检索、金融合规分析等高风险场景，但现有系统的可信性完全依赖对服务提供方的盲目信任。一旦信任假设被打破，攻击者可以在不被察觉的情况下操控检索结果：

| 风险 | 攻击行为 |
|------|---------|
| **B1 语料来源造假** | 向量数据库中的文档被替换为伪造内容 |
| **B2 语料嵌入绑定失败** | FAISS 中的嵌入向量被篡改以操控召回 |
| **B3 检索过程操控** | 相似度分值或 top-k 排序被人为调整 |
| **B4 编码推理欺骗** | 算力方以低参数模型替换约定编码器执行特征提取 |

签名、可信执行环境（TEE）、同态加密等现有方案在计算开销、部署门槛或可验证粒度上均难以同时满足多模态检索场景的要求。

---

## 系统概述

VeriMMR 将可验证性分解为三个独立但衔接的验证阶段，每个阶段针对一类攻击面：

```
图像语料库
    │
    │  语义成员资格验证  →  证明"返回文档确实来自承诺的语料库快照，嵌入向量与图像绑定"
    │  相似度排序验证    →  证明"相似度计算和 top-k 排序未被篡改"
    │  编码器推理验证    →  证明"嵌入由约定的多模态编码器对原始图像真实计算所得"
    ▼
用户获得可独立验证的检索结果
```

| 验证阶段 | 防御攻击 |
|---------|---------|
| 语义成员资格验证 | B1 语料来源造假、B2 语料嵌入绑定失败 |
| 相似度排序验证 | B3 检索过程操控 |
| 编码器推理验证 | B4 编码推理欺骗 |

---

## 快速开始

### 环境配置

```bash
conda env create -f environment_no_builds.yml
conda activate ultrarag
```

主要依赖：`torch 2.8.0+cu128`、`faiss-gpu`、`sentence-transformers`、`pymupdf`、`py_ecc`、`mmh3`、`gradio`

**模型路径**（默认）：
- 检索模型：`/path/to/models/jina-embeddings-v4`
- 生成模型：`/path/to/models/MiniCPM-V-4`

### 编译 zkLLM CUDA 二进制

```bash
cd src/zkllm
make all -j$(nproc)
mkdir -p bin && cp ppgen commit-param self-attn ffn rmsnorm skip-connection patch-merger conv3d-embed verify-ipa bin/
```

要求：CUDA ≥ 12.0，sm_89（RTX 4090）。其他 GPU 修改 Makefile 中 `ARCH`。

### 初始化权重承诺（首次部署，一次性离线操作）

```bash
python script/setup_all_visual_params.py \
    --model-path /path/to/models/jina-embeddings-v4 \
    --workdir zkllm-workdir/jina-v4
```

从 jina-v4 提取视觉组件（ViT、PatchMerger、Conv3d、RMSNorm）权重并生成密码学承诺，产物存入 `zkllm-workdir/`，之后无需重复运行。

### 建立可验证语料库

```bash
python script/build_verifiable_corpus.py --pdf data/nikon.pdf
```

自动完成：PDF 切页 → jina-v4 编码 → FAISS 索引 → ZAC 语料指纹 → zkLLM 预计算证明

示例 PDF 可从 [此处](https://download.nikonimglib.com/archive4/ywJ4K00fa2Lr05vv5OS00pV5Hg36/Z7Z6UM_TH(Sc)07.pdf) 下载。

### 启动 Demo

```bash
python script/interactive_demo.py
# 浏览器访问 http://127.0.0.1:7860
```

---

## 目录结构

```
UltraRAG/
├── script/
│   ├── interactive_demo.py              # 主 Demo（Gradio UI + 完整验证流水线）
│   ├── build_verifiable_corpus.py       # 一键建库
│   ├── build_corpus_zkllm_proofs.py     # 语料库侧 zkLLM 证明预计算
│   ├── phase1_corpus_fingerprint.py     # ZAC 语料库指纹（单独运行）
│   ├── phase2_sumcheck.py               # Sumcheck 协议演示与测试
│   ├── setup_all_visual_params.py       # 一键初始化所有视觉组件承诺
│   ├── setup_vit_params.py              # ViT 32 块权重承诺
│   ├── setup_patchmerger_params.py      # PatchMerger 权重承诺
│   ├── setup_conv3d_params.py           # Conv3d 嵌入权重承诺
│   ├── setup_rmsnorm_params.py          # RMSNorm 权重承诺
│   ├── setup_embedding_commitments.py   # 语料嵌入向量 IPA 承诺
│   ├── verify_vit.py                    # ViT 验证
│   ├── verify_layers.py                 # LLM 各层验证
│   └── verify_zkllm_full.py             # 全量 zkLLM 验证
├── src/
│   ├── zac/accumulator.py               # ZAC（BloomFilter + Pointproofs）
│   ├── sumcheck/inner_product.py        # Sumcheck（Global Batch 模式）
│   └── zkllm/
│       ├── self-attn.cu                 # zkAttn（含 GQA 适配）
│       ├── ffn.cu / rmsnorm.cu          # FFN / RMSNorm 证明
│       ├── patch-merger.cu / conv3d-embed.cu / verify-ipa.cu
│       ├── Makefile                     # nvcc sm_89
│       ├── load_jina_weights.py         # jina-v4 权重提取
│       └── bin/                         # 编译产物（需本地 make）
├── servers/                             # UltraRAG MCP 服务模块
├── corpora/image.jsonl                  # 图像语料库索引
├── embedding/embedding.npy              # 语料库 embedding（float32, N×2048）
├── index/index.index                    # FAISS IndexFlatIP
├── output/phase1/
│   ├── prover_state.json                # ZAC prover state
│   └── fingerprint.json                 # 语料库指纹（含 ZAC Root）
└── zkllm-workdir/jina-v4/
    ├── corpus_proof_*.json              # 每张图像的 zkLLM 预计算证明
    └── *.bin                            # jina-v4 权重承诺文件
```

---

## 技术选型

| 阶段 | 核心技术 | 作用 |
|------|---------|------|
| 语义成员资格验证 | ZAC（Bloom Filter + Pointproofs，BLS12-381） | O(1) 聚合成员证明，图像与嵌入向量跨层绑定 |
| 相似度排序验证 | Global Batch Sumcheck（Fiat-Shamir，Mersenne 域） | 全量内积可验证，Verifier 独立选 top-k，防排名伪造 |
| 编码器推理验证 | zkLLM（Sumcheck + IPA，含 GQA 适配） | 权重承诺 + 逐层推理证明，覆盖 ViT 到 LM 全链路 |

---

## 信任模型

服务提供方（Prover）持有完整模型权重和语料库，负责运行推理并随检索结果返回证明。用户或审计方（Verifier）只需持有公开承诺值：ZAC Root（48 字节，建库后一次性发布）和模型各层权重承诺（部署时发布一次）。Verifier 不需要模型，也不需要原始语料库，可独立验证任意检索请求的三阶段证明。

---

## 应用场景

**高风险决策支持**：医疗影像辅助诊断、法律文书检索、金融合规分析等场景监管要求日趋严格。三阶段验证保证检索结果可追溯、不可伪造，使 AI 辅助决策的全过程具备可审计性。

**多方联合知识库**：各参与方互不信任时，将 ZAC Root 存证于区块链，任意一方可独立验证其他方返回的检索结果来自约定的共同语料，Sumcheck 防止单方操控排名——在不共享原始数据的前提下实现可信联合检索。

**AI 服务可审计性**：服务提供方可在不公开模型权重的前提下，向用户或监管机构提供推理可验证性证明，补充 C2PA 等内容溯源标准未覆盖的"检索过程"环节。

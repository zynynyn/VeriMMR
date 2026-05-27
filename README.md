<div align="center">

# VeriMMR

**[中文](#中文) | [English](#english)**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0%2Bcu128-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Third Party](https://img.shields.io/badge/Includes-Apache%202.0%20%7C%20MIT-lightgrey)

</div>

---

<a id="中文"></a>

# VeriMMR：面向多模态语义数据的零知识可验证检索机制设计与实现

> 面向依赖多模态大模型进行语义检索的服务外包场景，使弱客户端无需信任算力方、无需重算或访问模型权重，即可独立验证从语义编码、相似度计算到结果排序的全链路诚实性。

## ❓ 为什么需要可验证检索？

多模态 RAG 系统正在被部署于医疗影像辅助诊断、法律文书检索、金融合规分析等高风险场景，但现有系统的可信性完全依赖对服务提供方的盲目信任。一旦信任假设被打破，攻击者可以在不被察觉的情况下操控检索结果：

| 风险 | 攻击行为 |
|------|---------|
| **B1 语料来源造假** | 向量数据库中的文档被替换为伪造内容 |
| **B2 语料嵌入绑定失败** | FAISS 中的嵌入向量被篡改以操控召回 |
| **B3 检索过程操控** | 相似度分值或 top-k 排序被人为调整 |
| **B4 编码推理欺骗** | 算力方以低参数模型替换约定编码器执行特征提取 |

签名、可信执行环境（TEE）、同态加密等现有方案在计算开销、部署门槛或可验证粒度上均难以同时满足多模态检索场景的要求。

## 🏗️ 系统概述

VeriMMR 将可验证性分解为三个独立但衔接的验证环节，每个环节针对一类攻击面：

```
图像语料库
    │
    │  语义成员资格验证  →  证明"返回文档确实来自承诺的语料库快照，嵌入向量与图像绑定"
    │  相似度排序验证    →  证明"相似度计算和 top-k 排序未被篡改"
    │  编码器推理验证    →  证明"嵌入由约定的多模态编码器对原始图像真实计算所得"
    ▼
用户获得可独立验证的检索结果
```

| 验证环节 | 防御攻击 |
|---------|---------|
| 语义成员资格验证 | B1 语料来源造假、B2 语料嵌入绑定失败 |
| 相似度排序验证 | B3 检索过程操控 |
| 编码器推理验证 | B4 编码推理欺骗 |

## 🔬 核心创新

**全链路统一可验证机制**：现有方案通常只覆盖存储层（签名/Merkle）或计算层（TEE），无法跨越两层的信任边界。VeriMMR 提出三环节联合架构，首次将密码学可验证性从原始模态输入延伸至最终检索排名，使持有少量公开承诺（无需模型权重或语料库明文）的弱客户端可独立复现验证，填补"结果正确但过程欺骗"的信任盲区。

**级联 ZAC 成员证明**：以复合指纹 `SHA256(image_bytes ∥ embedding_bytes)` 同时绑定文档与嵌入向量，用级联 BloomFilter（n=2）与 Pointproofs 向量承诺组合实现 O(1) 通信的聚合成员证明（96 字节，与语料库规模 N 无关），假阳性率从单层 ε 降至 ε²（实测 1% → 0%），在不暴露语料库明文或索引结构的前提下同时防御文档替换（B1）与嵌入替换（B2）两类攻击。

**批量内积聚合与密码学排序绑定**：将 N 条独立内积验证通过 Schwartz-Zippel 随机线性组合折叠为单条聚合 Sumcheck 证明，并以 Fiat-Shamir 变换消除多轮交互，无需可信设置；以 IPA 向量承诺取代原始浮点嵌入矩阵，将 Verifier 的信任基础从数值正确性提升为密码学绑定，不泄露查询或文档向量内容。

**多模态嵌入推理链端到端可验证化**：将 zkLLM 从单模态语言模型扩展为覆盖五个环节（Conv3d → ViT → PatchMerger → LM → Pooling）的多模态推理证明系统，核心技术包括 GQA 注意力密码学适配、窗口/全局注意力 NTT 精确切分、跨组件 Sumcheck 批量化（gate/up 投影与 q/k/v 投影分别归约为单次验证），在不公开模型参数的前提下向验证者证明嵌入向量由约定编码器真实计算所得。

---

## 🚀 快速开始

### 环境配置

```bash
conda env create -f environment_no_builds.yml
conda activate ultrarag
```

主要依赖：`torch 2.9.0+cu128`、`faiss-gpu`、`sentence-transformers`、`pymupdf`、`pyblst`、`mmh3`、`gradio`

**模型路径**（默认）：
- 检索模型：`/path/to/models/jina-embeddings-v4`
- 生成模型：`/path/to/models/MiniCPM-V-4`

### 编译 zkLLM CUDA 二进制

```bash
cd src/zkllm
make all -j$(nproc)
mkdir -p bin && cp ppgen commit-param self-attn ffn rmsnorm skip-connection patch-merger conv3d-embed verify-ipa open-ipa bin/
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

示例 PDF 可从[此处](https://download.nikonimglib.com/archive4/ywJ4K00fa2Lr05vv5OS00pV5Hg36/Z7Z6UM_TH(Sc)07.pdf)下载。

### 启动 Demo

```bash
python script/interactive_demo.py
# 浏览器访问 http://127.0.0.1:7860
```

> Demo 默认启用 IPA 向量承诺模式（在 `build_verifiable_corpus.py` 建库过程中已一次性完成嵌入承诺初始化）；若承诺产物缺失，则自动回落到 Mersenne 域 Sumcheck 路径，不影响功能演示。

## 📁 目录结构

```
UltraRAG/
├── script/
│   ├── interactive_demo.py              # 主 Demo（Gradio UI + 完整验证流水线）
│   ├── build_verifiable_corpus.py       # 一键建库
│   ├── build_corpus_zkllm_proofs.py     # 语料库侧 zkLLM 证明预计算
│   ├── phase1_corpus_fingerprint.py     # ZAC 语料库指纹（单独运行）
│   ├── phase2_sumcheck.py               # Sumcheck 协议演示与测试
│   ├── setup_all_visual_params.py       # 一键初始化所有视觉组件承诺
│   ├── setup_{vit,patchmerger,conv3d,rmsnorm}_params.py  # 各组件承诺
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
├── servers/                             # UltraRAG 检索/生成服务模块（直接 import 使用）
├── corpora/image.jsonl                  # 图像语料库索引
├── output/phase1/prover_state.json      # ZAC prover state + 语料指纹
└── zkllm-workdir/jina-v4/
    ├── corpus_proof_*.json              # 每张图像的 zkLLM 预计算证明
    └── *.bin                            # jina-v4 权重承诺文件
```

## ⚙️ 技术选型

| 环节 | 核心技术 | 作用 |
|------|---------|------|
| 语义成员资格验证 | ZAC（Bloom Filter + Pointproofs，BLS12-381） | O(1) 聚合成员证明，图像与嵌入向量跨层绑定 |
| 相似度排序验证 | Global Batch Sumcheck（Fiat-Shamir，Mersenne 域） | 全量内积可验证，Verifier 独立选 top-k，防排名伪造 |
| 编码器推理验证 | zkLLM（Sumcheck + IPA，含 GQA 适配） | 权重承诺 + 逐层推理证明，覆盖 ViT 到 LM 全链路 |

## 🔒 信任模型

服务提供方（Prover）持有完整模型权重和语料库，负责运行推理并随检索结果返回证明。用户或审计方（Verifier）只需持有公开承诺值：ZAC Root（96 字节，建库后一次性发布）和模型各层权重承诺（部署时发布一次）。Verifier 不需要模型，也不需要原始语料库，可独立验证任意检索请求的三环节证明。

## 💡 应用场景

**高风险决策支持**：医疗影像辅助诊断、法律文书检索、金融合规分析等场景监管要求日趋严格。三环节验证保证检索结果可追溯、不可伪造，使 AI 辅助决策的全过程具备可审计性。

**多方联合知识库**：各参与方互不信任时，将 ZAC Root 存证于区块链，任意一方可独立验证其他方返回的检索结果来自约定的共同语料，Sumcheck 防止单方操控排名——在不共享原始数据的前提下实现可信联合检索。

**AI 服务可审计性**：服务提供方可在不公开模型权重的前提下，向用户或监管机构提供推理可验证性证明，补充 C2PA 等内容溯源标准未覆盖的"检索过程"环节。

---

<a id="english"></a>

# VeriMMR: A Zero-Knowledge Verifiable Retrieval Mechanism for Multimodal Semantic Data

> Designed for outsourced services that rely on multimodal large models for semantic retrieval, enabling weak clients to independently verify the end-to-end honesty of semantic encoding, similarity computation, and result ranking — without trusting the compute provider, re-running computation, or accessing model weights.

## ❓ Why Verifiable Retrieval?

Multimodal RAG systems are increasingly deployed in high-stakes domains such as medical imaging assistance, legal document retrieval, and financial compliance analysis — yet their trustworthiness relies entirely on blind trust in the service provider. Once this assumption is violated, an attacker can silently manipulate retrieval results:

| Risk | Attack Behavior |
|------|----------------|
| **B1 Corpus Source Forgery** | Documents in the vector database are replaced with fabricated content |
| **B2 Corpus Embedding Binding Failure** | Embedding vectors in FAISS are tampered to manipulate recall |
| **B3 Retrieval Process Manipulation** | Similarity scores or top-k rankings are artificially adjusted |
| **B4 Encoder Inference Deception** | The compute provider substitutes a weaker model for the agreed encoder |

Existing defenses — signatures, TEE, homomorphic encryption — cannot simultaneously satisfy the overhead, deployment, and verifiability granularity requirements of a multimodal retrieval pipeline.

## 🏗️ System Overview

VeriMMR decomposes verifiability into three independent but connected stages, each targeting a distinct attack surface:

```
Image Corpus
    │
    │  Semantic Membership Verification  →  Proves "returned documents come from the committed corpus, embeddings bound to images"
    │  Similarity Ranking Verification   →  Proves "similarity scores and top-k ranking have not been tampered"
    │  Encoder Inference Verification    →  Proves "embeddings were computed by the agreed encoder from the original images"
    ▼
User receives independently verifiable retrieval results
```

| Verification Stage | Defended Attacks |
|-------------------|-----------------|
| Semantic Membership Verification | B1 Corpus Source Forgery, B2 Embedding Binding Failure |
| Similarity Ranking Verification | B3 Retrieval Process Manipulation |
| Encoder Inference Verification | B4 Encoder Inference Deception |

## 🔬 Key Innovations

**Full-Pipeline Unified Verifiable Mechanism**: Existing approaches typically cover either the storage layer (signatures/Merkle trees) or the compute layer (TEE), without bridging both trust boundaries. VeriMMR introduces a three-stage joint architecture that, for the first time, extends cryptographic verifiability from raw multimodal input to the final retrieval ranking. A lightweight Verifier holding only compact public commitments — no model weights, no raw corpus — can independently re-run verification, closing the trust gap of "correct results, deceptive process."

**Cascade ZAC Membership Proof**: Documents and their embedding vectors are jointly bound via the composite fingerprint `SHA256(image_bytes ∥ embedding_bytes)`. Combining a cascaded BloomFilter (n=2) with Pointproofs vector commitments yields O(1)-communication aggregated membership proofs of constant size (96 bytes, independent of corpus size N), reducing the false positive rate from ε (single layer) to ε² (measured: 1% → 0%). This simultaneously defends against document substitution (B1) and embedding substitution (B2) without exposing corpus content or index structure.

**Batch Inner Product Aggregation and Cryptographic Ranking Binding**: N independent inner product verifications are folded into a single aggregated Sumcheck proof via Schwartz-Zippel random linear combination, with Fiat-Shamir eliminating multi-round interaction — no trusted setup required. Replacing the raw floating-point embedding matrix with IPA vector commitments elevates the Verifier's trust basis from numerical correctness to cryptographic binding, without leaking query or document vector contents.

**End-to-End Verifiable Multimodal Embedding Inference**: zkLLM is extended from a single-modality language model to a five-stage multimodal inference proof system (Conv3d → ViT → PatchMerger → LM → Pooling). Key contributions include cryptographic adaptation for GQA attention, precise NTT partitioning for window/global attention, and cross-component Sumcheck batching (gate/up projections and q/k/v projections each reduced to a single verification), enabling a Verifier to confirm that embedding vectors were genuinely computed by the agreed encoder without disclosing model parameters.

---

## 🚀 Quick Start

### Environment Setup

```bash
conda env create -f environment_no_builds.yml
conda activate ultrarag
```

Key dependencies: `torch 2.9.0+cu128`, `faiss-gpu`, `sentence-transformers`, `pymupdf`, `pyblst`, `mmh3`, `gradio`

**Model paths** (defaults):
- Retrieval encoder: `/path/to/models/jina-embeddings-v4`
- Generation model: `/path/to/models/MiniCPM-V-4`

### Compile zkLLM CUDA Binaries

```bash
cd src/zkllm
make all -j$(nproc)
mkdir -p bin && cp ppgen commit-param self-attn ffn rmsnorm skip-connection patch-merger conv3d-embed verify-ipa open-ipa bin/
```

Requires: CUDA ≥ 12.0, sm_89 (RTX 4090). Adjust `ARCH` in Makefile for other GPUs.

### Initialize Weight Commitments (One-Time Offline Setup)

```bash
python script/setup_all_visual_params.py \
    --model-path /path/to/models/jina-embeddings-v4 \
    --workdir zkllm-workdir/jina-v4
```

Extracts jina-v4 visual component weights (ViT, PatchMerger, Conv3d, RMSNorm) and generates cryptographic commitments stored in `zkllm-workdir/`. This only needs to run once.

### Build a Verifiable Corpus

```bash
python script/build_verifiable_corpus.py --pdf data/nikon.pdf
```

Automatically handles: PDF page extraction → jina-v4 encoding → FAISS indexing → ZAC corpus fingerprint → zkLLM pre-computed proofs

A sample PDF is available [here](https://download.nikonimglib.com/archive4/ywJ4K00fa2Lr05vv5OS00pV5Hg36/Z7Z6UM_TH(Sc)07.pdf).

### Launch the Demo

```bash
python script/interactive_demo.py
# Open http://127.0.0.1:7860 in your browser
```

> The demo enables the IPA vector commitment mode by default (the embedding commitment is initialized in one shot during `build_verifiable_corpus.py`); if the commitment artifacts are missing, it automatically falls back to the Mersenne-field Sumcheck path without affecting the demonstration.

## 📁 Repository Structure

```
UltraRAG/
├── script/
│   ├── interactive_demo.py              # Main demo (Gradio UI + full verification pipeline)
│   ├── build_verifiable_corpus.py       # One-command corpus builder
│   ├── build_corpus_zkllm_proofs.py     # Corpus-side zkLLM proof pre-computation
│   ├── phase1_corpus_fingerprint.py     # ZAC corpus fingerprint (standalone)
│   ├── phase2_sumcheck.py               # Sumcheck protocol demo and tests
│   ├── setup_all_visual_params.py       # Initialize all visual component commitments
│   ├── setup_{vit,patchmerger,conv3d,rmsnorm}_params.py  # Per-component commitments
│   ├── setup_embedding_commitments.py   # Corpus embedding IPA commitments
│   ├── verify_vit.py                    # ViT verification
│   ├── verify_layers.py                 # LLM layer verification
│   └── verify_zkllm_full.py             # Full zkLLM verification
├── src/
│   ├── zac/accumulator.py               # ZAC (BloomFilter + Pointproofs)
│   ├── sumcheck/inner_product.py        # Sumcheck (Global Batch mode)
│   └── zkllm/
│       ├── self-attn.cu                 # zkAttn (with GQA adaptation)
│       ├── ffn.cu / rmsnorm.cu          # FFN / RMSNorm proofs
│       ├── patch-merger.cu / conv3d-embed.cu / verify-ipa.cu
│       ├── Makefile                     # nvcc sm_89
│       ├── load_jina_weights.py         # jina-v4 weight extraction
│       └── bin/                         # Compiled binaries (build with make)
├── servers/                             # UltraRAG retrieval/generation modules (imported directly)
├── corpora/image.jsonl                  # Image corpus index
├── output/phase1/prover_state.json      # ZAC prover state + corpus fingerprint
└── zkllm-workdir/jina-v4/
    ├── corpus_proof_*.json              # Per-image zkLLM pre-computed proofs
    └── *.bin                            # jina-v4 weight commitment files
```

## ⚙️ Technology Stack

| Stage | Core Technology | Role |
|-------|----------------|------|
| Semantic Membership Verification | ZAC (Bloom Filter + Pointproofs, BLS12-381) | O(1) aggregated membership proof; cross-layer binding of images and embedding vectors |
| Similarity Ranking Verification | Global Batch Sumcheck (Fiat-Shamir, Mersenne field) | Full-corpus inner products verifiable; Verifier independently selects top-k; prevents ranking forgery |
| Encoder Inference Verification | zkLLM (Sumcheck + IPA, with GQA adaptation) | Weight commitments + layer-wise inference proofs covering full chain from ViT to LM |

## 🔒 Trust Model

The service provider (Prover) holds complete model weights and corpus, and is responsible for running inference and returning proofs alongside retrieval results. The user or auditor (Verifier) only needs to hold public commitment values: the ZAC Root (96 bytes, published once after corpus construction) and per-layer model weight commitments (published once at deployment). The Verifier requires neither the model nor the original corpus to independently verify the three-stage proof for any retrieval request.

## 💡 Applications

**High-Stakes Decision Support**: Increasingly strict regulatory requirements in medical imaging, legal document retrieval, and financial compliance analysis demand auditability. The three-stage verification ensures retrieval results are traceable and unforgeable across the full decision pipeline.

**Multi-Party Federated Knowledge Bases**: When participants mutually distrust one another, anchoring the ZAC Root on a blockchain lets any party independently verify that returned results originate from the agreed shared corpus. Sumcheck prevents any single party from manipulating rankings — enabling trustworthy federated retrieval without sharing raw data.

**AI Service Auditability**: Service providers can furnish verifiable inference proofs without exposing model weights, complementing content provenance standards like C2PA for the "retrieval process" layer they do not cover.

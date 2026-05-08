<div align="center">

# VeriMMR

**[中文](#中文) | [English](#english)**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu128-ee4c2c?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Third Party](https://img.shields.io/badge/Includes-Apache%202.0%20%7C%20MIT-lightgrey)

</div>

---

<a id="中文"></a>

# VeriMMR：面向多模态语义检索的可验证框架

> 在不信任服务提供方的前提下，对"编码—嵌入—匹配—排序"全链路施加密码学约束，使任意第三方可独立验证多模态语义数据检索结果的完整性和可靠性。

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

## 🚀 快速开始

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

示例 PDF 可从[此处](https://download.nikonimglib.com/archive4/ywJ4K00fa2Lr05vv5OS00pV5Hg36/Z7Z6UM_TH(Sc)07.pdf)下载。

### 启动 Demo

```bash
python script/interactive_demo.py
# 浏览器访问 http://127.0.0.1:7860
```

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

| 阶段 | 核心技术 | 作用 |
|------|---------|------|
| 语义成员资格验证 | ZAC（Bloom Filter + Pointproofs，BLS12-381） | O(1) 聚合成员证明，图像与嵌入向量跨层绑定 |
| 相似度排序验证 | Global Batch Sumcheck（Fiat-Shamir，Mersenne 域） | 全量内积可验证，Verifier 独立选 top-k，防排名伪造 |
| 编码器推理验证 | zkLLM（Sumcheck + IPA，含 GQA 适配） | 权重承诺 + 逐层推理证明，覆盖 ViT 到 LM 全链路 |

## 🔒 信任模型

服务提供方（Prover）持有完整模型权重和语料库，负责运行推理并随检索结果返回证明。用户或审计方（Verifier）只需持有公开承诺值：ZAC Root（48 字节，建库后一次性发布）和模型各层权重承诺（部署时发布一次）。Verifier 不需要模型，也不需要原始语料库，可独立验证任意检索请求的三阶段证明。

## 💡 应用场景

**高风险决策支持**：医疗影像辅助诊断、法律文书检索、金融合规分析等场景监管要求日趋严格。三阶段验证保证检索结果可追溯、不可伪造，使 AI 辅助决策的全过程具备可审计性。

**多方联合知识库**：各参与方互不信任时，将 ZAC Root 存证于区块链，任意一方可独立验证其他方返回的检索结果来自约定的共同语料，Sumcheck 防止单方操控排名——在不共享原始数据的前提下实现可信联合检索。

**AI 服务可审计性**：服务提供方可在不公开模型权重的前提下，向用户或监管机构提供推理可验证性证明，补充 C2PA 等内容溯源标准未覆盖的"检索过程"环节。

---

<a id="english"></a>

# VeriMMR: A Verifiable Framework for Multimodal Semantic Retrieval

> Applying cryptographic constraints across the full "encode–embed–match–rank" pipeline without trusting the service provider, enabling any third party to independently verify the integrity and reliability of multimodal semantic retrieval results.

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

## 🚀 Quick Start

### Environment Setup

```bash
conda env create -f environment_no_builds.yml
conda activate ultrarag
```

Key dependencies: `torch 2.8.0+cu128`, `faiss-gpu`, `sentence-transformers`, `pymupdf`, `py_ecc`, `mmh3`, `gradio`

**Model paths** (defaults):
- Retrieval encoder: `/path/to/models/jina-embeddings-v4`
- Generation model: `/path/to/models/MiniCPM-V-4`

### Compile zkLLM CUDA Binaries

```bash
cd src/zkllm
make all -j$(nproc)
mkdir -p bin && cp ppgen commit-param self-attn ffn rmsnorm skip-connection patch-merger conv3d-embed verify-ipa bin/
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

The service provider (Prover) holds complete model weights and corpus, and is responsible for running inference and returning proofs alongside retrieval results. The user or auditor (Verifier) only needs to hold public commitment values: the ZAC Root (48 bytes, published once after corpus construction) and per-layer model weight commitments (published once at deployment). The Verifier requires neither the model nor the original corpus to independently verify the three-stage proof for any retrieval request.

## 💡 Applications

**High-Stakes Decision Support**: Increasingly strict regulatory requirements in medical imaging, legal document retrieval, and financial compliance analysis demand auditability. The three-stage verification ensures retrieval results are traceable and unforgeable across the full decision pipeline.

**Multi-Party Federated Knowledge Bases**: When participants mutually distrust one another, anchoring the ZAC Root on a blockchain lets any party independently verify that returned results originate from the agreed shared corpus. Sumcheck prevents any single party from manipulating rankings — enabling trustworthy federated retrieval without sharing raw data.

**AI Service Auditability**: Service providers can furnish verifiable inference proofs without exposing model weights, complementing content provenance standards like C2PA for the "retrieval process" layer they do not cover.

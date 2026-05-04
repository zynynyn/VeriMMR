# 架构图 SCSS 描述文件

本文件收录论文各阶段架构图的 SCSS（Subject / Composition / Structure / Style）描述，
供 AI 图像生成工具（如 GPT-4o、Midjourney 等）或手工绘图参考。

---

## 图1：Phase 2 全局批量 Sumcheck 可验证检索架构

```
Subject（主题）
  全局批量 Sumcheck 可验证检索架构（Phase 2）
  ——从语料库建库到在线检索响应，再到客户端独立验证的完整信任链

Composition（整体布局）
  纵向三段式布局，从上至下依次为：
    ① 离线预处理（浅灰色背景块，标注"一次性建库"）
    ② 在线检索响应（白色背景，标注"每次查询"）
    ③ 独立 Verifier（浅蓝色背景块，标注"客户端，无需信任 Prover"）
  三段之间用粗箭头连接，右侧附"公开信道"与"可信分发"标注列。
  整体宽度适中，高度约为宽度的 2 倍，适合竖排论文插图。

Structure（结构细节）

  ┌─────────────────────────────────────────────────────┐
  │  ① 离线预处理（Prover 端，一次性）                      │
  │                                                     │
  │  Phase 3 zkLLM 推理                                  │
  │    图像 I_i  →  jina-v4 编码  →  embedding v_i        │
  │                    ↓  量化（scale=65536）               │
  │                   v̂_i ∈ ℤ^d（int32）                  │
  │                    ↓  commit-param（GPU, ~0.4s/条）      │
  │                   cm_i ∈ 𝔾₁（144 B Jacobian G1 点）     │
  │                                                     │
  │  输出（公开分发，一次性）：                               │
  │    • ZAC Root（48 B）——Phase 1 语料库指纹               │
  │    • {cm_i}_{i=1}^N ——embedding_commitments.bin     │
  │      N=303 × 144 B = 43 KB（Verifier 持有此文件）       │
  └───────────────────┬─────────────────────────────────┘
                      │  公开分发（离线可信渠道）
  ┌───────────────────▼─────────────────────────────────┐
  │  ② 在线检索响应（Prover 端，每次查询）                    │
  │                                                     │
  │  输入：用户查询 q（float32 × 2048）                     │
  │    ↓  量化 → q̂_int                                   │
  │                                                     │
  │  Step A：计算 N 个内积分值                              │
  │    s_i = ⟨q̂, v̂_i⟩ mod P_FR,  i = 1…N（N=303）        │
  │                                                     │
  │  Step B：Fiat-Shamir 聚合                             │
  │    ρ = SHA256(s_1 ‖ … ‖ s_N)  ← 防篡改锚点            │
  │    w = Σᵢ ρⁱ · v̂_i（聚合向量，dim=2048）               │
  │    s_batch = Σᵢ ρⁱ · sᵢ                              │
  │                                                     │
  │  Step C：单次 Sumcheck（11 轮）                         │
  │    证明 ⟨q̂, w⟩ = s_batch                             │
  │    → sc_proof（264 B = 11×3×8B）                      │
  │                                                     │
  │  Step D：IPA Oracle 开放（open-ipa，GPU，~2s）           │
  │    在 Sumcheck 挑战点 r* = (r₁,…,r₁₁) 处开放 w        │
  │    → oracle_proof（3,852 B）                          │
  │                                                     │
  │  输出（返回给 Verifier）：                               │
  │    {s_i}_{i=1}^N + sc_proof + oracle_proof           │
  │    总传输 ≈ 303×32 + 264 + 3852 ≈ 13.8 KB            │
  └───────────────────┬─────────────────────────────────┘
                      │  返回检索响应
  ┌───────────────────▼─────────────────────────────────┐
  │  ③ 独立 Verifier（客户端，无需信任 Prover）               │
  │                                                     │
  │  持有（离线获取，无需在线更新）：                           │
  │    • {cm_i}（43 KB）                                 │
  │    • ZAC Root（48 B）                                │
  │                                                     │
  │  Check 1：防分值替换                                   │
  │    重推 ρ' = SHA256({s_i}) → 检查 ρ' == proof.ρ       │
  │    ✗ → 任意分值被替换即拒绝                              │
  │                                                     │
  │  Check 2：线性一致性                                   │
  │    验证 s_batch == Σᵢ ρⁱ · s_i                        │
  │    ✗ → s_batch 与分值列表不一致即拒绝                    │
  │                                                     │
  │  Check 3：Sumcheck 协议验证（11 轮 fold check）          │
  │    逐轮验证 g_i(0)+g_i(1) == g_{i-1}(r_{i-1})         │
  │    最终 oracle 等式检查                                 │
  │    ✗ → 内积声称值错误即拒绝                              │
  │                                                     │
  │  Check 4：IPA Oracle 绑定验证                          │
  │    聚合 cm_w = Σᵢ ρⁱ · cm_i  （G1 标量乘，~3s）         │
  │    verify_ipa_embedding(oracle_proof, cm_w)          │
  │      Binding check：C_init == cm_w ？                 │
  │      Fold check：11 轮折叠链 → w_final 合法？           │
  │    ✗ → w 与承诺不绑定即拒绝（无需持有原始 embedding）      │
  │                                                     │
  │  最终输出：                                            │
  │    Verifier 独立排序 {s_i} → top-k（不信任 Prover 排名） │
  └─────────────────────────────────────────────────────┘

  [右侧安全标注列]
    Check 1 旁：Fiat-Shamir 锁定，修改任意 s_i → ρ 变化 → sc_proof 失效
    Check 3 旁：Soundness ≤ (N+22)/P_FR ≈ 2^{-247}（BLS12-381 Fr 域）
    Check 4 旁：IPA Binding 基于离散对数困难假设（计算不可区分）
    跨段连接旁：ZAC 绑定——cm_i 与 Phase 3 zkLLM proof 共享 embedding 来源

Style（视觉风格）
  三段背景色：离线预处理浅灰、在线响应白色、独立 Verifier 浅蓝；
  Check 1–4 的验证步骤用绿色勾/红色叉图标标注"可证伪点"；
  数据流箭头黑色实线，密码学承诺路径橙色虚线；
  关键尺寸（43KB / 264B / 3852B / 2^{-247}）用 monospace 字体加粗标注；
  图注下方附两栏对比小表：
    non-IPA：Verifier 持有 2.4MB，proof 264B，soundness 2^{-53}
    IPA/GPU：Verifier 持有 43KB，proof 4116B，soundness 2^{-247}，verify ~5.3s/query
```

---

## 图2：Phase 3 zkLLM 可验证推理流水线

```
Subject（主题）
  zkLLM 可验证多模态编码流水线（Phase 3）
  ——权重承诺建库、推理证明生成、客户端独立验证三段信任链

Composition（整体布局）
  纵向三段式，从上至下：
    ① 离线预处理（浅灰背景，标注"一次性"）
    ② 在线推理 + 证明生成（白色背景，上半=语料侧/图像，下半=查询侧/文本）
    ③ 独立 Verifier（浅蓝背景）
  整体高度约为宽度的 2 倍，适合竖排论文插图。

Structure（结构细节）

  ┌─────────────────────────────────────────────────────┐
  │  ① 离线预处理（一次性）                                  │
  │                                                     │
  │  A. 权重承诺生成                                       │
  │     对所有层权重 W_i：commit-param → cm_W（G1 点）       │
  │     输出：{cm_W}（权重承诺集，公开分发，~40MB）             │
  │                                                     │
  │  B. 语料图像推理 + proof 预计算（N=303 张）               │
  │     每张图 I_i：                                      │
  │       图像 → 模型推理 → embedding v_i（2048维）          │
  │       同步生成 proof bundle（绑定 cm_W）：               │
  │         Conv3d + ViT 32块 + PatchMerger → IPA proofs │
  │         LM 全部 36 层 → IPA proofs                   │
  │         Pooling → Sumcheck（MeanPool）+ Rescaling（L2Norm）│
  │       → corpus_proof_i（离线存储，查询时读取 <1ms）       │
  │     → ZAC 建库：SHA256(I_i ‖ v_i) → ZAC Root（48B）   │
  │                                                     │
  │  输出（公开分发）：{cm_W}、ZAC Root、corpus_proof_i       │
  └───────────────────┬─────────────────────────────────┘
                      │  公开分发
  ┌───────────────────▼─────────────────────────────────┐
  │  ② 在线推理 + 证明生成                                  │
  │                                                     │
  │  ── 语料侧（图像，离线预计算，查询时直接读取）──               │
  │     corpus_proof_i 已涵盖完整图像路径（全量 5 组件）       │
  │     查询时延迟：<1ms                                   │
  │                                                     │
  │  ── 查询侧（文本，实时生成，异步后台）────────────────      │
  │     输入：文本 query q                                 │
  │     Fiat-Shamir 层选择：                              │
  │       challenge = SHA256(q ‖ client_nonce)           │
  │       → 随机选 K=6 层（从全部 36 层中选）                │
  │       每次查询独立覆盖 1/6 层，多轮累积检出               │
  │     推理：Token Embedding → LM K=6 层                 │
  │     同步生成 proof bundle（绑定 cm_W）：               │
  │       LM 6层 × 6 → 36个 IPA proof                    │
  │       Pooling → Sumcheck proof                       │
  │     → embedding v_q + query_proof_bundle             │
  │     查询时延迟：~46s（异步后台）                         │
  │     ※ 审计/存证场景可配置全 36 层（~7 min）             │
  └───────────────────┬─────────────────────────────────┘
                      │  proof bundle → Verifier
  ┌───────────────────▼─────────────────────────────────┐
  │  ③ 独立 Verifier（无需信任 Prover）                     │
  │                                                     │
  │  持有（离线获取）：                                     │
  │    {cm_W}（~40MB）、ZAC Root（48B）                    │
  │    {cm_i}（embedding 承诺，43KB，用于 Phase 2 IPA oracle）│
  │                                                     │
  │  ── 离线语料侧验证（corpus_proof 预计算，一次性）──        │
  │                                                     │
  │  Step 1：IPA 批量验证（GPU，verify-ipa binary）         │
  │    508 个 IPA proof（Conv3d×1 + ViT×288 +            │
  │    PatchMerger×3 + LLM corpus×216）：                 │
  │      fold_ok：折叠链数值自洽（k=11 轮 Sumcheck）         │
  │      binding_ok：opening == cm_W(u_out) ✓            │
  │    C3b 实验：binding_ok=False → 篡改检出 100%          │
  │                                                     │
  │  Step 2：MeanPool Sumcheck 验证                       │
  │    8 个随机挑战 r：Σ_{t:mask}(H[t]·r) == p·r ✓        │
  │    证明 p = mean_pool(last_hidden[mask]) 正确          │
  │    [注] L2Norm 当前为量化误差界检查，非正式 ZK 证明        │
  │                                                     │
  │  ── 在线查询侧验证（每次查询，半诚实模型）──               │
  │                                                     │
  │  Step 3：Fiat-Shamir 层选择可复现                      │
  │    Verifier 独立重算 challenge = SHA256(q ‖ nonce)    │
  │    确认 K=6 层选择非 Prover 可控                       │
  │                                                     │
  │  Step 4：查询侧 IPA 验证（K=6 层，每层 3 步）            │
  │    FFN binary rc=0 + Attn-linear rc=0 + zkAttn rc=0 │
  │    （C++ 内部：Sumcheck 一致 + verifyWeightClaim 通过） │
  │                                                     │
  │  Step 5：Phase 2 Sumcheck + IPA Oracle 验证           │
  │    Check 1: ρ' = SHA256({s_i}) == proof.ρ ✓          │
  │    Check 2: s_batch == Σρⁱ·sᵢ ✓                     │
  │    Check 3: 11 轮 Sumcheck fold check ✓              │
  │    Check 4: cm_w = Σρⁱ·cm_i → oracle binding ✓      │
  │                                                     │
  │  Step 6：ZAC 成员验证（语料侧）                          │
  │    SHA256(I_i ‖ v_i) 配对方程 → ∈ ZAC Root ✓         │
  │                                                     │
  │  最终判断（interactive demo）：                        │
  │    all_ok = Step1–2(corpus_ok) AND Step3–4(zkllm_ok) │
  │             AND Step5(sc_ok) AND Step6(zac_ok)      │
  └─────────────────────────────────────────────────────┘

  [关键数字注释]
    离线建库：~76s/张（单 GPU），303 张 ≈ 6.42h
    语料侧 proof：~1.9MB/张（508 个 IPA proof）
    查询侧 proof：~70KB（18 个 IPA proof + Sumcheck）
    独立验证：≤ 11 min / 语料图（2 GPU 并行）

Style（视觉风格）
  三段背景色：离线浅灰、在线白色、Verifier 浅蓝；
  ② 段内部用虚线分隔语料侧与查询侧，分别标注耗时；
  proof bundle 用统一的深灰圆角小框表示，不展开文件名；
  Verifier 三步用编号绿色图标；
  图注下无对比表格，仅保留关键数字注释列。
```

---

## 图3：Phase 3 编码器完整可验证架构（双路径 + 优化适配）

```
Subject（主题）
  jina-v4 / Qwen2.5-VL 多模态编码器的完整可验证计算架构
  ——视觉路径与文本路径的 zkLLM 适配、GQA/Window Attention 证明方案
    及四项工程优化的全景展示

Composition（整体布局）
  左右双列：左列为"计算数据流"（前向推理），右列为"证明生成流"（逐层 IPA）；
  两列之间用虚线水平对应连接（每个模块 ↔ 其 proof）；
  纵向从上至下分三个区段：
    上段：输入分支（视觉路径 / 文本路径 并行）
    中段：ViT 视觉编码（视觉路径专属，32 blocks）
    下段：语言模型（LM，双路径共用，36 layers）+ 输出；
  右侧额外附"优化标注气泡列"，指向对应模块；
  整体宽幅横排，高度约为宽度的 1.5 倍。

Structure（结构细节）

  ══════════════════════════════════════════════════════
  上段：输入分支
  ══════════════════════════════════════════════════════

  [视觉路径]                         [文本路径]
  图像（RGB，任意分辨率）              查询文本
    ↓ 预处理 → 448×448 tile           ↓ Tokenizer
    ↓ Conv3d Patch Embed              ↓ Token Embedding（词表×2048）
      kernel 2×14×14                 → 文本 token 序列（seq×2048）→ LM 输入
      → 1024 patch tokens            证明：Embedding 查表（不需 IPA，
        （dim=1280）                       权重维度小，信任边界）
      证明：矩阵乘展开
            → conv3d-embed-ipa-proof.bin

  ══════════════════════════════════════════════════════
  中段：ViT 编码（视觉路径，32 Blocks）
  ══════════════════════════════════════════════════════

  输入：1024 patch tokens（seq=1024，dim=1280）

  Block 类型 A：Window Attention（共 28 块：0–6, 8–14, 16–22, 24–30）
  ┌──────────────────────────────┐   ┌────────────────────────────┐
  │ Python 预分窗：1024→16×64     │   │ 16 次分窗分别调用 self-attn  │
  │ RMSNorm（dim=1280）          │   │ → rmsnorm-ipa-proof.bin     │
  │ Q proj（1280→1280）          │   │ → q-ipa-proof.bin           │
  │ K proj（1280→1280，n_kv=16） │   │ → k-ipa-proof.bin           │
  │ V proj（1280→1280）          │   │ → v-ipa-proof.bin           │
  │ MHA（WIN_SEQ=64，seq²=4096=  │   │ → softmax-proof.bin        │
  │   2^12，满足 NTT 约束）       │   │   （tLookup，CUB histogram）  │
  │ O proj（1280→1280）          │   │ → o-ipa-proof.bin           │
  │ RMSNorm₂ + SwiGLU FFN       │   │ → rmsnorm2-ipa-proof.bin    │
  │   gate_proj（3420×1280）     │   │ → gate-ipa-proof.bin        │
  │   up_proj  （3420×1280）     │   │ → up-ipa-proof.bin          │
  │   down_proj（1280×3420）     │   │ → down-ipa-proof.bin        │
  └──────────────────────────────┘   └────────────────────────────┘
    每块 9 个 IPA proof

  Block 类型 B：Full Attention（共 4 块：7, 15, 23, 31）
  ┌──────────────────────────────┐
  │ 同 Block A，但 seq=1024       │   证明同 Block A（9 个 proof/块）
  │ seq²=1024²=2^20 ✓（NTT）     │   seq 更大，耗时更长
  │ n_kv=16（ViT 是 MHA，非 GQA）│
  └──────────────────────────────┘

  → Spatial Merge MLP（PatchMerger）
    4×concat（1280→5120）→ Linear+GELU → Linear（5120→2048）
    → 视觉 token 序列（seq'×2048）
    证明 → patchmerger-ipa-proof.bin

  ══════════════════════════════════════════════════════
  下段：LM 36 Decoder Layers（双路径共用）
  ══════════════════════════════════════════════════════

  输入：视觉 token（seq'×2048）+ 文本 token（seq_t×2048）拼接

  每层 Decoder（×36）：
  ┌──────────────────────────────┐   ┌────────────────────────────┐
  │ RMSNorm₁（dim=2048）         │   │ → rmsnorm-ipa-proof.bin     │
  │                              │   │                            │
  │ GQA Self-Attention           │   │                            │
  │   Q proj（2048→2048，n_q=16）│   │ → q-ipa-proof.bin           │
  │   K proj（2048→256，n_kv=2） │   │ → k-ipa-proof.bin           │
  │   V proj（2048→256）         │   │ → v-ipa-proof.bin           │
  │   KV 广播：group=8（8Q共1KV）│   │ ↑ GQA 转置技巧              │
  │   QKᵀ/√d → tLookup Softmax  │   │ → softmax-proof.bin        │
  │   （CUB histogram 并行）     │   │   （NTT Sumcheck）           │
  │   seq=1024，seq²=2^20 ✓      │   │                            │
  │   O proj（2048→2048）        │   │ → o-ipa-proof.bin           │
  │                              │   │                            │
  │ RMSNorm₂ + SwiGLU FFN       │   │ → rmsnorm2-ipa-proof.bin    │
  │   gate_proj（11008×2048）    │   │ → gate-ipa-proof.bin ┐     │
  │   up_proj  （11008×2048）    │   │ → up-ipa-proof.bin   ├ Batch│
  │   down_proj（2048×11008）    │   │ → down-ipa-proof.bin ┘ Sumcheck│
  └──────────────────────────────┘   └────────────────────────────┘
    每层 6 个 IPA proof（rmsnorm×1 + q/k/v/o×4 + ffn×3 → 实际8，
    Batch Sumcheck 将 gate+up 合并，down 独立）

  → Mean Pooling（mask=1 的 token 求均值）+ L2 Norm
    证明 → pooling-sumcheck-proof.json（Python Sumcheck）

  → 输出 embedding（dim=2048）
    → ZAC 登记（Phase 1 绑定）
    → Phase 2 Sumcheck 检索证明

  ══════════════════════════════════════════════════════
  [右侧优化标注气泡列]
  ══════════════════════════════════════════════════════

  ① GQA 转置技巧（LM 层，K/V 路径旁）
     "n_kv=2 广播 → n_q=16，避免显式复制
      转置 KV 后直接 group-per-head loop，
      attn score = QK^T 后按 group 归��"

  ② Window Attention NTT 适配（ViT Block A 旁）
     "WIN_SEQ=64 → win²=4096=2^12 ✓ NTT 约束
      Python 预分窗：(1024,1280)→16×(64,1280)
      16 次独立 zkAttn 调用，每次 seq=64"

  ③ CUB Histogram 并行 tLookup（Softmax 旁）
     "tLookup_phase1/2 改用 GPU CUB 设备级 histogram
      替换串行 for-loop 扫描，Softmax 吞吐 3× 提升"

  ④ Batch Sumcheck：gate+up 共享输入（FFN 旁）
     "gate/up 共享同一 X_reduced（partial_me 计算一次）
      合并为单次 zkip，节省 ~40% FFN Prover 时间"

  ⑤ verify-ipa 批量模式（右列汇总旁）
     "一次 GPU kernel 批量验证 N 个 IPA proof
      9 proof/block × 32 blocks = 288 proof 并行
      2.5× 加速 vs 串行验证"

  ⑥ 2-GPU 并行验证（验证阶段顶部）
     "ViT 32 blocks → GPU0/GPU1 各 16 blocks
      LLM 36 layers → GPU0 偶数层，GPU1 奇数层
      总验证时间 ≤ 11 min（vs 单 GPU ~27 min）"

Style（视觉风格）
  双列布局：左列（计算）白色背景，右列（证明）浅黄色背景；
  三区段（输入/ViT/LM）用不同粗细分隔线区分，标注区段名；
  视觉路径 vs 文本路径用不同颜色（图像蓝色 / 文本绿色）的分支箭头；
  Block A/B 用实线/虚线矩形框区分（Window=实线，Full=虚线）；
  优化气泡用黄色圆角矩形，红色加粗字体，指向线橙色虚线；
  GQA group 结构示意：左侧 8 个小方块（Q heads）指向 1 个大方块（KV head），
    嵌套在 K/V proj 旁边；
  NTT 约束用灰色小标注框（"seq²=2^k ✓"）附在每个 Attn 模块右下角
```

---

## 图4：端到端 Prover-Verifier 关系图

```
Subject（主题）
  UltraRAG 可验证多模态检索系统端到端信任链
  ——三阶段证明的 Prover/Verifier 角色分工与密码学绑定关系

Composition（整体布局）
  纵向三层结构，从下至上依次为 Phase 1（ZAC corpus 指纹）、
  Phase 2（Sumcheck 排名验证）、Phase 3（zkLLM 模型计算验证）；
  每层内部横向分为 Prover（左）和 Verifier（右）两列；
  层间用纵向"绑定箭头"连接，表示密码学承诺的跨层关联。

Structure（结构细节）

  [Phase 1 层 — ZAC Corpus 指纹]
    Prover：
      N 张图像 → SHA256(image_bytes_i ‖ emb_i) → Bloom Filter → Pointproofs
      → ZAC Root（48B G1 点）+ prover_state.json
    Verifier：
      持有 ZAC Root → 收到成员证明 π̂ → 配对验证
      e(cm, Σtᵢ·g₂^{α^{q+1-i}}) ?= e(π̂, g₂)·gT^{...}
    绑定：Root 承诺全部 (image, embedding) 对，防止替换攻击

  [Phase 2 层 — Sumcheck 排名验证]
    Prover：
      接收 query q → 计算 N 个内积 {s_i} → Fiat-Shamir ρ
      → 单次 Sumcheck over w=Σρⁱvᵢ → sc_proof
      → （IPA）open-ipa → oracle_proof
    Verifier：
      重推 ρ → 验证 s_batch → fold 11 轮 Sumcheck
      → （IPA）聚合 cm_w=Σρⁱcm_i → verify_ipa_embedding
      → 独立排序 {s_i} → 输出 top-k
    绑定：IPA oracle 将 w 绑定到 {cm_i}，{cm_i} 与 Phase 1 embedding 同源

  [Phase 3 层 — zkLLM 模型计算验证]
    Prover（服务器，含 GPU）：
      原始图像 → ViT 32 blocks（每块 9 次 IPA） → PatchMerger → Pooling
      → embedding → 写入 Phase 1 ZAC
      证明文件：288 个 ViT IPA proof + 各层 rmsnorm/ffn proof
    Verifier（轻量客户端）：
      持有模型承诺 {cm_W}（离线分发）
      → verify-ipa binary（GPU，批量 2.5×加速）
      → 2-GPU 并行（32 ViT blocks ≤ 3min，36 LLM layers ≤ 5min）
      → 全部 binding_ok=True → 确认计算未被篡改
    绑定：IPA 承诺将模型权重矩阵 W 绑定到公开参数 pp，
          Fiat-Shamir 随机挑战防止 Prover 事后选择有利挑战点

  [跨层绑定箭头]
    Phase 3 embedding 输出 → Phase 1 ZAC 成员证明（image+emb 哈希）
    Phase 1 ZAC cm_i      → Phase 2 IPA oracle cm_w 聚合
    Phase 2 top-k 结果    → 返回用户（已密码学保证排名完整性）

  [威胁模型标注框]
    "B1 图像替换" → 被 Phase 1 ZAC 检测
    "B2 Embedding 替换" → 被 Phase 1 ZAC 跨层绑定检测
    "B3 排名操控" → 被 Phase 2 Sumcheck Schwartz-Zippel 检测
    "B4 权重篡改" → 被 Phase 3 IPA binding 检测

Style（视觉风格）
  三层用不同背景色区分：Phase 1 浅绿、Phase 2 浅蓝、Phase 3 浅紫；
  Prover 列用暖色调（橙/红边框），Verifier 列用冷色调（蓝/绿边框）；
  跨层绑定箭头用粗黑实线 + 锁形图标；
  威胁模型标注框用红色虚线边框，置于图右侧；
  图注下方附整体性能表：离线建库时间、在线证明时间、在线验证时间
```

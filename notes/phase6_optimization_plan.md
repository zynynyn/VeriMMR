# Phase 6 优化实施计划

> 创建时间：2026-04-30  
> 前置：Phase 1–5 已完成（ZAC corpus fingerprint、zkLLM 36层IPA、ViT 32块、PatchMerger、Pooling、C++ verify-ipa、2-GPU并行）

---

## 背景与动机

### 当前系统状态（Phase 5 完成后）

| 组件 | 耗时（Phase 5 优化后） | 备注 |
|------|----------------------|------|
| ViT 32 blocks（2-GPU） | ~3 min | C++ verify-ipa + ProcessPoolExecutor |
| LLM 36 layers（2-GPU） | ~5 min | C++ verify-ipa + parallel |
| PatchMerger | ~0.1s | 可忽略 |
| Conv3d | ~0.7s | 可忽略 |
| **全量证明/页** | **~8 min** | 离线摊销 |

### 两个待解决问题

**问题1：ViT soundness gap**  
`build_corpus_full_proof.py` 中 `SEQ_VIT=1024`，对真实图像（nikon 文档页产生 2520 patches）做截断。ZK 证明只覆盖前 1024 个 patch，后 1496 个 patch 未被证明，构成 soundness gap。原来 32/32 pass 是因为 `_make_input` 生成随机数据（无真实 token 截断问题），且所有块使用 full attention（seq=1024）而非 window attention。

**问题2：验证速度优化**  
zkGPT (EuroSys 2024) 提出两项技术，经分析适用于当前系统：
- **Constraint Fusion**：合并相邻 rounding 约束，减少 tLookup range check 次数 ~50%
- **Circuit Squeeze**：不同输入的矩阵乘可批量进单个 sumcheck，1.7× 加速
- 两者联合：prover 侧约 **2.5× 加速**，全量证明从 8 min → ~3 min

### 在线查询验证可行性

文字查询 seq=64（32-128 token），zkAttn 仅 0.2s/层，36层合计估计 **15-30s**（加入优化2+3后）。这意味着：
- 离线（corpus side）：每页几分钟，完全可接受
- 在线（query side）：短文本查询近实时可验证
- **论文贡献点**：从单边证明（corpus only）扩展为双边可验证 RAG

---

## 实施步骤

---

### Step A：修复 ViT Soundness Gap（Option A）

**目标**：对真实图像正确处理全部 n_patches（如2520），不截断；window attention 块按真实窗口数运行，full attention 块 pad 到最近2的幂次（≤4096）。

#### A1. `build_corpus_full_proof.py` 修改

**当前问题**（line 169）：
```python
_pad_save(captured[key], SEQ_VIT, path)   # SEQ_VIT=1024 → 截断
```

**修改方案**：对 ViT block inputs **不截断**，直接保存全量 patches：
```python
# 修改 _pad_save 为不截断版本（ViT block 专用）
def _save_full(act: torch.Tensor, path: Path):
    if act.dim() == 3:
        act = act[0]
    S, D = act.shape
    (act * SCALE).round().to(torch.int32).numpy().astype(np.int32).tofile(str(path))
    files.append(path)
    return S

# ViT block inputs（lines 162-169）改为：
for bi in range(VIT_N_BLOCKS):
    key = f"vit_{bi}"
    if key not in captured:
        continue
    vit_wd = workdir / f"vit-b{bi}"
    vit_wd.mkdir(exist_ok=True)
    path = vit_wd / f"vit-block-{bi}-h_in.bin"
    n_real = _save_full(captured[key], path)
    # 记录真实长度（供 verify_vit_block 使用）
    meta[f"vit_b{bi}_n_patches"] = n_real
```

同时在 `meta` 中保存每个 block 的真实 patch 数，并在 JSON 结果中记录。

#### A2. `verify_vit.py` 修改

**当前问题**：
- `SEQ_LEN=1024`、`NUM_WINS=16` 硬编码
- `_make_input` 始终生成 1024×1280 随机数（real data 时需从 h_in.bin 读取）
- `_split_qkv_for_windows` 硬编码从 `SEQ_LEN` 大小切割
- full attention 块也硬编码 seq=1024

**修改方案**：

**① 新增辅助函数 `_get_seq_params(h_in_path, block_idx)`**：
```python
def _get_seq_params(h_in_path: Path, block_idx: int):
    """
    根据 h_in.bin 的实际大小推导 seq_len 和 pad_seq。
    - n_real: 文件中实际 patch 数
    - seq_len: window blocks → WIN_SEQ=64（每窗口）；full → pad 到最近2^k
    - n_wins: window blocks → ceil(n_real / WIN_SEQ)；full → 1
    """
    if not h_in_path.exists():
        return SEQ_LEN, SEQ_LEN, NUM_WINS  # fallback: 随机数模式
    total_ints = h_in_path.stat().st_size // 4
    n_real = total_ints // VIT_HIDDEN
    if block_idx in FULL_ATT_BLOCKS:
        # pad 到最近 2^k（≥ n_real），最大 4096
        pad = 1
        while pad < n_real:
            pad <<= 1
        return n_real, pad, 1
    else:
        n_wins = (n_real + WIN_SEQ - 1) // WIN_SEQ   # ceil
        return n_real, WIN_SEQ, n_wins
```

**② 修改 `_split_qkv_for_windows`**：接受 `n_real` 和 `n_wins` 参数，不再硬编码 `SEQ_LEN`：
```python
def _split_qkv_for_windows(vit_wd, prefix, n_real, n_wins):
    q_src = vit_wd / f"{prefix}-temp_Q.bin"
    if not q_src.exists():
        return False
    Q = np.fromfile(str(q_src), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    K = np.fromfile(str(vit_wd/f"{prefix}-temp_K.bin"), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    V = np.fromfile(str(vit_wd/f"{prefix}-temp_V.bin"), dtype=np.int32).reshape(n_real, VIT_HIDDEN)
    for w in range(n_wins):
        sl = slice(w * WIN_SEQ, min((w+1)*WIN_SEQ, n_real))
        chunk_q = Q[sl]; chunk_k = K[sl]; chunk_v = V[sl]
        # 如果最后一窗口不足 WIN_SEQ，pad to WIN_SEQ
        if chunk_q.shape[0] < WIN_SEQ:
            pad_rows = WIN_SEQ - chunk_q.shape[0]
            chunk_q = np.vstack([chunk_q, np.zeros((pad_rows, VIT_HIDDEN), np.int32)])
            chunk_k = np.vstack([chunk_k, np.zeros((pad_rows, VIT_HIDDEN), np.int32)])
            chunk_v = np.vstack([chunk_v, np.zeros((pad_rows, VIT_HIDDEN), np.int32)])
        wp = f"{prefix}-win{w}"
        chunk_q.tofile(str(vit_wd/f"{wp}-temp_Q.bin"))
        chunk_k.tofile(str(vit_wd/f"{wp}-temp_K.bin"))
        chunk_v.tofile(str(vit_wd/f"{wp}-temp_V.bin"))
    return True
```

**③ 修改 `verify_vit_block`**：在函数开头调用 `_get_seq_params`，把 `SEQ_LEN`/`NUM_WINS` 替换为动态值：
```python
def verify_vit_block(block_idx, workdir, gpu_id=0):
    prefix = f"vit-block-{block_idx}"
    h_in = vit_wd / f"{prefix}-h_in.bin"
    n_real, seq_len_eff, n_wins = _get_seq_params(h_in, block_idx)
    
    # _make_input 仅在 h_in 不存在时（随机模式）使用 SEQ_LEN
    _make_input(h_in, SEQ_LEN, VIT_HIDDEN)   # 只有文件不存在才写
    
    # Step 1 rmsnorm：使用 seq_len_eff（full: pad后大小；window: 全量n_real）
    # 注意：rmsnorm 输入是全量序列，seq 参数应为 n_real（pad到seq_len_eff）
    ...
    
    # Step 2 self-attn linear：seq = n_real（pad到seq_len_eff传给binary）
    ...
    
    # Step 3 zkAttn：window → n_wins窗口各64；full → seq_len_eff
    ...
    
    # Step 4/6/7：rmsnorm/ffn 用 seq_len_eff
    ...
    
    # 清理时删除 n_wins 个窗口文件（不再硬编码 NUM_WINS）
    for w in range(n_wins):
        ...
```

**④ `_rms_inv` 计算**：接受动态 seq_len 参数（不硬编码 SEQ_LEN）

**A3. 处理 full attention 块 seq=4096 的 zkSoftmax 参数**

Full attention 块 seq=4096 时，`self-attn.cu` 的 attn 模式已有 K=4 fallback：
```cpp
if (seq_sq == (1U << 20)) {          // seq=1024
    softmax_bs = {1U<<8, 1U<<20, 1U<<20};
    ...
} else {                              // seq=64 window → 已有 K=4
    softmax_bs = {256U, seq_sq, seq_sq, seq_sq};
    ...
}
```
seq=4096 → seq_sq = 2^24，上述 else 分支自动处理（K=4，range=256×(2^24)³=2^80 >> 激活值范围）。**无需修改 C++ binary**。

---

### Step B：Constraint Fusion（优化2）

**目标**：合并相邻两次 Rescaling 的 range check，减少 tLookup 调用次数 ~50%。

#### B1. 理论基础

当前 `self-attn.cu` attn 模式（lines 185-189）：
```cpp
auto out_h_  = rs2(out_h);     // rs2.sf = 1<<16
auto out_h__ = rs1(out_h_);    // rs1.sf = 1<<16
rs1.prove(out_h_, out_h__);    // tLookup range check #1
rs2.prove(out_h,  out_h_);     // tLookup range check #2
```

融合后：`out_h__ = floor(out_h / (sf²))`，remainder `r = out_h mod (sf²)` 分解为两个 sf-进制数字 `(r₁, r₀)`：
- `r₁ = floor(r / sf)` ∈ [0, sf)
- `r₀ = r mod sf`      ∈ [0, sf)
- **一次 tLookup** 对 `(r₁, r₀)` 做二维范围校验，代替两次独立 tLookup

#### B2. `rescaling.cuh` 新增声明

```cpp
class Rescaling {
public:
    // 现有接口（保留）
    FrTensor operator()(const FrTensor& X);
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_);
    
    // 新增：串联两次 rescaling 的融合证明
    // 证明 X →[/sf]→ X_ →[/sf]→ X__，合并为一次 tLookup
    static vector<Claim> prove_chain(
        const FrTensor& X, const FrTensor& X_, const FrTensor& X__,
        uint sf);
};
```

#### B3. `rescaling.cu` 实现 `prove_chain`

核心思路：
1. 计算 combined remainder：`R = X - X__ * sf²`（field arithmetic，GPU kernel）
2. 分解：`R1 = R / sf`（整除），`R0 = R mod sf`（via 现有 decomp 逻辑）
3. 对 `R1` 做 tLookup range [0, sf)，对 `R0` 做 tLookup range [0, sf)
4. 但这只是把两次 tLookup 合并验证，关键是**共享一次 multi_hadamard_sumchecks**
5. 实现：`tLookupRange(sf*sf)` 但用 digit decomposition 分两路，借用现有 `tlookup.cuh` 的多基表接口

**预期收益**：
- self-attn attn 模式：2 次 tLookup → 1 次，tLookup 开销 -50%
- ffn 中有类似 down_rescale 调用，同样适用
- 总体 tLookup 减少约 30-50%，对应 prover 时间缩短约 1.5×

#### B4. 修改调用处

`src/zkllm/self-attn.cu` attn 模式：
```cpp
// 替换 lines 185-189
auto out_h_  = rs2(out_h);
auto out_h__ = rs1(out_h_);
Rescaling::prove_chain(out_h, out_h_, out_h__, rs1.scaling_factor);
```

`src/zkllm/ffn.cu`（类似位置）同样替换。

---

### Step C：Circuit Squeeze（优化3）——不同输入批量 Sumcheck

**目标**：把 FFN 中 gate_proj、up_proj（共享 input）和 down_proj（input=SiLU(gate)·up）的三个矩阵乘合并为更少的 sumcheck，参照 zkGPT Appendix D Algorithm 1。

#### C1. 理论基础（zkGPT Equation 8）

对 M 个矩阵乘 Y_i = A_i × B（不同 A_i，同一 B），用随机挑战 α_i 组合：
```
α₁Ỹ₁(r₁) + α₂Ỹ₂(r₂) = Σ_c [α₁Ã₁(r₁||c)·B̃(r||c) + α₂Ã₂(r₂||c)·B̃(r||c)]
```
即在 c 维度上做联合 sumcheck，B 的贡献被 α 加权合并。

**不同输入（不同 B）的情况**（gate/up vs down_proj）：用 zkGPT Equation 8 更一般形式，需要拼接 A 矩阵（bookkeeping table 方式）。

#### C2. `zkfc.cuh` 新增声明

```cpp
// 现有 prove_batch（已实现：同 input，批量 q/k/v）
static vector<Claim> prove_batch(
    const FrTensor& X,
    const vector<pair<zkFC*, FrTensor*>>& layers_and_outputs);

// 新增：不同 input 的批量证明（Circuit Squeeze, Algorithm 1）
struct LayerIO { zkFC* layer; const FrTensor* input; const FrTensor* output; };
static vector<Claim> prove_batch_diff_x(
    const vector<LayerIO>& ios,
    vector<Polynomial>& proof);
```

#### C3. `zkfc.cu` 实现 `prove_batch_diff_x`

参照 zkGPT Algorithm 1 的 bookkeeping table 方式：

```
步骤：
1. 为每个 layer_i 生成随机挑战 α_i（Fiat-Shamir）
2. 对每个 Y_i 生成独立的 u_output_i = random_vec(ceilLog2(rows×cols))
3. 将各 Y_i 的 multi_dim_me 结果组合为 claim_combined = Σ α_i × Y_i(u_output_i)
4. 用 bookkeeping table（EQ_x[i], EQ_y 数组，Algorithm 1）把多路求和化为一个 sumcheck
5. 对每个 W_i 独立做 IPA open（输入点共享部分 → 节省 partial_me 计算）
```

**预期收益**（FFN gate+up+down 三层合并）：
- 原来：3 次独立 sumcheck + 3 次 IPA 
- 合并后：1 次联合 sumcheck + 3 次 IPA
- sumcheck 耗时减少 ~2/3，对 FFN prover 时间约 1.7× 加速

#### C4. 修改 `ffn.cu`

```cpp
// 替换 lines 91-97（当前：3次独立 prove + verifyWeightClaim）
auto layer_ios = vector<zkFC::LayerIO>{
    {&down_layer, &down_in_, &down_out},
    {&gate_layer, &input,    &gate_out},
    {&up_layer,   &input,    &up_out},
};
vector<Polynomial> batch_proof;
auto claims = zkFC::prove_batch_diff_x(layer_ios, batch_proof);
verifyWeightClaim(down_proj, claims[0], workdir+"/"+prefix+"-mlp.down_proj-ipa-proof.bin");
verifyWeightClaim(gate_proj, claims[1], workdir+"/"+prefix+"-mlp.gate_proj-ipa-proof.bin");
verifyWeightClaim(up_proj,   claims[2], workdir+"/"+prefix+"-mlp.up_proj-ipa-proof.bin");
```

---

### Step D：2-GPU 并行扩展

#### D1. 已完成（Phase 5）
- `verify_vit.py`：`ProcessPoolExecutor(max_workers=2)`，偶数块→GPU0，奇数块→GPU1
- `verify_layers.py`：同样的 2-GPU 模式

#### D2. 需要扩展（Phase 6）
- `build_corpus_full_proof.py`：当前 ViT 32 块串行调用 `verify_vit_block`
  → 改为在 `prove_full_image` 内用 `ProcessPoolExecutor` 并行跑 ViT 32 块
- 注意：ViT 每块使用独立 `vit-b{N}` 子目录，无文件冲突，安全并行
- 参数分配：`blocks_gpu0 = [0,2,4,...]`, `blocks_gpu1 = [1,3,5,...]`

---

### Step E：真实数据 Smoke Test

**前提**：A、B、C、D 均完成后，对真实 nikon 文档页运行端到端 smoke test。

**流程**：
```bash
# 1. 单页 smoke test
python script/build_corpus_full_proof.py \
    --limit 1 --overwrite \
    2>&1 | tee logs/phase6_smoke_real_data.log

# 2. 检查输出
cat zkllm-workdir/jina-v4/corpus_full_proof_*.json | python -m json.tool | head -50

# 3. ViT soundness 验证（第一页，block 0 和 7）
python script/verify_vit.py --blocks 0 7 --workdir zkllm-workdir/jina-v4

# 4. 完整 ViT 32 块（单页，真实 2520 patches）
python script/verify_vit.py --blocks $(seq 0 31) --parallel

# 5. LLM 36 层（真实激活）
python script/verify_layers.py --layers $(seq 0 35) --parallel
```

**验收标准**：
- ViT 32/32 pass（含 window blocks 正确 ceil 窗口数）
- LLM 36/36 pass
- 无 "截断" 相关 warning
- `notes/experiment_results/` 记录真实耗时

---

## 预期优化效果对比

| 配置 | ViT 32块 | LLM 36层 | 全量/页 |
|------|---------|---------|--------|
| Phase 5 基准（seq=1024，截断） | 3 min | 5 min | 8 min |
| +Step A（seq=4096 full，真实数据） | 5 min | 5 min | 10 min |
| +Step B（Constraint Fusion） | 3.5 min | 3.5 min | 7 min |
| +Step C（Circuit Squeeze） | 2 min | 2 min | 4 min |
| +Step D（2-GPU 全管道并行） | 1 min | 2 min | 3 min |
| **目标：Phase 6 完成** | **~1 min** | **~2 min** | **~3 min/页** |

**在线查询验证（文字，seq=64）**：
| 配置 | 36层估计 |
|------|---------|
| Phase 5 基准 | ~30-60s |
| +B+C 优化 | **~15-25s** |

---

## 实施优先级与依赖关系

```
Step A（soundness 修复）
  └─ 前提：无特殊依赖，优先实施
  └─ 影响：Step E（真实数据测试依赖此步骤）

Step B（Constraint Fusion）
  └─ 前提：熟悉 rescaling.cuh/cu，工作量中等
  └─ 影响：Step C 可独立并行开发

Step C（Circuit Squeeze）
  └─ 前提：理解 zkGPT Algorithm 1，工作量较大
  └─ 不依赖 Step B

Step D（2-GPU 扩展）
  └─ 前提：Step A 完成（需要动态 seq_len 下的正确并行）
  └─ 改动小，优先级高

Step E（真实数据测试）
  └─ 前提：Step A + D 完成
  └─ B、C 可后补（优化不影响正确性验证）
```

**建议执行顺序**：A → D → E（smoke test）→ B → C（优化）→ E（完整测试）

---

## 关键文件清单

| 文件 | 操作 | 影响步骤 |
|------|------|---------|
| `script/build_corpus_full_proof.py` | 修改：`_save_full`，移除 `SEQ_VIT` 截断 | A |
| `script/verify_vit.py` | 修改：动态 `seq_len`、`n_wins`，`_get_seq_params` | A |
| `src/zkllm/rescaling.cuh` | 新增：`prove_chain` 声明 | B |
| `src/zkllm/rescaling.cu` | 实现：`prove_chain`（digit decomp + 联合 tLookup） | B |
| `src/zkllm/self-attn.cu` | 修改：调用 `prove_chain` 替代两次 `prove` | B |
| `src/zkllm/ffn.cu` | 修改：同上 + 调用 `prove_batch_diff_x` | B, C |
| `src/zkllm/zkfc.cuh` | 新增：`prove_batch_diff_x` 声明 + `LayerIO` struct | C |
| `src/zkllm/zkfc.cu` | 实现：`prove_batch_diff_x`（bookkeeping table） | C |
| `src/zkllm/Makefile` | 确认所有修改文件在编译目标中 | B, C |
| `script/build_corpus_full_proof.py` | 修改：ViT 32块改用 ProcessPoolExecutor | D |
| `notes/implementation_log.md` | 随时追加：各步骤耗时、通过率、错误记录 | 全程 |

---

## 参考文献

- Qu et al. "zkGPT: An Efficient Non-interactive Zero-knowledge Proof Framework for LLM Inference." EuroSys 2024.
  - Table 5：constraint fusion 1.5×，circuit squeeze 1.7×，合计 14.7× vs 单线程无优化
  - Table 7：各层 rounding 数量 before/after fusion
  - Appendix D Algorithm 1：bookkeeping table 批量矩阵乘
- Sun et al. "zkLLM: Zero Knowledge Proofs for Large Language Models." 2024.
  - 原始系统设计，tLookup、Rescaling、zkSoftmax 实现

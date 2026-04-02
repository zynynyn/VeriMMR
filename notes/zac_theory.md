# ZAC 原理详解：从 Merkle Tree 到 O(1) 证明

---

## 第一部分：Merkle Tree 为什么是 O(log N)

### 问题背景

你有一个语料库，100 张图片。有人问你："第 37 张图真的在你的语料库里吗？" 你怎么证明？

最笨的办法：把全部 100 张图发过去，让对方自己数。但这太贵了——你希望证明尽量短。

### Merkle Tree 的做法

```
         Root (32B)
        /           \
    H(0-3)          H(4-7)
    /    \           /    \
 H(0-1) H(2-3)   H(4-5) H(6-7)
  / \    / \      / \    / \
L0  L1  L2  L3   L4  L5  L6  L7

每个叶子：    Li = SHA256(image_i)
每个内部节点：H  = SHA256(左孩子 || 右孩子)
```

**证明"L2 在树里"的过程：**

把这条路径上的"兄弟节点"发出去：

```
证明 = [ L3, H(0-1), H(4-7) ]  <- 3 个哈希
```

验证方自己算：

```
SHA256(L2 || L3)         = H(2-3)
SHA256(H(0-1) || H(2-3)) = H(0-3)
SHA256(H(0-3) || H(4-7)) = Root ?
```

**结论**：N 个叶子，树高 = log2(N)，证明 = log2(N) 个哈希值。

- N=1,000 时，证明包含约 10 个哈希（320 字节）
- N=1,000,000 时，证明包含约 20 个哈希（640 字节）

**这就是 O(log N)。**

---

## 第二部分：ZAC 为什么是 O(1)

### 第一步：Bloom Filter（把集合变成向量）

先解决"任意集合"的问题。Merkle tree 要求元素有固定顺序和位置，Bloom Filter 更灵活。

```python
# 代码里的 BloomFilter.gen(S):
def gen(self, S):
    v = [0] * self.q           # 长度 q 的全零向量
    for elem in S:
        for seed in range(self.k):
            pos = mmh3.hash(elem, seed=seed) % self.q
            v[pos] = 1         # 把对应位置设为 1
    return v
```

举例，S = {"img_a", "img_b", "img_c"}，k=3，q=10：

```
img_a 的哈希位置：[1, 5, 7]
img_b 的哈希位置：[2, 5, 9]
img_c 的哈希位置：[0, 3, 7]

v = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
```

**查成员**："img_a 在集合里吗？" 检查位置 1,5,7 是否全为 1，是，则很可能在（有小概率假阳性 epsilon）。

关键性质：**不管集合有多大，BF 向量 v 的长度 q 是固定的**（由误判率参数 epsilon 和集合上界 N 决定）。

最优参数公式（论文公式 2）：

```
q = ceil( -N * ln(epsilon) / ln(2)^2 )   (位向量长度)
k = ceil( (q/N) * ln(2) )                (哈希函数个数)
```

例：N=50，epsilon=0.01，则 q=479，k=7。

### 第二步：Pointproofs（把向量承诺压成一个椭圆曲线点）

这是核心魔法，基于**椭圆曲线配对密码学（BLS12-381）**。

**直觉：多项式承诺**

选椭圆曲线生成元 g，秘密选数 alpha，公开 CRS：

```
CRS = ( g^alpha, g^(alpha^2), g^(alpha^3), ..., g^(alpha^q) )
```

生成后 alpha 必须销毁，类似 KZG trusted setup。

对向量 v = (v1, v2, ..., v_{q-1}) 的承诺：

```
C = g^( v1*alpha^1 + v2*alpha^2 + ... + v_{q-1}*alpha^{q-1} + r*alpha^q )
```

这是**一个椭圆曲线点**，无论 v 有多长，C 的大小固定 = **48 字节**（BLS12-381 G1 点压缩大小）。

代码里的实现：

```python
# Pointproofs.commit(v, r):
def commit(self, m, r):
    C = multiply(self._P[q-1], r)           # g1^(r*alpha^q)
    for i, val in enumerate(m[:q-1]):
        if val != 0:
            C = add(C, multiply(self._P[i], val))  # += g1^(vi * alpha^{i+1})
    return C
# 结果：C 是一个 G1 点，48 字节，与向量长度无关
```

### 第三步：Pointproofs 成员证明（单个位置）

**证明"位置 i 的值是 vi"：**

```
pi_i = g^( sum_{j != i} v'_j * alpha^{q+1-i+j} )
```

其中 v' = (v1, ..., v_{q-1}, r)（把随机数 r 追加进去）。

这也是**一个 G1 点，48 字节**。不管 q 多大，证明永远是一个点。

### 第四步：聚合（多个位置合并成一个证明）

证明 Bloom Filter 中 k 个位置全为 1，不需要发 k 个证明，而是**聚合成一个**：

```python
# Pointproofs.aggregate:
def aggregate(self, C, I, v_I, proofs):
    agg = Z1                                  # G1 的零元（无穷远点）
    for pos, pi in zip(I, proofs):
        t = self._challenge(pos, C, I, v_I)   # Fiat-Shamir 随机挑战 ti
        agg = add(agg, multiply(pi, t))        # agg += pi_i ^ ti
    return agg
# 结果：不管聚合了多少个证明，agg 还是一个 G1 点，48 字节
```

**这就是 O(1) 的来源**：无论集合有多大、BF 涉及多少个位置，最终证明永远是一个 G1 点。

### 第五步：配对验证

验证方持有公开承诺 cm 和证明 pi_hat，通过**双线性配对**验证：

```
e( cm,  sum_i ti * g2^{alpha^{q+1-i}} )
    =?=
e( pi_hat, g2 ) * gT^{ alpha^{q+1} * sum_i vi*ti }
```

`e: G2 x G1 -> GT` 是双线性映射，满足 `e(aP, bQ) = e(P,Q)^{ab}`。

代码里的实现：

```python
# Pointproofs.verify:
def verify(self, C, I, v_I, pi_hat):
    t_vals = [self._challenge(pos, C, I, v_I) for pos in I]

    # LHS：sum ti * g2^{alpha^{q-pos}}，然后与 cm 配对
    V_sum = Z2
    for pos, t in zip(I, t_vals):
        exp = self.q - pos
        V_sum = add(V_sum, multiply(self._V[exp-1], t))
    lhs = pairing(V_sum, C)          # e(G2_part, cm)

    # RHS：e(g2, pi_hat) * gT^{sum vi*ti}
    exp_sum = sum(v * t for v, t in zip(v_I, t_vals)) % curve_order
    rhs = pairing(G2, pi_hat) * (self._gT_aq1 ** exp_sum)

    return lhs == rhs
```

**验证复杂度**：2 次配对 + 若干点乘，**与原始集合 N 无关** → O(1)。

---

## 第三部分：性能警告从哪来

### CRS 生成是瓶颈

```python
# Pointproofs._setup(alpha):
self._P  = [multiply(G1, a[i]) for i in range(1, q+1)]    # q 次 G1 点乘
self._Px = [multiply(G1, a[q+2+j]) for j in range(q-1)]   # q-1 次 G1 点乘
self._V  = [multiply(G2, a[i]) for i in range(1, q+1)]    # q 次 G2 点乘
```

**每次 `multiply(G1, scalar)` 在 py_ecc 纯 Python 里需要约 0.01 秒。**

以真实语料库参数为例：

```
N=50 张图  -> q=479
CRS 总次数 = 479 + 478 + 479 = 1436 次点乘
耗时 ≈ 1436 * 0.01s ≈ 14 秒

N=200 张图（论文实验规模）-> q≈1918
耗时 ≈ 5754 * 0.01s ≈ 60 秒
```

### 为什么 py_ecc 慢

py_ecc 是纯 Python 实现，每次椭圆曲线点乘需要数百次 256-bit 大整数乘法，没有任何硬件加速。

论文作者使用的是 Rust 实现的 Pointproofs 库（github.com/algorand/pointproofs），同等操作只需毫秒级。

### 性能对比

| 操作 | py_ecc (Python) | Rust Pointproofs |
|------|----------------|-----------------|
| CRS 生成（q~500） | ~15s | ~0.1s |
| 单次 Commit | ~0.5s | ~5ms |
| 单次证明生成 | ~1s | ~10ms |
| 单次验证（含 2 次配对） | ~2s | ~20ms |

对毕业设计答辩的意义：py_ecc 足以展示正确性和协议流程，答辩时说"生产环境替换为 Rust Pointproofs 库性能提升 100x 以上"即可。

---

## 第四部分：项目中的具体实现

### 文件结构

```
src/zac/
├── __init__.py           <- 导出 ZACAccumulator, BloomFilter, Pointproofs
└── accumulator.py        <- 全部实现（约 480 行）

script/
└── phase1_corpus_fingerprint.py  <- 自动化驱动脚本
```

### 三层架构

```
+-----------------------------------------------------+
|  ZACAccumulator  （对外接口层）                      |
|  · from_corpus(jsonl, npy) -- 从语料库构建           |
|  · root_hex() -> 96字符 = 48字节指纹                 |
|  · prove_membership(hash) -> proof dict             |
|  · verify_membership(hash, proof) -> bool           |
|  · save(path) -> JSON 指纹清单                       |
+-----------------------------------------------------+
|  Pointproofs  （向量承诺层，BLS12-381）              |
|  · _setup(alpha) -- 生成 CRS（P, Px, V, gT）        |
|  · commit(v, r) -> G1 点（48B）                     |
|  · prove(i, v, r) -> G1 点（48B）                   |
|  · aggregate(C, I, vI, pis) -> G1 点（48B，恒定）   |
|  · verify(C, I, vI, pi_hat) -> bool（配对方程）      |
+-----------------------------------------------------+
|  BloomFilter  （集合 -> 向量转换层）                 |
|  · optimal_params(N, epsilon) -- 计算最优 q, k      |
|  · gen(S) -> v in {0,1}^q                          |
|  · check(v, elem) -> bool                          |
|  · membership_indices(S_hat) -> BF 位置列表         |
+-----------------------------------------------------+
```

### 信任链（论文核心逻辑）

```
图片文件                    SHA256                   ZAC
image/doc/page_0.jpg  ----------->  hash0  -> |
image/doc/page_1.jpg  ----------->  hash1  -> S = {hash0, hash1, ...}
...                                 ...        |
                                               v BF.Gen(S)
                                             向量 v in {0,1}^q
                                               v PR.Commit(v, r)
                                             cm in G1  <- 48字节"数字指纹"
                                             公开存证

检索时：
  取回 image_j -> SHA256 -> hj
    -> ZAC.ProveM：计算 BF 位置 I，逐位置出证明，聚合 -> 48字节 pi
    -> ZAC.VerifyM：任何人用公开 cm 跑配对验证
    -> 确认"这张图真的在原始语料库里"
```

### Merkle vs ZAC 对比总结

| 维度 | Merkle Tree | ZAC（本项目）|
|------|------------|-------------|
| 证明大小 | O(log N) x 32B | **48B 恒定** |
| 验证复杂度 | O(log N) 次哈希 | **O(1)，2 次配对** |
| 非成员证明 | 需额外结构 | Bloom Filter 天然支持 |
| 零知识性 | 不隐藏集合大小 | **隐藏集合内容和大小** |
| 动态更新 | 重建路径，O(log N) | UpdCom，O(q) 乘法 |
| 实现复杂度 | 简单 | 需要双线性配对 |
| 数学基础 | SHA256 哈希 | BLS12-381 椭圆曲线 |

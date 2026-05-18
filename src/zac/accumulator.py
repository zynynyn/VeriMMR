"""
ZAC: Zero-knowledge Dynamic Universal Accumulator
==================================================
Python implementation of ZAC (Dang et al., TPS-ISA 2022) for UltraRAG corpus
verification.

Design
------
  S  = { SHA256(image_bytes_i) : image_i in corpus }   — committed set
  v  = BF.Gen(S) ∈ {0,1}^q                             — Bloom Filter vector
  cm = PR.Commit(v, r) ∈ G1                             — 48-byte ZAC Root

Membership proof for image_j:
  I  = ∪{ h_l(SHA256(image_j)) : l=1..k }              — BF positions
  π̂  = PR.Aggregate( PR.Prove(i, v, r) for i∈I )       — 48-byte proof

Verification (public, offline):
  e(cm, g₂^{Σ tᵢ·α^{q+1-i}}) = e(π̂, g₂) · gT^{α^{q+1}·Σ vᵢ·tᵢ}

Both commitment and proof are a single BLS12-381 G1 point = 48 bytes.
Proof size is constant, independent of corpus size.

Reference
---------
  Dang et al. "ZAC: Efficient Zero-Knowledge Dynamic Universal Accumulator
  and Application to Zero-Knowledge Elementary Database." TPS-ISA 2022.
  Underlying VC scheme: Gorbunov et al. "Pointproofs." CCS 2020.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mmh3

from pyblst import BlstP1Element, BlstP2Element, BlstFP12Element, miller_loop, final_verify


# ---------------------------------------------------------------------------
# BLS12-381 constants and pyblst helpers
# ---------------------------------------------------------------------------

curve_order = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

# BLS12-381 generator points (standard compressed encoding)
_G1_BYTES = bytes.fromhex(
    "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac5"
    "86c55e83ff97a1aeffb3af00adb22c6bb"
)
_G2_BYTES = bytes.fromhex(
    "93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049"
    "334cf11213945d57e5ac7d055d042b7e"
    "024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d17"
    "70bac0326a805bbefd48056c8c121bdb8"
)
_G1: BlstP1Element = BlstP1Element.uncompress(_G1_BYTES)
_G2: BlstP2Element = BlstP2Element.uncompress(_G2_BYTES)


def _g1_mult(pt: BlstP1Element, scalar: int) -> BlstP1Element:
    return pt.scalar_mul(int(scalar) % curve_order)


def _g1_add(pt1: BlstP1Element, pt2: BlstP1Element) -> BlstP1Element:
    return pt1 + pt2


def _g2_mult(pt: BlstP2Element, scalar: int) -> BlstP2Element:
    return pt.scalar_mul(int(scalar) % curve_order)


def _g2_add(pt1: BlstP2Element, pt2: BlstP2Element) -> BlstP2Element:
    return pt1 + pt2


def _p1_is_inf(pt: BlstP1Element) -> bool:
    """True iff pt is the G1 point at infinity."""
    return pt.compress()[0] == 0xC0


def _p2_is_inf(pt: BlstP2Element) -> bool:
    """True iff pt is the G2 point at infinity."""
    return pt.compress()[0] == 0xC0


# ---------------------------------------------------------------------------
# Scalar utilities
# ---------------------------------------------------------------------------

def _random_zp(seed: Optional[bytes] = None) -> int:
    """Uniformly random scalar in Z_p."""
    raw = hashlib.sha256(seed).digest() if seed is not None else os.urandom(32)
    return int.from_bytes(raw, "big") % curve_order


def _hash_to_zp(data: bytes) -> int:
    """Deterministic hash of bytes → scalar in Z_p (used as Fiat-Shamir challenge)."""
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % curve_order


# ---------------------------------------------------------------------------
# G1 compression / decompression  (BLS12-381 standard 48-byte encoding)
# ---------------------------------------------------------------------------

def _g1_compress(pt: BlstP1Element) -> bytes:
    """Compress a G1 point to 48 bytes (standard BLS12-381 serialization)."""
    return pt.compress()


def _g1_decompress(data: bytes) -> "BlstP1Element | None":
    """Decompress 48-byte BLS12-381 G1 point. Returns None on failure."""
    if len(data) != 48:
        return None
    try:
        return BlstP1Element.uncompress(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bloom Filter  (BF)
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Standard Bloom Filter. Maps an arbitrary set S ⊆ {0,1}* to a binary
    vector v ∈ {0,1}^q using k independent hash functions.

    Optimal parameters for set bound N and false-positive rate ε:
      q = ⌈-N·ln ε / ln²2⌉   (bit-vector length)
      k = ⌈(q/N)·ln 2⌉        (number of hash functions)

    Implemented with MurmurHash3 (mmh3) with seeds seed_offset, ..., seed_offset+k-1.
    Different seed_offset values make BF layers independent (used in cascade mode).
    """

    def __init__(self, q: int, k: int, seed_offset: int = 0) -> None:
        self.q = q
        self.k = k
        self._seed_offset = seed_offset

    @classmethod
    def optimal_params(cls, N: int, epsilon: float = 0.01,
                       seed_offset: int = 0) -> "BloomFilter":
        q = max(1, math.ceil(-N * math.log(epsilon) / (math.log(2) ** 2)))
        k = max(1, round((q / N) * math.log(2)))
        return cls(q=q, k=k, seed_offset=seed_offset)

    def _positions(self, element: bytes) -> List[int]:
        return [mmh3.hash(element, seed=self._seed_offset + i, signed=False)
                % self.q for i in range(self.k)]

    def gen(self, S: Set[bytes]) -> List[int]:
        """BF.Gen(S) → binary vector v of length q."""
        v = [0] * self.q
        for elem in S:
            for pos in self._positions(elem):
                v[pos] = 1
        return v

    def check(self, v: List[int], element: bytes) -> bool:
        return all(v[pos] == 1 for pos in self._positions(element))

    def membership_indices(self, S_hat: Set[bytes]) -> List[int]:
        """I = ∪_{ŝ∈Ŝ} {h₁(ŝ),...,hₖ(ŝ)}, sorted."""
        idx: Set[int] = set()
        for elem in S_hat:
            for pos in self._positions(elem):
                idx.add(pos)
        return sorted(idx)


# ---------------------------------------------------------------------------
# Pointproofs  (PR) — Vector Commitment over BLS12-381
# ---------------------------------------------------------------------------

class Pointproofs:
    """
    Pointproofs aggregatable vector commitment (Gorbunov et al., CCS 2020).
    Instantiated over BLS12-381.  Used as zkAVC inside ZAC.

    Notation (1-indexed throughout, matching the paper):
      q   : vector length  (= BF bit-array size)
      α   : trapdoor scalar in Z_p  (secret, discard after Setup)
      CRS : P  = [g1^α, ..., g1^αq]            — G1 powers
            Px = [g1^{α^{q+2}}, ..., g1^{α^{2q}}] — cross-term G1 powers
            V  = [g2^α, ..., g2^αq]            — G2 powers (for verification)

    Commit(m, r):
      C = g1^{r·α^q} · ∏_{i=1}^{q-1} g1^{mᵢ·αⁱ}    (G1 point)

    Prove(i, m, r):
      πᵢ = ∏_{j≠i, j=1..q} g1^{m'ⱼ · α^{q+1-i+j}}  (G1 point)
      where m' = (m₁,...,m_{q-1}, r)

    Aggregate(C, I, m[I], {πᵢ}):
      tᵢ = H'(i, C, I, m[I]),  π̂ = ∏ πᵢ^{tᵢ}        (G1 point)

    Verify(C, I, m[I], π̂):
      e(C, ∑ᵢ tᵢ·g2^{α^{q+1-i}}) ?= e(π̂, g2) · gT^{α^{q+1}·∑ᵢ mᵢ·tᵢ}
      Implemented as 2-pairing check via bilinearity:
        gT^{α^{q+1}·exp_sum} = e(exp_sum·P[0], V[q-1])
      So: final_verify( miller_loop(C, V_sum),
                        miller_loop(π̂, G2) * miller_loop(exp_sum·P[0], V[q-1]) )
    """

    def __init__(self, q: int, alpha: Optional[int] = None) -> None:
        self.q = q
        self._setup(alpha if alpha is not None else _random_zp())

    def _setup(self, alpha: int) -> None:
        self._alpha = alpha          # kept for prover-state serialization
        p = curve_order

        # Precompute α^i mod p for i = 1 .. 2q
        a = [0] * (2 * self.q + 2)
        a[1] = alpha % p
        for i in range(2, 2 * self.q + 2):
            a[i] = (a[i - 1] * alpha) % p

        # P[i-1] = g1^{α^i},  i = 1..q
        self._P: List[BlstP1Element] = [_g1_mult(_G1, a[i]) for i in range(1, self.q + 1)]

        # Px[j] = g1^{α^{q+2+j}},  j = 0..q-2  (i.e. α^{q+2}..α^{2q})
        self._Px: List[BlstP1Element] = [_g1_mult(_G1, a[self.q + 2 + j]) for j in range(self.q - 1)]

        # V[i-1] = g2^{α^i},  i = 1..q
        self._V: List[BlstP2Element] = [_g2_mult(_G2, a[i]) for i in range(1, self.q + 1)]

    # -----------------------------------------------------------------------
    # CRS serialization  (skip expensive _setup on reload)
    # -----------------------------------------------------------------------

    def save_crs(self, path: str) -> None:
        """
        Serialize the precomputed CRS to a binary file (compressed coordinates).
        Format (big-endian):
          [4B]      q
          [q × 48B]   P   (G1 compressed)
          [q-1 × 48B] Px  (G1 compressed)
          [q × 96B]   V   (G2 compressed)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.q.to_bytes(4, "big"))
            for pt in self._P:
                f.write(pt.compress())      # 48B each
            for pt in self._Px:
                f.write(pt.compress())      # 48B each
            for pt in self._V:
                f.write(pt.compress())      # 96B each

    def _setup_from_cache(self, path: str) -> None:
        """Load precomputed CRS from compressed-format binary file."""
        with open(path, "rb") as f:
            q = int.from_bytes(f.read(4), "big")
            assert q == self.q, f"CRS cache mismatch: expected q={self.q}, got {q}"
            self._P  = [BlstP1Element.uncompress(f.read(48)) for _ in range(q)]
            self._Px = [BlstP1Element.uncompress(f.read(48)) for _ in range(q - 1)]
            self._V  = [BlstP2Element.uncompress(f.read(96)) for _ in range(q)]

    # -----------------------------------------------------------------------
    # CRS accessors
    # -----------------------------------------------------------------------

    def _g1_alpha_i(self, i: int) -> BlstP1Element:
        """g1^{α^i} for i ∈ [1,q] ∪ [q+2, 2q].  α^{q+1} not in CRS."""
        if 1 <= i <= self.q:
            return self._P[i - 1]
        if self.q + 2 <= i <= 2 * self.q:
            return self._Px[i - (self.q + 2)]
        raise ValueError(f"α^{i} not available in CRS (q={self.q}).")

    def _g2_alpha_i(self, i: int) -> BlstP2Element:
        """g2^{α^i} for i ∈ [1, q]."""
        if 1 <= i <= self.q:
            return self._V[i - 1]
        raise ValueError(f"g2^α^{i} not available in CRS (q={self.q}).")

    # -----------------------------------------------------------------------
    # Protocol
    # -----------------------------------------------------------------------

    def commit(self, m: List[int], r: int) -> BlstP1Element:
        """
        C = g1^{r·α^q} · ∏_{i=1}^{q-1} (g1^{α^i})^{mᵢ}
        m is 0-indexed, length q-1; internally 1-indexed.
        """
        C = _g1_mult(self._P[self.q - 1], r % curve_order)  # g1^{r·α^q}
        for i1 in range(1, self.q):                           # i = 1..q-1
            mi = m[i1 - 1]
            if mi == 0:
                continue
            # BF fast path: m ∈ {0,1}, so scalar_mul(1) = identity; use add directly
            base = self._P[i1 - 1]
            C = _g1_add(C, base if mi == 1 else _g1_mult(base, mi))
        return C

    def prove(self, pos: int, m: List[int], r: int) -> BlstP1Element:
        """
        πᵢ = ∏_{j∈[q], j≠i} g1^{m'ⱼ · α^{q+1-i+j}}
        pos is 0-indexed.  Internally: i = pos+1 (1-indexed).
        m'  = (m[0], ..., m[q-2], r)  — length q.
        """
        i = pos + 1                           # 1-indexed position
        m_prime = list(m[: self.q - 1]) + [r]  # length q

        pi: Optional[BlstP1Element] = None
        for j in range(1, self.q + 1):        # j = 1..q  (1-indexed)
            if j == i:
                continue
            exp = self.q + 1 - i + j          # α^{q+1-i+j}
            if exp == self.q + 1:             # not in CRS
                continue
            mj = m_prime[j - 1]
            if mj == 0:
                continue
            # BF fast path: m ∈ {0,1}, skip scalar_mul for mj==1
            base = self._g1_alpha_i(exp)
            term = base if mj == 1 else _g1_mult(base, mj)
            pi = term if pi is None else _g1_add(pi, term)
        # Identity fallback (all-zero m_prime edge case)
        if pi is None:
            pi = _g1_mult(_G1, 0)
        return pi

    def _challenge(self, pos: int, C: BlstP1Element, I: List[int], v_I: List[int]) -> int:
        """tᵢ = H'(i, C, I, v[I]) — Fiat-Shamir challenge."""
        data = (
            pos.to_bytes(4, "big")
            + _g1_compress(C)
            + b"".join(p.to_bytes(4, "big") for p in I)
            + bytes(v_I)
        )
        return _hash_to_zp(data)

    def aggregate(self, C: BlstP1Element, I: List[int], v_I: List[int],
                  proofs: List[BlstP1Element]) -> BlstP1Element:
        """
        π̂ = ∏ᵢ∈I πᵢ^{tᵢ}   (aggregated proof, single G1 point)
        """
        agg: Optional[BlstP1Element] = None
        for pos, pi in zip(I, proofs):
            t = self._challenge(pos, C, I, v_I)
            term = _g1_mult(pi, t)
            agg = term if agg is None else _g1_add(agg, term)
        if agg is None:
            agg = _g1_mult(_G1, 0)
        return agg

    def verify(self, C: BlstP1Element, I: List[int], v_I: List[int],
               pi_hat: BlstP1Element) -> bool:
        """
        Check: e(C, V_sum) == e(π̂, G2) · e(exp_sum·P[0], V[q-1])
        where V_sum = Σᵢ tᵢ · g2^{α^{q+1-i}},  exp_sum = Σᵢ vᵢ·tᵢ mod p.

        Implemented via bilinearity to avoid GT exponentiation:
          gT^{α^{q+1}·exp_sum} = e(exp_sum·P[0], V[q-1])
        Single final_verify(lhs_ml, rhs_ml) call (one final_exp for both sides).
        """
        t_vals = [self._challenge(pos, C, I, v_I) for pos in I]

        # V_sum = Σᵢ tᵢ · g2^{α^{q+1-i1}}   where i1 = pos+1 (1-indexed)
        V_sum: Optional[BlstP2Element] = None
        for pos, t in zip(I, t_vals):
            i1  = pos + 1
            exp = self.q + 1 - i1           # = q - pos  (≥1 when pos ≤ q-1)
            if 1 <= exp <= self.q:
                term = _g2_mult(self._g2_alpha_i(exp), t)
                V_sum = term if V_sum is None else _g2_add(V_sum, term)
        if V_sum is None or _p2_is_inf(V_sum):
            return False

        exp_sum = sum(v * t for v, t in zip(v_I, t_vals)) % curve_order
        p0_scaled = _g1_mult(self._P[0], exp_sum)   # exp_sum · P[0]

        # e(C, V_sum) == e(π̂, G2) · e(exp_sum·P[0], V[q-1])
        lhs_ml = miller_loop(C, V_sum)                                         # G1=C, G2=V_sum
        rhs_ml = (miller_loop(pi_hat, _G2)
                  * miller_loop(p0_scaled, self._V[self.q - 1]))               # G1 first, G2 second
        return final_verify(lhs_ml, rhs_ml)


# ---------------------------------------------------------------------------
# ZAC Accumulator  — public API
# ---------------------------------------------------------------------------

class ZACAccumulator:
    """
    ZAC accumulator for UltraRAG corpus image verification.

    The accumulated set:
        S = { SHA256(image_bytes) for each image in corpus }

    Public output (corpus fingerprint):
        commitment cm ∈ G1  →  48-byte BLS12-381 G1 point

    Usage
    -----
    # Build from existing corpus
    acc = ZACAccumulator.from_corpus(
        embeddings_path="embedding/embedding.npy",
        corpus_jsonl="corpora/image.jsonl",
        corpus_base_dir="corpora/",
    )
    print(acc.root_hex())                   # 96-char hex string, 48 bytes

    # Prove and verify membership
    h = acc.image_hash("corpora/image/doc/page_0.jpg")
    proof = acc.prove_membership(h)
    assert acc.verify_membership(h, proof)

    # Save fingerprint to disk
    acc.save("output/phase1/fingerprint.json")
    """

    def __init__(
        self,
        S: Set[bytes],
        N_bound: Optional[int] = None,
        epsilon: float = 0.01,
        alpha: Optional[int] = None,
        n_filters: int = 1,
    ) -> None:
        """
        S         : set of elements to accumulate (bytes, e.g. SHA256 digests).
        N_bound   : upper bound on |S| for BF parameter choice (default: |S|).
        epsilon   : per-layer BF false-positive rate (default 1%).
        alpha     : Pointproofs trapdoor. None → randomly generated per layer.
        n_filters : number of independent BF+PR layers in cascade.
                    Compound FPR = epsilon^n_filters (e.g. n=2: 1%→0.01%).
                    n_filters=1 is fully backward-compatible.
        """
        self._S = frozenset(S)
        self._n_filters = n_filters
        N = max(N_bound or len(S), 1)

        # Compute k from the first layer to define seed spacing between layers.
        _bf0 = BloomFilter.optimal_params(N, epsilon, seed_offset=0)
        k = _bf0.k

        # Build n_filters independent layers.
        # Layer i uses mmh3 seeds [i*k, i*k+1, ..., i*k+k-1] so hash functions
        # are disjoint across layers, making FP events statistically independent.
        self._layers: List[dict] = []
        for i in range(n_filters):
            bf = BloomFilter.optimal_params(N, epsilon, seed_offset=i * k)
            v  = bf.gen(set(S))
            r  = _random_zp()
            pr = Pointproofs(q=bf.q + 1, alpha=alpha)
            cm = pr.commit(v, r)
            self._layers.append({"bf": bf, "v": v, "r": r, "pr": pr, "cm": cm})

        # Expose layer-0 attributes for backward compatibility.
        L = self._layers[0]
        self._bf: BloomFilter       = L["bf"]
        self._v:  List[int]         = L["v"]
        self._r:  int               = L["r"]
        self._pr: Pointproofs       = L["pr"]
        self._cm                    = L["cm"]

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    @property
    def commitment(self):
        """Raw G1 point (ZAC commitment)."""
        return self._cm

    def root_bytes(self) -> bytes:
        """48-byte BLS12-381 G1 compressed point (layer 0)."""
        return _g1_compress(self._cm)

    def root_hex(self) -> str:
        """
        Concatenation of all layer commitments as hex.
        Single-filter (n=1): 96-char hex of the 48-byte G1 point (backward compat).
        Cascade (n>1):       n×96 chars — one 48-byte G1 point per layer,
                             concatenated in layer order.
        """
        return "".join(_g1_compress(L["cm"]).hex() for L in self._layers)

    @staticmethod
    def image_hash(image_path: str) -> bytes:
        """SHA256(image_bytes) — 旧接口，仅用于旧版 prover_state 兼容。
        新代码请使用 image_embedding_hash。"""
        p = Path(image_path)
        if p.exists():
            return hashlib.sha256(p.read_bytes()).digest()
        return hashlib.sha256(str(p).encode()).digest()

    @staticmethod
    def image_embedding_hash(image_path: str, embedding: "np.ndarray") -> bytes:
        """
        SHA256(image_bytes ∥ embedding_bytes) — 跨层承诺绑定版本。

        将图片文件字节与对应的 float32 embedding 拼接后哈希，
        确保任何对 embedding 的篡改都会使 ZAC 成员证明失效。
        与 from_corpus 使用相同的构造方式，两侧必须保持一致。
        """
        import numpy as np
        p = Path(image_path)
        img_bytes = p.read_bytes() if p.exists() else str(p).encode()
        emb_bytes = np.asarray(embedding, dtype=np.float32).tobytes()
        return hashlib.sha256(img_bytes + emb_bytes).digest()

    def prove_membership(self, element: bytes) -> dict:
        """
        ZAC.ProveM for one element.
        Single-filter: returns flat format (backward compat).
        Cascade (n_filters>1): returns {"layer_proofs": [...]} format.
        """
        S_hat = {element}

        if self._n_filters == 1:
            I = self._bf.membership_indices(S_hat)
            v_I = [self._v[i] for i in I]
            individual = [self._pr.prove(i, self._v, self._r) for i in I]
            pi_hat = self._pr.aggregate(self._cm, I, v_I, individual)
            return {
                "element_hex": element.hex(),
                "I": I,
                "v_I": v_I,
                "proof_hex": _g1_compress(pi_hat).hex(),
                "cm_hex": self.root_hex(),
            }

        # Cascade: one proof per layer.
        layer_proofs = []
        for L in self._layers:
            I = L["bf"].membership_indices(S_hat)
            v_I = [L["v"][i] for i in I]
            individual = [L["pr"].prove(i, L["v"], L["r"]) for i in I]
            pi_hat = L["pr"].aggregate(L["cm"], I, v_I, individual)
            layer_proofs.append({
                "I": I,
                "v_I": v_I,
                "proof_hex": _g1_compress(pi_hat).hex(),
                "cm_hex": _g1_compress(L["cm"]).hex(),
            })
        return {
            "element_hex": element.hex(),
            "cm_hex": self.root_hex(),
            "n_filters": self._n_filters,
            "layer_proofs": layer_proofs,
        }

    def prove_membership_batch(self, elements: List[bytes]) -> dict:
        """
        ZAC.ProveM for a subset Ŝ = {elements}.

        Single-filter (n_filters=1): produces ONE 48-byte G1 proof (old format).
        Cascade (n_filters>1):       produces one 48-byte proof per layer;
                                     compound FPR = epsilon^n_filters.
        """
        S_hat = set(elements)
        layer_proofs = []
        for L in self._layers:
            I      = L["bf"].membership_indices(S_hat)
            v_I    = [L["v"][i] for i in I]
            indiv  = [L["pr"].prove(i, L["v"], L["r"]) for i in I]
            pi_hat = L["pr"].aggregate(L["cm"], I, v_I, indiv)
            layer_proofs.append({
                "I":         I,
                "v_I":       v_I,
                "proof_hex": _g1_compress(pi_hat).hex(),
                "cm_hex":    _g1_compress(L["cm"]).hex(),
            })

        base = {
            "elements_hex": sorted(e.hex() for e in elements),
            "cm_hex":       self.root_hex(),
            "bf_q":         self._bf.q,
            "bf_k":         self._bf.k,
        }
        if self._n_filters == 1:
            # Backward-compatible single-layer format.
            lp = layer_proofs[0]
            return {**base, "I": lp["I"], "v_I": lp["v_I"],
                    "proof_hex": lp["proof_hex"]}
        # Cascade format.
        return {**base, "n_filters": self._n_filters,
                "layer_proofs": layer_proofs}

    def verify_membership_batch(self, elements: List[bytes], proof: dict) -> bool:
        """
        ZAC.VerifyM for a batch of elements.
        Accepts both single-filter (old) and cascade (new) proof formats.
        """
        if proof.get("cm_hex") != self.root_hex():
            return False

        S_hat = set(elements)

        if self._n_filters == 1:
            # Backward-compatible path.
            I      = self._bf.membership_indices(S_hat)
            v_I    = [1] * len(I)
            pi_hat = _g1_decompress(bytes.fromhex(proof["proof_hex"]))
            if pi_hat is None:
                return False
            return self._pr.verify(self._cm, I, v_I, pi_hat)

        # Cascade: ALL layers must pass.
        layer_proofs = proof.get("layer_proofs", [])
        if len(layer_proofs) != self._n_filters:
            return False
        for L, lp in zip(self._layers, layer_proofs):
            I      = L["bf"].membership_indices(S_hat)
            v_I    = [1] * len(I)
            pi_hat = _g1_decompress(bytes.fromhex(lp["proof_hex"]))
            if pi_hat is None:
                return False
            if not L["pr"].verify(L["cm"], I, v_I, pi_hat):
                return False
        return True

    def verify_membership(self, element: bytes, proof: dict) -> bool:
        """
        ZAC.VerifyM for a single element.
        """
        if proof.get("cm_hex") != self.root_hex():
            return False

        S_hat = {element}

        if self._n_filters == 1:
            I      = self._bf.membership_indices(S_hat)
            v_I    = [1] * len(I)
            pi_hat = _g1_decompress(bytes.fromhex(proof["proof_hex"]))
            if pi_hat is None:
                return False
            return self._pr.verify(self._cm, I, v_I, pi_hat)

        layer_proofs = proof.get("layer_proofs", [])
        if len(layer_proofs) != self._n_filters:
            return False
        for L, lp in zip(self._layers, layer_proofs):
            I      = L["bf"].membership_indices(S_hat)
            v_I    = [1] * len(I)
            pi_hat = _g1_decompress(bytes.fromhex(lp["proof_hex"]))
            if pi_hat is None:
                return False
            if not L["pr"].verify(L["cm"], I, v_I, pi_hat):
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, output_path: str) -> None:
        """Save public corpus fingerprint manifest to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "scheme": "ZAC (Bloom Filter + Pointproofs / BLS12-381)",
            "reference": "Dang et al., TPS-ISA 2022",
            "commitment_hex": self.root_hex(),
            "commitment_size_bytes": 48 * self._n_filters,
            "n_filters": self._n_filters,
            "bf_q": self._bf.q,
            "bf_k": self._bf.k,
            "num_elements": len(self._S),
            "elements_hex": sorted(e.hex() for e in self._S),
            "bf_nonzero_positions": [i for i, b in enumerate(self._v) if b],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def save_prover_state(self, path: str) -> None:
        """
        Persist full prover state (trapdoor α, blinding r, BF vector v) for all layers.
        Each layer gets a companion .crs binary so future loads skip the CRS rebuild.
        NOTE: α is a secret — protect this file appropriately.
        """
        base = Path(path).with_suffix("")
        layers_state = []
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for i, L in enumerate(self._layers):
            crs_path = str(base) + f"_layer{i}.crs"
            layers_state.append({
                "bf_q":        L["bf"].q,
                "bf_k":        L["bf"].k,
                "seed_offset": L["bf"]._seed_offset,
                "v":           L["v"],
                "r_hex":       hex(L["r"]),
                "alpha_hex":   hex(L["pr"]._alpha),
                "cm_hex":      _g1_compress(L["cm"]).hex(),
                "crs_cache":   crs_path,
            })
            L["pr"].save_crs(crs_path)

        state = {
            "scheme":       "ZAC-prover-state-v3",
            "n_filters":    self._n_filters,
            "layers":       layers_state,
            "cm_hex":       self.root_hex(),
            "elements_hex": sorted(e.hex() for e in self._S),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_prover_state(cls, path: str) -> "ZACAccumulator":
        """
        Restore a ZACAccumulator from a file saved by save_prover_state().

        Supports v2 (single-layer) and v3 (multi-layer / cascade) state files.
        If companion .crs files exist, loads them directly (~5 s per layer).
        """
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        obj = cls.__new__(cls)
        obj._S = frozenset(bytes.fromhex(h) for h in state["elements_hex"])

        scheme = state.get("scheme", "ZAC-prover-state-v2")

        if scheme == "ZAC-prover-state-v2":
            # Legacy single-layer format — upgrade to v3 in memory.
            layers_raw = [{
                "bf_q":        state["bf_q"],
                "bf_k":        state["bf_k"],
                "seed_offset": 0,
                "v":           state["v"],
                "r_hex":       state["r_hex"],
                "alpha_hex":   state["alpha_hex"],
                "crs_cache":   state.get("crs_cache",
                                         str(Path(path).with_suffix(".crs"))),
            }]
            n_filters = 1
        else:
            layers_raw = state["layers"]
            n_filters  = state.get("n_filters", len(layers_raw))

        obj._n_filters = n_filters
        obj._layers    = []

        for lr in layers_raw:
            bf = BloomFilter(q=lr["bf_q"], k=lr["bf_k"],
                             seed_offset=lr.get("seed_offset", 0))
            v  = lr["v"]
            r  = int(lr["r_hex"], 16)
            q_pr = lr["bf_q"] + 1

            pr = Pointproofs.__new__(Pointproofs)
            pr.q      = q_pr
            pr._alpha = int(lr["alpha_hex"], 16)
            crs_path  = lr.get("crs_cache", "")
            if crs_path and Path(crs_path).exists():
                pr._setup_from_cache(crs_path)
            else:
                pr._setup(pr._alpha)

            cm = pr.commit(v, r)
            obj._layers.append({"bf": bf, "v": v, "r": r, "pr": pr, "cm": cm})

        # Backward-compat aliases pointing to layer 0.
        L = obj._layers[0]
        obj._bf, obj._v, obj._r, obj._pr, obj._cm = (
            L["bf"], L["v"], L["r"], L["pr"], L["cm"]
        )
        return obj

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_corpus(
        cls,
        embeddings_path: str,
        corpus_jsonl: str,
        corpus_base_dir: str,
        epsilon: float = 0.01,
        n_filters: int = 1,
    ) -> "ZACAccumulator":
        """
        Build ZAC accumulator from UltraRAG image corpus.

        跨层承诺绑定（Cross-layer Commitment Binding）：
        S = { SHA256(image_bytes_i ∥ embedding_bytes_i) }

        将 embedding 纳入承诺元素，确保服务器无法在不改变 ZAC Root 的情况下
        单独替换 FAISS 中的 embedding 向量，从而关闭 embedding 偷换攻击路径。

        元素构造：element_i = SHA256(image_file_bytes_i ∥ float32_embedding_bytes_i)
        """
        import numpy as np

        corpus_dir = Path(corpus_base_dir)
        embeddings = np.load(embeddings_path).astype(np.float32)  # (N, D)
        S: Set[bytes] = set()

        with open(corpus_jsonl, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rel = item.get("image_path", "")
                img_path = (corpus_dir / rel).resolve()
                img_bytes = (img_path.read_bytes() if img_path.exists()
                             else str(img_path).encode())
                emb_bytes = (embeddings[idx].tobytes()
                             if idx < len(embeddings) else b"")
                h = hashlib.sha256(img_bytes + emb_bytes).digest()
                S.add(h)

        return cls(S=S, epsilon=epsilon, n_filters=n_filters)

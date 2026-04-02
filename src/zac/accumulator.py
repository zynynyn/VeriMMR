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

from py_ecc.optimized_bls12_381 import (
    G1, G2,
    add,          # point addition  (G1 or G2)
    multiply,     # scalar mult      (G1 or G2)
    pairing,      # e: G2 × G1 → GT  (note: G2 first!)
    Z1, Z2,       # identity elements for G1, G2
    curve_order,  # prime order p of G1/G2
    field_modulus,
)
from py_ecc.fields import optimized_bls12_381_FQ as FQ
from py_ecc.fields import optimized_bls12_381_FQ12 as FQ12
from py_ecc.fields import optimized_bls12_381_FQ2 as FQ2


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
# G1 compression  (BLS12-381 standard 48-byte encoding)
# ---------------------------------------------------------------------------

def _g1_compress(point) -> bytes:
    """Compress a G1 affine point to 48 bytes (standard BLS12-381 serialization)."""
    if point == Z1:
        return b"\xc0" + b"\x00" * 47  # compressed infinity flag

    # Convert from projective (X:Y:Z) to affine (x, y)
    x_fq, y_fq, z_fq = point
    z_inv = pow(z_fq.n, field_modulus - 2, field_modulus)
    x = (x_fq.n * z_inv) % field_modulus
    y = (y_fq.n * z_inv) % field_modulus

    flag = 0x80  # compression bit
    if y > (field_modulus - 1) // 2:
        flag |= 0x20  # sign bit

    x_bytes = bytearray(x.to_bytes(48, "big"))
    x_bytes[0] |= flag
    return bytes(x_bytes)


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

    Implemented with MurmurHash3 (mmh3) with seeds 0, 1, ..., k-1.
    """

    def __init__(self, q: int, k: int) -> None:
        self.q = q
        self.k = k

    @classmethod
    def optimal_params(cls, N: int, epsilon: float = 0.01) -> "BloomFilter":
        q = max(1, math.ceil(-N * math.log(epsilon) / (math.log(2) ** 2)))
        k = max(1, round((q / N) * math.log(2)))
        return cls(q=q, k=k)

    def _positions(self, element: bytes) -> List[int]:
        return [mmh3.hash(element, seed=i, signed=False) % self.q for i in range(self.k)]

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

    Performance note:
      py_ecc is pure Python; CRS generation for q~100 takes ~10-60 s.
      For production use the Rust Pointproofs library (github.com/algorand/pointproofs).
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
        self._P: List = [multiply(G1, a[i]) for i in range(1, self.q + 1)]

        # Px[j] = g1^{α^{q+2+j}},  j = 0..q-2  (i.e. α^{q+2}..α^{2q})
        self._Px: List = [multiply(G1, a[self.q + 2 + j]) for j in range(self.q - 1)]

        # V[i-1] = g2^{α^i},  i = 1..q
        self._V: List = [multiply(G2, a[i]) for i in range(1, self.q + 1)]

        # gT^{α^{q+1}} = e(g2^{α^q}, g1^α)  = e(V[q-1], P[0])
        # pairing signature: pairing(Q: G2, P: G1) → FQ12
        self._gT_aq1: FQ12 = pairing(self._V[self.q - 1], self._P[0])

    # -----------------------------------------------------------------------
    # CRS serialization  (skip expensive _setup on reload)
    # -----------------------------------------------------------------------

    def save_crs(self, path: str) -> None:
        """
        Serialize the precomputed CRS to a binary file (projective coordinates).
        Format (big-endian, fixed 48 bytes per field element):
          [4B] q
          [q   × 144B] P   (G1 projective: X 48B, Y 48B, Z 48B)
          [q-1 × 144B] Px  (G1 projective)
          [q   × 288B] V   (G2 projective: X0,X1, Y0,Y1, Z0,Z1 each 48B)
        gT_aq1 is recomputed from V[q-1] and P[0] on load (one pairing, ~0.3 s).
        """
        def fq_n(fq) -> int:
            return fq.n if hasattr(fq, "n") else int(fq)

        def g1_bytes(pt) -> bytes:
            x, y, z = pt
            return (fq_n(x).to_bytes(48, "big") +
                    fq_n(y).to_bytes(48, "big") +
                    fq_n(z).to_bytes(48, "big"))

        def g2_bytes(pt) -> bytes:
            x, y, z = pt
            out = b""
            for coord in (x, y, z):
                c0, c1 = coord.coeffs
                out += fq_n(c0).to_bytes(48, "big") + fq_n(c1).to_bytes(48, "big")
            return out  # 6 × 48 = 288 bytes

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.q.to_bytes(4, "big"))
            for pt in self._P:
                f.write(g1_bytes(pt))
            for pt in self._Px:
                f.write(g1_bytes(pt))
            for pt in self._V:
                f.write(g2_bytes(pt))

    def _setup_from_cache(self, path: str) -> None:
        """Load precomputed CRS from projective-format binary file."""
        def read_g1(f) -> tuple:
            data = f.read(144)
            x = FQ(int.from_bytes(data[0:48],   "big"))
            y = FQ(int.from_bytes(data[48:96],  "big"))
            z = FQ(int.from_bytes(data[96:144], "big"))
            return (x, y, z)

        def read_g2(f) -> tuple:
            data = f.read(288)
            def rd(i): return int.from_bytes(data[i*48:(i+1)*48], "big")
            x = FQ2((rd(0), rd(1)))
            y = FQ2((rd(2), rd(3)))
            z = FQ2((rd(4), rd(5)))
            return (x, y, z)

        with open(path, "rb") as f:
            q = int.from_bytes(f.read(4), "big")
            assert q == self.q, f"CRS cache mismatch: expected q={self.q}, got {q}"
            self._P  = [read_g1(f) for _ in range(q)]
            self._Px = [read_g1(f) for _ in range(q - 1)]
            self._V  = [read_g2(f) for _ in range(q)]
        # Recompute gT^{α^{q+1}} = e(V[q-1], P[0])  — one pairing, ~0.3 s
        self._gT_aq1: FQ12 = pairing(self._V[self.q - 1], self._P[0])

    # -----------------------------------------------------------------------
    # CRS accessors
    # -----------------------------------------------------------------------

    def _g1_alpha_i(self, i: int):
        """g1^{α^i} for i ∈ [1,q] ∪ [q+2, 2q].  α^{q+1} not in CRS."""
        if 1 <= i <= self.q:
            return self._P[i - 1]
        if self.q + 2 <= i <= 2 * self.q:
            return self._Px[i - (self.q + 2)]
        raise ValueError(f"α^{i} not available in CRS (q={self.q}).")

    def _g2_alpha_i(self, i: int):
        """g2^{α^i} for i ∈ [1, q]."""
        if 1 <= i <= self.q:
            return self._V[i - 1]
        raise ValueError(f"g2^α^{i} not available in CRS (q={self.q}).")

    # -----------------------------------------------------------------------
    # Protocol
    # -----------------------------------------------------------------------

    def commit(self, m: List[int], r: int):
        """
        C = g1^{r·α^q} · ∏_{i=1}^{q-1} (g1^{α^i})^{mᵢ}
        m is 0-indexed, length q-1; internally 1-indexed.
        """
        C = multiply(self._P[self.q - 1], r % curve_order)  # g1^{r·α^q}
        for i1 in range(1, self.q):                          # i = 1..q-1
            if m[i1 - 1] != 0:
                C = add(C, multiply(self._P[i1 - 1], m[i1 - 1]))
        return C

    def prove(self, pos: int, m: List[int], r: int):
        """
        πᵢ = ∏_{j∈[q], j≠i} g1^{m'ⱼ · α^{q+1-i+j}}
        pos is 0-indexed.  Internally: i = pos+1 (1-indexed).
        m'  = (m[0], ..., m[q-2], r)  — length q.
        """
        i = pos + 1                           # 1-indexed position
        m_prime = list(m[: self.q - 1]) + [r]  # length q

        pi = Z1
        for j in range(1, self.q + 1):        # j = 1..q  (1-indexed)
            if j == i:
                continue
            exp = self.q + 1 - i + j          # α^{q+1-i+j}
            if exp == self.q + 1:             # not in CRS
                continue
            if m_prime[j - 1] == 0:
                continue
            pi = add(pi, multiply(self._g1_alpha_i(exp), m_prime[j - 1]))
        return pi

    def _challenge(self, pos: int, C, I: List[int], v_I: List[int]) -> int:
        """tᵢ = H'(i, C, I, v[I]) — Fiat-Shamir challenge."""
        data = (
            pos.to_bytes(4, "big")
            + _g1_compress(C)
            + b"".join(p.to_bytes(4, "big") for p in I)
            + bytes(v_I)
        )
        return _hash_to_zp(data)

    def aggregate(self, C, I: List[int], v_I: List[int], proofs: List):
        """
        π̂ = ∏ᵢ∈I πᵢ^{tᵢ}   (aggregated proof, single G1 point)
        """
        agg = Z1
        for pos, pi in zip(I, proofs):
            t = self._challenge(pos, C, I, v_I)
            agg = add(agg, multiply(pi, t))
        return agg

    def verify(self, C, I: List[int], v_I: List[int], pi_hat) -> bool:
        """
        Check: e(C, Σᵢ tᵢ·g2^{α^{q+1-i}}) ?= e(π̂, g2) · gT^{α^{q+1}·Σᵢ vᵢ·tᵢ}
        pairing order: pairing(Q: G2, P: G1) → FQ12
        GT operation: FQ12 field multiplication and exponentiation.
        """
        t_vals = [self._challenge(pos, C, I, v_I) for pos in I]

        # --- LHS ---
        # Σᵢ tᵢ · g2^{α^{q+1-i1}}   where i1 = pos+1 (1-indexed)
        V_sum = Z2
        for pos, t in zip(I, t_vals):
            i1 = pos + 1
            exp = self.q + 1 - i1           # = q - pos  (≥1 when pos ≤ q-1)
            if 1 <= exp <= self.q:
                V_sum = add(V_sum, multiply(self._g2_alpha_i(exp), t))
        # If V_sum is Z2 (all exps out of range), verification cannot succeed
        if V_sum == Z2:
            return False
        lhs: FQ12 = pairing(V_sum, C)      # e(g2_part, C_G1)

        # --- RHS ---
        # e(π̂, g2) · gT^{α^{q+1} · Σᵢ vᵢ·tᵢ}
        rhs_pairing: FQ12 = pairing(G2, pi_hat)   # e(g2, π̂)
        exp_sum = sum(v * t for v, t in zip(v_I, t_vals)) % curve_order
        rhs: FQ12 = rhs_pairing * (self._gT_aq1 ** exp_sum)

        return lhs == rhs


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
    ) -> None:
        """
        S       : set of elements to accumulate (bytes, e.g. SHA256 digests).
        N_bound : upper bound on |S| for BF parameter choice (default: |S|).
        epsilon : BF false-positive rate (default 1%).
        alpha   : Pointproofs trapdoor. None → randomly generated.
        """
        self._S = frozenset(S)
        N = max(N_bound or len(S), 1)

        self._bf = BloomFilter.optimal_params(N, epsilon)
        self._v: List[int] = self._bf.gen(set(S))
        self._r: int = _random_zp()

        # Pointproofs commits to a vector of length (q_pr - 1).
        # BF generates a vector of length bf.q, so we set q_pr = bf.q + 1
        # so that all BF positions [0, bf.q-1] stay in the valid message slots.
        self._pr = Pointproofs(q=self._bf.q + 1, alpha=alpha)
        self._cm = self._pr.commit(self._v, self._r)

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    @property
    def commitment(self):
        """Raw G1 point (ZAC commitment)."""
        return self._cm

    def root_bytes(self) -> bytes:
        """48-byte BLS12-381 G1 compressed point."""
        return _g1_compress(self._cm)

    def root_hex(self) -> str:
        """96-character hex string of the 48-byte ZAC root."""
        return self.root_bytes().hex()

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
        Returns JSON-serializable dict with the 48-byte aggregated proof.
        """
        S_hat = {element}
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

    def prove_membership_batch(self, elements: List[bytes]) -> dict:
        """
        ZAC.ProveM for a subset Ŝ = {elements}.
        Produces ONE aggregated 48-byte G1 proof covering all elements.
        This is the Phase-2 call: called after top-k retrieval.
        """
        S_hat = set(elements)
        I = self._bf.membership_indices(S_hat)
        v_I = [self._v[i] for i in I]

        individual = [self._pr.prove(i, self._v, self._r) for i in I]
        pi_hat = self._pr.aggregate(self._cm, I, v_I, individual)

        return {
            "elements_hex": sorted(e.hex() for e in elements),
            "I": I,
            "v_I": v_I,
            "proof_hex": _g1_compress(pi_hat).hex(),
            "cm_hex": self.root_hex(),
            "bf_q": self._bf.q,
            "bf_k": self._bf.k,
        }

    def verify_membership_batch(self, elements: List[bytes], proof: dict) -> bool:
        """
        ZAC.VerifyM for a batch of elements against an aggregated proof.
        The verifier reconstructs I from the elements (public computation),
        then checks the single pairing equation.
        """
        if proof.get("cm_hex") != self.root_hex():
            return False

        S_hat = set(elements)
        I = self._bf.membership_indices(S_hat)
        v_I = [1] * len(I)   # all BF positions must be 1 for membership

        pi_bytes = bytes.fromhex(proof["proof_hex"])
        pi_hat = _g1_decompress(pi_bytes)
        if pi_hat is None:
            return False

        return self._pr.verify(self._cm, I, v_I, pi_hat)

    def verify_membership(self, element: bytes, proof: dict) -> bool:
        """
        ZAC.VerifyM — the verifier only needs the element, the proof dict,
        and the public commitment (stored in proof["cm_hex"]).
        No access to the original set or BF vector needed.
        """
        if proof.get("cm_hex") != self.root_hex():
            return False  # proof is for a different commitment

        S_hat = {element}
        I = self._bf.membership_indices(S_hat)
        v_I = [1] * len(I)   # all positions must be 1 (membership condition)

        pi_bytes = bytes.fromhex(proof["proof_hex"])
        pi_hat = _g1_decompress(pi_bytes)
        if pi_hat is None:
            return False

        return self._pr.verify(self._cm, I, v_I, pi_hat)

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
            "commitment_size_bytes": 48,
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
        Persist full prover state (trapdoor α, blinding r, BF vector v).
        Also writes a companion .crs binary file so future loads skip the
        expensive CRS rebuild (~170 s → ~5 s).
        NOTE: α is a secret — protect this file appropriately.
        """
        crs_path = str(Path(path).with_suffix(".crs"))
        state = {
            "scheme": "ZAC-prover-state-v2",
            "bf_q": self._bf.q,
            "bf_k": self._bf.k,
            "v": self._v,
            "r_hex": hex(self._r),
            "alpha_hex": hex(self._pr._alpha),
            "cm_hex": self.root_hex(),
            "elements_hex": sorted(e.hex() for e in self._S),
            "crs_cache": crs_path,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        self._pr.save_crs(crs_path)

    @classmethod
    def load_prover_state(cls, path: str) -> "ZACAccumulator":
        """
        Restore a ZACAccumulator from a file saved by save_prover_state().

        If a companion .crs binary file exists, loads the precomputed CRS
        directly (~5 s: point reconstruction + 1 pairing).
        Otherwise falls back to recomputing from α (~170 s).
        """
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        S = {bytes.fromhex(h) for h in state["elements_hex"]}
        alpha = int(state["alpha_hex"], 16)
        r = int(state["r_hex"], 16)
        q_pr = state["bf_q"] + 1

        obj = cls.__new__(cls)
        obj._S = frozenset(S)
        obj._bf = BloomFilter(q=state["bf_q"], k=state["bf_k"])
        obj._v = state["v"]
        obj._r = r

        # Try loading CRS from cache to skip EC multiply rebuilding
        crs_path = state.get("crs_cache") or str(Path(path).with_suffix(".crs"))
        pr = Pointproofs.__new__(Pointproofs)
        pr.q = q_pr
        pr._alpha = alpha
        if Path(crs_path).exists():
            pr._setup_from_cache(crs_path)
        else:
            pr._setup(alpha)   # fallback: full rebuild

        obj._pr = pr
        obj._cm = obj._pr.commit(obj._v, obj._r)
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

        return cls(S=S, epsilon=epsilon)


# ---------------------------------------------------------------------------
# G1 decompression  (48 bytes → projective point)
# ---------------------------------------------------------------------------

def _g1_decompress(data: bytes):
    """Decompress 48-byte BLS12-381 G1 point. Returns None on failure."""
    if len(data) != 48:
        return None
    if data == b"\xc0" + b"\x00" * 47:
        return Z1

    if not (data[0] & 0x80):
        return None  # not compressed

    sort_flag = bool(data[0] & 0x20)
    x_bytes = bytearray(data)
    x_bytes[0] &= 0x1f
    x = int.from_bytes(bytes(x_bytes), "big")

    p = field_modulus
    y_sq = (pow(x, 3, p) + 4) % p
    # Tonelli-Shanks shortcut: p ≡ 3 (mod 4)
    y = pow(y_sq, (p + 1) // 4, p)
    if pow(y, 2, p) != y_sq:
        return None  # x not on curve

    if sort_flag != (y > p // 2):
        y = p - y

    return (FQ(x), FQ(y), FQ(1))  # projective Z=1

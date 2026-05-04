"""
Sumcheck-based verifiable inner product and ranking proof.

Protocol
--------
  - Prove  s = q · v  using multi-round non-interactive Sumcheck (Fiat-Shamir)
  - Prove  s₁ ≥ s₂ ≥ … ≥ sₖ  via explicit non-negative delta witnesses

Reference
---------
  Thaler, "Proofs, Arguments, and Zero-Knowledge" (2022), Chapter 4.
  zkGPT inner-product argument (Qu et al., 2024).

Field
-----
  ℤ_p, p = 2^61 − 1 (Mersenne prime, fits in Python native big-int, fast mod).

Complexity
----------
  Prover:    O(d) field mults  (~2d across all ell rounds)
  Verifier:  O(d) field mults  (same: MLE evaluation at challenge point)
  Proof size: ell × 3 field elements  (ell = ⌈log₂ d⌉)
              For d=2048: 11 × 3 × 8 B = 264 B per inner product
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ── Field arithmetic (Mersenne prime 2^61 − 1) ────────────────────────────────

P: int = (1 << 61) - 1          # 2^61 − 1, Mersenne prime
INV2: int = (P + 1) >> 1        # 2^{-1} mod P  (since P is odd prime)

# ── BLS12-381 scalar field (for IPA-mode Sumcheck) ────────────────────────────

P_FR: int = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
INV2_FR: int = (P_FR + 1) >> 1   # 2^{-1} mod P_FR

# BLS12-381 Fp (for G1 Montgomery decode)
_P_FP: int = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
_R_FP: int = pow(2, 384, _P_FP)
_R_FP_INV: int = pow(_R_FP, -1, _P_FP)

_ROOT = Path(__file__).parent.parent.parent.resolve()
_BIN_DIR = _ROOT / "src" / "zkllm"


def _m(x: int) -> int:
    """Reduce x modulo P.  Python's % always returns non-negative result."""
    return x % P


def _mfr(x: int) -> int:
    """Reduce x modulo P_FR (BLS12-381 scalar field)."""
    return x % P_FR


def _poly2_eval(g0: int, g1: int, g2: int, r: int) -> int:
    """
    Evaluate the unique degree-≤2 polynomial through (0,g0),(1,g1),(2,g2) at r.
    All arguments are field elements (already reduced mod P).

    Lagrange interpolation:
        c = g0
        a = (g2 − 2·g1 + g0) · 2^{-1}
        b = g1 − g0 − a
        result = c + r·(b + r·a)
    """
    a = _m(_m(g2 - _m(2 * g1) + g0) * INV2)
    b = _m(g1 - g0 - a)
    c = g0
    return _m(c + r * _m(b + r * a))


def _poly2_eval_fr(g0: int, g1: int, g2: int, r: int) -> int:
    """Same as _poly2_eval but over BLS12-381 scalar field P_FR."""
    a = _mfr(_mfr(g2 - _mfr(2 * g1) + g0) * INV2_FR)
    b = _mfr(g1 - g0 - a)
    c = g0
    return _mfr(c + r * _mfr(b + r * a))


def _fs_challenge(transcript: bytes, g0: int, g1: int, g2: int) -> Tuple[int, bytes]:
    """Derive a non-interactive field challenge via Fiat-Shamir (SHA-256)."""
    msg = (
        transcript
        + g0.to_bytes(8, "big")
        + g1.to_bytes(8, "big")
        + g2.to_bytes(8, "big")
    )
    h = hashlib.sha256(msg).digest()
    r = int.from_bytes(h, "big") % P
    return r, transcript + h


def _fs_challenge_fr(transcript: bytes, g0: int, g1: int, g2: int) -> Tuple[int, bytes]:
    """Fiat-Shamir for BLS12-381 Fr field (32-byte elements, ~255-bit field)."""
    msg = (
        transcript
        + g0.to_bytes(32, "big")
        + g1.to_bytes(32, "big")
        + g2.to_bytes(32, "big")
    )
    h = hashlib.sha256(msg).digest()
    r = int.from_bytes(h, "big") % P_FR
    return r, transcript + h


# ── Quantization ───────────────────────────────────────────────────────────────

def quantize(v: List[float], scale: int = 65536) -> List[int]:
    """
    Convert float32 vector to integer field elements.
    Multiply by scale and round to nearest integer.
    scale=65536 (2^16): aligns with zkLLM quantization, L∞ error ~2e-5,
    overflow-safe: scale^2 * D = 8.8e12 << p = 2.3e18.
    """
    return [_m(int(round(x * scale))) for x in v]


# ── Sumcheck Prover ────────────────────────────────────────────────────────────

def prove_inner_product(
    q: List[Union[int, float]],
    v: List[Union[int, float]],
    scale: int = 65536,
) -> dict:
    """
    Non-interactive Sumcheck proof that  H = Σⱼ q_j · v_j  (over ℤ_p).

    The claimed inner product H is the sum of element-wise products of the
    (quantized) integer vectors.  If float inputs are supplied they are
    quantized first with the given scale.

    Returns
    -------
    dict with keys:
        H        – claimed inner product (field element, int)
        ell      – number of Sumcheck rounds = ⌈log₂ d⌉
        d        – original vector length
        scale    – quantization scale applied to float inputs
        rounds   – list of (g0, g1, g2) per round  (each a 3-tuple of ints)
        final_q  – last folded value of Q  (for verifier sanity check)
        final_v  – last folded value of V
    """
    # Quantize float inputs
    if q and isinstance(q[0], float):
        q_int: List[int] = quantize(q, scale)
    else:
        q_int = [_m(x) for x in q]
    if v and isinstance(v[0], float):
        v_int: List[int] = quantize(v, scale)
    else:
        v_int = [_m(x) for x in v]

    d = len(q_int)
    assert len(v_int) == d, f"q and v length mismatch: {d} vs {len(v_int)}"

    ell = max(1, math.ceil(math.log2(d))) if d > 1 else 1
    n = 1 << ell                            # pad to next power of 2

    Q: List[int] = q_int + [0] * (n - d)
    V: List[int] = v_int + [0] * (n - d)

    # Claimed sum: H = Σ Q[j]·V[j]
    H: int = _m(sum(_m(Q[j] * V[j]) for j in range(n)))

    transcript: bytes = H.to_bytes(8, "big")
    rounds: List[Tuple[int, int, int]] = []

    for _ in range(ell):
        half = len(Q) >> 1

        # g(0) = Σ_{j<half} Q[j]·V[j]
        g0 = _m(sum(_m(Q[j] * V[j]) for j in range(half)))
        # g(1) = Σ_{j<half} Q[half+j]·V[half+j]
        g1 = _m(sum(_m(Q[half + j] * V[half + j]) for j in range(half)))
        # g(2): linear interpolation Q_lin(j,2) = 2·Q[half+j] − Q[j]
        g2 = _m(sum(
            _m(_m(2 * Q[half + j] - Q[j]) * _m(2 * V[half + j] - V[j]))
            for j in range(half)
        ))

        rounds.append((g0, g1, g2))
        r, transcript = _fs_challenge(transcript, g0, g1, g2)

        # Fold: Q'[j] = Q[j] + r·(Q[half+j] − Q[j])
        Q = [_m(Q[j] + _m(r * _m(Q[half + j] - Q[j]))) for j in range(half)]
        V = [_m(V[j] + _m(r * _m(V[half + j] - V[j]))) for j in range(half)]

    return {
        "H": H,
        "ell": ell,
        "d": d,
        "scale": scale,
        "rounds": [(int(g0), int(g1), int(g2)) for g0, g1, g2 in rounds],
        "final_q": int(Q[0]),
        "final_v": int(V[0]),
    }


# ── Sumcheck Verifier ──────────────────────────────────────────────────────────

def verify_inner_product(
    q: List[Union[int, float]],
    v: List[Union[int, float]],
    proof: dict,
    scale: int | None = None,
) -> bool:
    """
    Verify a Sumcheck inner-product proof.

    The verifier:
      1. Re-derives all Fiat-Shamir challenges from the proof (no new randomness).
      2. Checks each round equation: g_i(0) + g_i(1) == C_{i-1}.
      3. Evaluates the multilinear extensions of q and v at the challenge point
         (O(d) work) and checks the final oracle equation.

    Returns True iff all checks pass.
    """
    if scale is None:
        scale = proof.get("scale", 65536)

    if q and isinstance(q[0], float):
        q_int = quantize(q, scale)
    else:
        q_int = [_m(x) for x in q]
    if v and isinstance(v[0], float):
        v_int = quantize(v, scale)
    else:
        v_int = [_m(x) for x in v]

    d   = proof["d"]
    ell = proof["ell"]
    H   = proof["H"]
    rounds = proof["rounds"]

    if len(rounds) != ell:
        return False

    n = 1 << ell
    Q: List[int] = q_int + [0] * (n - d)
    V: List[int] = v_int + [0] * (n - d)

    # Re-derive challenges (deterministic, same as prover)
    transcript: bytes = H.to_bytes(8, "big")
    challenges: List[int] = []
    for g0, g1, g2 in rounds:
        r, transcript = _fs_challenge(transcript, g0, g1, g2)
        challenges.append(r)

    # Round-by-round check
    C: int = H
    for (g0, g1, g2), r in zip(rounds, challenges):
        if _m(g0 + g1) != _m(C):
            return False                    # consistency check failed
        C = _poly2_eval(g0, g1, g2, r)

    # Final oracle check: C_ell == MLE_q(r*) · MLE_v(r*)
    for r in challenges:
        half = len(Q) >> 1
        Q = [_m(Q[j] + _m(r * _m(Q[half + j] - Q[j]))) for j in range(half)]
        V = [_m(V[j] + _m(r * _m(V[half + j] - V[j]))) for j in range(half)]

    oracle = _m(Q[0] * V[0])
    if oracle != _m(C):
        return False

    # Prover's declared final values must match
    if proof["final_q"] != Q[0] or proof["final_v"] != V[0]:
        return False

    return True


# ── Ranking Proof ──────────────────────────────────────────────────────────────

def prove_ranking(scores: List[int]) -> dict:
    """
    Prove  s₁ ≥ s₂ ≥ … ≥ sₖ  by revealing non-negative deltas.

    delta_i = s_i − s_{i+1} ≥ 0  is the explicit non-negativity witness.
    Any verifier can check: s_i − s_{i+1} == delta_i  AND  delta_i ≥ 0.

    Note: scores here are the claimed integer inner products (H values from
    Sumcheck proofs, converted back to signed ints for comparison).
    """
    deltas: List[int] = []
    for i in range(len(scores) - 1):
        delta = scores[i] - scores[i + 1]
        if delta < 0:
            raise ValueError(
                f"Scores not sorted at position {i}: "
                f"s[{i}]={scores[i]} < s[{i+1}]={scores[i+1]}"
            )
        deltas.append(delta)
    return {"scores": list(scores), "deltas": deltas}


def verify_ranking(scores: List[int], proof: dict) -> bool:
    """Verify the ranking proof."""
    if proof.get("scores") != scores:
        return False
    deltas = proof.get("deltas", [])
    if len(deltas) != len(scores) - 1:
        return False
    return all(
        d >= 0 and scores[i] - scores[i + 1] == d
        for i, d in enumerate(deltas)
    )


# ── Combined: prove/verify full retrieval result ───────────────────────────────

def _field_to_signed(h: int, p: int = P) -> int:
    """Convert a field element to a signed integer for score comparison."""
    return h if h <= p // 2 else h - p


def prove_retrieval(
    q_vec: List[float],
    corpus_vecs: List[List[float]],
    scale: int = 65536,
) -> dict:
    """
    For one query and its top-k retrieved vectors, generate:
      - One Sumcheck inner-product proof per (q, v_i) pair
      - One ranking proof that the scores are non-increasing

    The inner-product proofs are ~264 bytes each (d=2048, ell=11).
    The ranking proof is k−1 delta values (trivially small).

    Returns a single proof dict suitable for verify_retrieval().
    """
    ip_proofs: List[dict] = []
    scores: List[int] = []

    for v in corpus_vecs:
        proof = prove_inner_product(q_vec, v, scale=scale)
        ip_proofs.append(proof)
        # Convert field element H back to signed int for ordering comparison
        scores.append(_field_to_signed(proof["H"]))

    rank_proof = prove_ranking(scores)

    return {
        "k": len(corpus_vecs),
        "scale": scale,
        "ip_proofs": ip_proofs,
        "rank_proof": rank_proof,
    }


def verify_retrieval(
    q_vec: List[float],
    corpus_vecs: List[List[float]],
    proof: dict,
) -> bool:
    """
    Verify all inner-product proofs and the ranking proof.
    Returns True only if every check passes.
    """
    scale = proof.get("scale", 65536)
    ip_proofs = proof.get("ip_proofs", [])
    rank_proof = proof.get("rank_proof", {})

    if len(ip_proofs) != len(corpus_vecs):
        return False

    scores: List[int] = []
    for v, ip_proof in zip(corpus_vecs, ip_proofs):
        if not verify_inner_product(q_vec, v, ip_proof, scale=scale):
            return False
        scores.append(_field_to_signed(ip_proof["H"]))

    return verify_ranking(scores, rank_proof)


# ── Global Batch Proof ─────────────────────────────────────────────────────────
#
# Proves ALL N inner products simultaneously with a SINGLE Sumcheck instance.
#
# Technique: random linear combination (Schwartz-Zippel)
#   1. Prover announces all N scores  s_i = q · v_i
#   2. Fiat-Shamir derives ρ  from the announced scores
#   3. Prover computes aggregated vector  w = Σᵢ ρⁱ · vᵢ
#   4. Prover runs one Sumcheck to prove  s_batch = q · w
#      where  s_batch = Σᵢ ρⁱ · sᵢ  (checkable by Verifier from announced scores)
#   5. If the Sumcheck passes, all N announced scores are correct
#      (soundness error: N/p ≈ 2^{-52})
#
# Verifier then independently sorts the N proven scores and selects top-k.
# No "ranking proof" needed — Verifier owns the entire score list.
#
# Proof size: N × 8B (scores) + 264B (one Sumcheck) ≈ 2.3 KB  for N=288
# Compare to local mode: k × 264B + ranking = ~1.4 KB  (but no global guarantee)

def prove_global_batch(
    q_vec: List[float],
    all_corpus_vecs: List[List[float]],
    scale: int = 65536,
    ipa_mode: bool = False,
    pp_path: Optional[str] = None,
    workdir: Optional[str] = None,
    pp_generators=None,
) -> dict:
    """
    Batch Sumcheck proof certifying ALL N inner products in one shot.

    ipa_mode=True:
      - Switches to BLS12-381 Fr field (P_FR ≈ 2^255) for soundness error N/2^255
      - After Sumcheck, generates an IPA oracle opening proof for w at r*
      - Includes oracle_proof (bytes) in the returned dict
      - Requires pp_path (embedding public params) and workdir (temp file dir)

    Returns dict with:
        type         – "global"
        N            – corpus size
        scale        – quantization scale
        scores       – List[int] of N field elements  (all inner products mod P or P_FR)
        rho          – Fiat-Shamir challenge (int)
        s_batch      – Σᵢ ρⁱ · sᵢ  (field element)
        sc_proof     – single Sumcheck proof for  q · w
        ipa_mode     – bool, whether IPA oracle proof is included
        oracle_proof – (ipa_mode only) bytes, IPA opening proof for w at r*
    """
    Pf = P_FR if ipa_mode else P
    mf = _mfr if ipa_mode else _m
    fs = _fs_challenge_fr if ipa_mode else _fs_challenge
    p2e = _poly2_eval_fr if ipa_mode else _poly2_eval
    score_bytes = 32 if ipa_mode else 8

    def quantize_vec(v):
        if v and isinstance(v[0], float):
            return [mf(int(round(x * scale))) for x in v]
        return [mf(x) for x in v]

    q_int = quantize_vec(q_vec)
    d = len(q_int)
    N = len(all_corpus_vecs)

    all_v_int: List[List[int]] = []
    scores: List[int] = []
    for v in all_corpus_vecs:
        v_int = quantize_vec(v)
        all_v_int.append(v_int)
        s = mf(sum(mf(q_int[j] * v_int[j]) for j in range(d)))
        scores.append(s)

    # Fiat-Shamir: derive ρ from all announced scores
    rho: int = int.from_bytes(
        hashlib.sha256(b"".join(s.to_bytes(score_bytes, "big") for s in scores)).digest(),
        "big",
    ) % Pf

    # Aggregated vector  w = Σᵢ ρⁱ · vᵢ
    w: List[int] = [0] * d
    rho_pow: int = 1
    for v_int in all_v_int:
        for j in range(d):
            w[j] = mf(w[j] + mf(rho_pow * v_int[j]))
        rho_pow = mf(rho_pow * rho)

    # s_batch = Σᵢ ρⁱ · sᵢ
    rho_pow = 1
    s_batch: int = 0
    for s in scores:
        s_batch = mf(s_batch + mf(rho_pow * s))
        rho_pow = mf(rho_pow * rho)

    # Single Sumcheck over w (inline for field flexibility)
    ell = max(1, math.ceil(math.log2(d))) if d > 1 else 1
    n = 1 << ell
    Q: List[int] = q_int + [0] * (n - d)
    W: List[int] = w + [0] * (n - d)

    H: int = mf(sum(mf(Q[j] * W[j]) for j in range(n)))
    assert H == s_batch, f"Internal: batch H={H} != s_batch={s_batch}"

    transcript: bytes = H.to_bytes(score_bytes, "big")
    rounds: List[Tuple[int, int, int]] = []
    challenges: List[int] = []

    for _ in range(ell):
        half = len(Q) >> 1
        g0 = mf(sum(mf(Q[j] * W[j]) for j in range(half)))
        g1 = mf(sum(mf(Q[half + j] * W[half + j]) for j in range(half)))
        g2 = mf(sum(mf(mf(2 * Q[half + j] - Q[j]) * mf(2 * W[half + j] - W[j])) for j in range(half)))
        rounds.append((g0, g1, g2))
        r, transcript = fs(transcript, g0, g1, g2)
        challenges.append(r)
        Q = [mf(Q[j] + mf(r * mf(Q[half + j] - Q[j]))) for j in range(half)]
        W = [mf(W[j] + mf(r * mf(W[half + j] - W[j]))) for j in range(half)]

    sc_proof = {
        "H": H, "ell": ell, "d": d, "scale": 1,
        "rounds": [(int(g0), int(g1), int(g2)) for g0, g1, g2 in rounds],
        "final_q": int(Q[0]), "final_v": int(W[0]),
        "ipa_mode": ipa_mode,
    }

    result = {
        "type": "global",
        "N": N,
        "scale": scale,
        "scores": scores,
        "rho": rho,
        "s_batch": s_batch,
        "sc_proof": sc_proof,
        "ipa_mode": ipa_mode,
    }

    if ipa_mode:
        oracle_proof = _generate_oracle_proof(
            w_int=w,
            challenges=challenges,
            pp_path=pp_path,
            workdir=workdir,
            pp_generators=pp_generators,
        )
        result["oracle_proof"] = oracle_proof

    return result


def verify_global_batch(
    q_vec: List[float],
    all_corpus_vecs: Optional[List[List[float]]],
    proof: dict,
    top_k: int = 5,
    commitment_path: Optional[str] = None,
) -> dict:
    """
    Verify the global batch proof and return the independently-determined top-k.

    Steps:
      1. Re-derive ρ from the announced scores (Fiat-Shamir)
      2. In non-IPA mode: Reconstruct w = Σᵢ ρⁱ · vᵢ, verify Sumcheck q · w
      3. In IPA mode: Verify IPA oracle proof for w at r*; no raw vectors needed
      5. If all pass: sort scores, pick top-k independently

    Returns dict:
        verified        – bool
        top_k_indices   – List[int] of top-k corpus indices (highest score first)
        top_k_scores    – List[int] corresponding signed scores
        oracle_ok       – (ipa_mode) bool: IPA fold + binding passed
    """
    ipa_mode: bool = proof.get("ipa_mode", False)
    Pf = P_FR if ipa_mode else P
    mf = _mfr if ipa_mode else _m
    fs = _fs_challenge_fr if ipa_mode else _fs_challenge
    p2e = _poly2_eval_fr if ipa_mode else _poly2_eval
    score_bytes = 32 if ipa_mode else 8
    signed_field_to_int = lambda h: (h if h <= Pf // 2 else h - Pf)

    scale  = proof.get("scale", 65536)
    scores = proof.get("scores", [])
    rho    = proof.get("rho")
    s_batch_claimed = proof.get("s_batch")
    sc_proof = proof.get("sc_proof", {})
    N = proof.get("N", 0)

    FAIL = {"verified": False, "top_k_indices": [], "top_k_scores": [], "oracle_ok": None}

    if len(scores) != N:
        return FAIL

    def quantize_vec(v):
        if v and isinstance(v[0], float):
            return [mf(int(round(x * scale))) for x in v]
        return [mf(x) for x in v]

    q_int = quantize_vec(q_vec)
    d = len(q_int)

    # Re-derive ρ
    rho_check: int = int.from_bytes(
        hashlib.sha256(b"".join(s.to_bytes(score_bytes, "big") for s in scores)).digest(),
        "big",
    ) % Pf
    if rho_check != rho:
        return FAIL

    # Check s_batch == Σᵢ ρⁱ · sᵢ
    rho_pow = 1
    s_batch_check: int = 0
    for s in scores:
        s_batch_check = mf(s_batch_check + mf(rho_pow * s))
        rho_pow = mf(rho_pow * rho)
    if s_batch_check != s_batch_claimed:
        return FAIL

    if ipa_mode:
        # IPA path: verify Sumcheck and oracle proof without raw corpus vectors
        oracle_ok, w_final_oracle = _verify_global_sumcheck_ipa(
            q_int, sc_proof, commitment_path, proof.get("oracle_proof", b""),
            rho, scores, N, d, mf, fs, p2e, Pf, score_bytes,
        )
        if not oracle_ok:
            return {**FAIL, "oracle_ok": False}
    else:
        # Classic path: reconstruct w from raw vectors
        if len(all_corpus_vecs) != N:
            return FAIL
        w: List[int] = [0] * d
        rho_pow = 1
        for v in all_corpus_vecs:
            v_int = quantize_vec(v)
            for j in range(d):
                w[j] = mf(w[j] + mf(rho_pow * v_int[j]))
            rho_pow = mf(rho_pow * rho)

        if not verify_inner_product(q_int, w, sc_proof, scale=1):
            return FAIL
        oracle_ok = True

    signed = [signed_field_to_int(s) for s in scores]
    sorted_idx = sorted(range(N), key=lambda i: signed[i], reverse=True)
    top_idx = sorted_idx[:top_k]

    return {
        "verified": True,
        "top_k_indices": top_idx,
        "top_k_scores": [signed[i] for i in top_idx],
        "oracle_ok": oracle_ok,
    }


def _verify_global_sumcheck_ipa(
    q_int, sc_proof, commitment_path, oracle_proof_bytes,
    rho, scores, N, d, mf, fs, p2e, Pf, score_bytes,
) -> Tuple[bool, int]:
    """
    Verify the Sumcheck + IPA oracle for the global batch proof.
    Returns (all_ok, w_final) where w_final is the IPA-attested oracle value.
    """
    ell    = sc_proof["ell"]
    H      = sc_proof["H"]
    rounds = sc_proof["rounds"]

    if len(rounds) != ell:
        return False, 0

    n = 1 << ell
    Q: List[int] = q_int + [0] * (n - d)

    # Re-derive Sumcheck challenges and collect r*
    transcript: bytes = H.to_bytes(score_bytes, "big")
    challenges: List[int] = []
    for g0, g1, g2 in rounds:
        r, transcript = fs(transcript, g0, g1, g2)
        challenges.append(r)

    # Round-by-round Sumcheck consistency
    C: int = H
    for (g0, g1, g2), r in zip(rounds, challenges):
        if mf(g0 + g1) != mf(C):
            return False, 0
        C = p2e(g0, g1, g2, r)

    # Evaluate q MLE at r* (O(d) work)
    for r in challenges:
        half = len(Q) >> 1
        Q = [mf(Q[j] + mf(r * mf(Q[half + j] - Q[j]))) for j in range(half)]
    q_final = Q[0]

    # Verify IPA oracle proof
    ipa_result = verify_ipa_embedding(oracle_proof_bytes, commitment_path, rho, scores, N, mf, Pf)
    if not ipa_result["fold_ok"] or not ipa_result["binding_ok"]:
        return False, 0

    w_final = ipa_result["w_final"]

    # Final oracle check: C_ell == q_final * w_final (mod Pf)
    oracle_val = mf(q_final * w_final)
    if oracle_val != mf(C):
        return False, 0

    return True, w_final


# ── IPA Oracle Proof Utilities ──────────────────────────────────────────────────

def _read_fp_mont(data: bytes, offset: int) -> int:
    raw = int.from_bytes(data[offset:offset + 48], "little")
    return (raw * _R_FP_INV) % _P_FP


def _read_g1_jacobian(data: bytes, offset: int):
    """blstrs Jacobian (Montgomery coords) → py_ecc projective."""
    from py_ecc.optimized_bls12_381 import FQ, Z1
    X = _read_fp_mont(data, offset)
    Y = _read_fp_mont(data, offset + 48)
    Z = _read_fp_mont(data, offset + 96)
    if Z == 0:
        return Z1
    z_inv  = pow(Z, _P_FP - 2, _P_FP)
    z_inv2 = (z_inv * z_inv) % _P_FP
    z_inv3 = (z_inv2 * z_inv) % _P_FP
    return (FQ((X * z_inv2) % _P_FP), FQ((Y * z_inv3) % _P_FP), FQ(1))


def _read_fr_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 32], "little")


def _load_g1_commitments(commitment_path: str):
    """Load N×144-byte Jacobian G1 commitments from file."""
    with open(commitment_path, "rb") as f:
        data = f.read()
    n = len(data) // 144
    return [_read_g1_jacobian(data, i * 144) for i in range(n)]


def verify_ipa_embedding(
    oracle_proof: bytes,
    commitment_path: Optional[str],
    rho: int,
    scores: List[int],
    N: int,
    mf,
    Pf: int,
) -> dict:
    """
    Verify IPA oracle proof for the aggregated embedding vector w.

    Arguments:
        oracle_proof     – raw bytes of the IPA proof (from open-ipa binary)
        commitment_path  – path to embedding_commitments.bin (N×144 Jacobian G1)
        rho, scores, N   – Fiat-Shamir batch challenge and score announcement
        mf, Pf           – field reduction function and prime

    Returns dict with keys: fold_ok, binding_ok, w_final (int)
    """
    FAIL = {"fold_ok": False, "binding_ok": False, "w_final": 0}

    if not oracle_proof or len(oracle_proof) < 12:
        return FAIL

    # Parse proof binary
    magic = int.from_bytes(oracle_proof[0:4], "little")
    if magic != 0x49504100:
        return {**FAIL}
    k       = int.from_bytes(oracle_proof[4:8],   "little")
    com_log = int.from_bytes(oracle_proof[8:12],  "little")

    off = 12
    try:
        C_init = _read_g1_jacobian(oracle_proof, off); off += 144
        u_out  = [_read_fr_le(oracle_proof, off + i * 32) for i in range(com_log)]
        off += com_log * 32
        u_in   = [_read_fr_le(oracle_proof, off + i * 32) for i in range(k)]
        off += k * 32
        rounds = []
        for _ in range(k):
            L0 = _read_g1_jacobian(oracle_proof, off); off += 144
            L1 = _read_g1_jacobian(oracle_proof, off); off += 144
            rounds.append((L0, L1))
        g_final = _read_g1_jacobian(oracle_proof, off); off += 144
        w_final = _read_fr_le(oracle_proof, off)
    except Exception:
        return FAIL

    # Fold check (G1 arithmetic, Python)
    from py_ecc.optimized_bls12_381 import add, multiply, eq
    C = C_init
    for u, (L0, L1) in zip(u_in, rounds):
        omu  = (1 - u) % P_FR
        omu2 = (omu * omu) % P_FR
        uomu = (u * omu) % P_FR
        u2   = (u * u) % P_FR
        C = add(add(multiply(L0, omu2), multiply(C, uomu)), multiply(L1, u2))
    fold_ok = eq(C, multiply(g_final, w_final))

    # Binding check: C_init == cm_w = Σᵢ ρⁱ · cm_i
    binding_ok = False
    if commitment_path and os.path.exists(commitment_path):
        try:
            cms = _load_g1_commitments(commitment_path)
            from py_ecc.optimized_bls12_381 import Z1 as G1_ZERO
            cm_w = G1_ZERO
            rho_pow = 1
            for cm in cms:
                cm_w = add(cm_w, multiply(cm, rho_pow))
                rho_pow = (rho_pow * rho) % P_FR
            binding_ok = eq(C_init, cm_w)
        except Exception:
            binding_ok = False

    return {"fold_ok": fold_ok, "binding_ok": binding_ok, "w_final": w_final}


def _generate_oracle_proof(
    w_int: List[int],
    challenges: List[int],
    pp_path: Optional[str],
    workdir: Optional[str],
    pp_generators=None,
) -> bytes:
    """
    Generate IPA oracle opening proof for w at challenges r*.

    If pp_generators is provided (list of G1 points), uses pure-Python CPU path
    (generate_oracle_proof_python) without touching GPU.  Otherwise falls back to
    the ./open-ipa C++ GPU binary which requires pp_path and workdir.

    Temp files are written inside a private TemporaryDirectory so concurrent
    callers sharing the same workdir never collide, and all scratch files are
    cleaned up automatically even on exception.
    """
    if pp_generators is not None:
        return generate_oracle_proof_python(w_int, challenges, pp_generators)

    open_ipa = _BIN_DIR / "open-ipa"
    if not open_ipa.exists():
        raise FileNotFoundError(f"open-ipa binary not found at {open_ipa}")
    if not pp_path or not os.path.exists(pp_path):
        raise FileNotFoundError(f"pp_path not found: {pp_path}")

    import numpy as np

    # Always use a fresh temp dir — no name collisions, auto-cleaned on exit
    with tempfile.TemporaryDirectory(prefix="ipa_oracle_", dir=workdir) as td:
        td = Path(td)
        w_file     = str(td / "w.bin")
        u_file     = str(td / "u.bin")
        proof_file = str(td / "proof.bin")

        # Write w as Fr standard-form (32 bytes/element, little-endian uint256).
        # w_int values are P_FR-range after modular reduction — int32 would truncate.
        # FrTensor(filename) in C++ reads these as-is (standard form Fr_t).
        with open(w_file, "wb") as wf:
            for x in w_int:
                wf.write((x % P_FR).to_bytes(32, "little"))

        # Write Sumcheck challenges r* as standard-form Fr_t (32 bytes each).
        # REVERSED: Python folds MSB-first (high half split) but C++ me_open_step
        # folds LSB-first (adjacent pairs). Reversing here aligns both to the
        # same multilinear evaluation point.
        with open(u_file, "wb") as uf:
            for r in reversed(challenges):
                uf.write(r.to_bytes(32, "little"))

        result = subprocess.run(
            [str(open_ipa), pp_path, w_file, u_file, proof_file, str(len(w_int))],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"open-ipa failed (rc={result.returncode}): "
                + result.stderr.decode(errors="replace")[-300:]
            )

        with open(proof_file, "rb") as pf:
            return pf.read()
        # TemporaryDirectory context exit → w.bin, u.bin, proof.bin auto-deleted


# ── Pure-Python IPA prover (CPU fallback, no GPU binary) ──────────────────────

def _g1_to_jacobian_bytes(pt) -> bytes:
    """Convert py_ecc G1 projective point to 144-byte blstrs Jacobian (Montgomery coords).

    py_ecc uses PROJECTIVE coords (X:Y:Z) with affine = (X/Z, Y/Z), so we
    normalize to affine first (Z=1), then store in Montgomery form so that
    _read_g1_jacobian can decode back to the same point.
    """
    from py_ecc.optimized_bls12_381 import Z1
    if pt == Z1:
        return b"\x00" * 144
    x, y, z = pt
    # Normalize to affine: projective affine = (x/z, y/z)
    z_inv  = pow(z.n, _P_FP - 2, _P_FP)
    x_aff  = x.n * z_inv % _P_FP
    y_aff  = y.n * z_inv % _P_FP
    # Store in Montgomery form with Z=1 (= R_FP mod P_FP in Montgomery)
    xm = x_aff * _R_FP % _P_FP
    ym = y_aff * _R_FP % _P_FP
    zm = _R_FP % _P_FP        # 1 in Montgomery form
    return xm.to_bytes(48, "little") + ym.to_bytes(48, "little") + zm.to_bytes(48, "little")


def ipa_prove_python(
    w_int: List[int],
    generators,
    challenges_for_ipa: List[int],
) -> bytes:
    """
    Pure-Python IPA prover — equivalent to open-ipa C++ binary but runs on CPU.

    w_int               : list of d integers (standard-form mod P_FR)
    generators          : list of d G1 points (py_ecc projective)
    challenges_for_ipa  : IPA fold challenges in LSB-first order
                          = list(reversed(sumcheck_challenges))

    Returns proof bytes in the same format as open-ipa writes to disk, so the
    existing verify_ipa_embedding() Python verifier accepts the result without
    any changes.
    """
    from py_ecc.optimized_bls12_381 import add, multiply, Z1

    scalars = list(w_int)
    gens    = list(generators)

    # C_init = Σ_j G_j * w_j  (initial Pedersen commitment, proof[0])
    C_init = Z1
    for g, s in zip(gens, scalars):
        if s % P_FR != 0:
            C_init = add(C_init, multiply(g, int(s % P_FR)))

    k       = len(challenges_for_ipa)
    com_log = 0  # com.size = 1 → com_log = 0

    L0s: List = []
    L1s: List = []

    for u in challenges_for_ipa:
        new_size = (len(scalars) + 1) // 2
        new_scalars: List[int] = []
        new_gens    = []
        L0 = Z1   # Σ_j G[2j+1] * W[2j]   (temp0)
        L1 = Z1   # Σ_j G[2j]   * W[2j+1] (temp1)

        for gid in range(new_size):
            gid0 = 2 * gid
            gid1 = 2 * gid + 1

            s0 = int(scalars[gid0] % P_FR)
            s1 = int(scalars[gid1] % P_FR) if gid1 < len(scalars) else 0
            g0 = gens[gid0]
            g1 = gens[gid1] if gid1 < len(gens) else Z1

            # me_open_step: new_scalar = (1-u)*s0 + u*s1
            new_s = (s0 + u * (s1 - s0)) % P_FR
            new_scalars.append(int(new_s))

            # new_generator = u*g0 + (1-u)*g1
            new_g = add(multiply(g0, int(u % P_FR)), multiply(g1, int((1 - u) % P_FR)))
            new_gens.append(new_g)

            # L0 = Σ G[gid1] * s0,  L1 = Σ G[gid0] * s1
            if s0 != 0:
                L0 = add(L0, multiply(g1, s0))
            if s1 != 0:
                L1 = add(L1, multiply(g0, s1))

        L0s.append(L0)
        L1s.append(L1)
        scalars = new_scalars
        gens    = new_gens

    g_final = gens[0]
    w_final = int(scalars[0] % P_FR)

    # Serialize in open-ipa format
    magic = 0x49504100
    buf = struct.pack("<III", magic, k, com_log)
    buf += _g1_to_jacobian_bytes(C_init)
    # u_out (com_log=0): nothing
    # u_in: challenges_for_ipa (LSB-first, same as open-ipa reads them)
    for r in challenges_for_ipa:
        buf += (int(r) % P_FR).to_bytes(32, "little")
    for L0, L1 in zip(L0s, L1s):
        buf += _g1_to_jacobian_bytes(L0)
        buf += _g1_to_jacobian_bytes(L1)
    buf += _g1_to_jacobian_bytes(g_final)
    buf += w_final.to_bytes(32, "little")
    return buf


def generate_random_pp_python(d: int, seed: int = 0):
    """
    Generate d random G1 generators for a Pedersen commitment scheme.
    Returns a list of d py_ecc G1 projective points.
    """
    import random as _rnd
    from py_ecc.optimized_bls12_381 import G1, multiply
    rng = _rnd.Random(seed)
    pp = []
    for _ in range(d):
        r = rng.randint(1, P_FR - 1)
        pp.append(multiply(G1, r))
    return pp


def generate_oracle_proof_python(
    w_int: List[int],
    challenges: List[int],
    pp_generators,
) -> bytes:
    """
    CPU-only oracle proof generation using py_ecc G1 arithmetic.

    w_int        : aggregated embedding vector (d P_FR integers)
    challenges   : Sumcheck challenges in MSB-first order (Python convention)
    pp_generators: list of d G1 points (from generate_random_pp_python)

    challenges are reversed internally to match C++ me_open LSB-first convention.
    """
    challenges_for_ipa = list(reversed(challenges))
    return ipa_prove_python(w_int, pp_generators, challenges_for_ipa)

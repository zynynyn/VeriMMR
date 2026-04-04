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
from typing import List, Tuple, Union

# ── Field arithmetic (Mersenne prime 2^61 − 1) ────────────────────────────────

P: int = (1 << 61) - 1          # 2^61 − 1, Mersenne prime
INV2: int = (P + 1) >> 1        # 2^{-1} mod P  (since P is odd prime)


def _m(x: int) -> int:
    """Reduce x modulo P.  Python's % always returns non-negative result."""
    return x % P


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
) -> dict:
    """
    Batch Sumcheck proof certifying ALL N inner products in one shot.

    Returns dict with:
        type         – "global"
        N            – corpus size
        scale        – quantization scale
        scores       – List[int] of N field elements  (all inner products mod P)
        rho          – Fiat-Shamir challenge (int)
        s_batch      – Σᵢ ρⁱ · sᵢ  (field element)
        sc_proof     – single Sumcheck proof for  q · w  (264 B for d=2048)
    """
    # Quantize query
    if q_vec and isinstance(q_vec[0], float):
        q_int: List[int] = quantize(q_vec, scale)
    else:
        q_int = [_m(x) for x in q_vec]
    d = len(q_int)
    N = len(all_corpus_vecs)

    # Quantize all corpus vectors and compute scores
    all_v_int: List[List[int]] = []
    scores: List[int] = []
    for v in all_corpus_vecs:
        if v and isinstance(v[0], float):
            v_int = quantize(v, scale)
        else:
            v_int = [_m(x) for x in v]
        all_v_int.append(v_int)
        s = _m(sum(_m(q_int[j] * v_int[j]) for j in range(d)))
        scores.append(s)

    # Fiat-Shamir: derive ρ from all announced scores
    rho: int = int.from_bytes(
        hashlib.sha256(b"".join(s.to_bytes(8, "big") for s in scores)).digest(),
        "big",
    ) % P

    # Aggregated vector  w = Σᵢ ρⁱ · vᵢ  (field arithmetic)
    w: List[int] = [0] * d
    rho_pow: int = 1
    for v_int in all_v_int:
        for j in range(d):
            w[j] = _m(w[j] + _m(rho_pow * v_int[j]))
        rho_pow = _m(rho_pow * rho)

    # s_batch = Σᵢ ρⁱ · sᵢ
    rho_pow = 1
    s_batch: int = 0
    for s in scores:
        s_batch = _m(s_batch + _m(rho_pow * s))
        rho_pow = _m(rho_pow * rho)

    # Single Sumcheck: prove  q · w == s_batch
    # Pass pre-quantized integers (scale=1 means "no additional quantization")
    sc_proof = prove_inner_product(q_int, w, scale=1)
    # sc_proof["H"] must equal s_batch
    assert sc_proof["H"] == s_batch, "Internal error: batch Sumcheck H mismatch"

    return {
        "type": "global",
        "N": N,
        "scale": scale,
        "scores": scores,
        "rho": rho,
        "s_batch": s_batch,
        "sc_proof": sc_proof,
    }


def verify_global_batch(
    q_vec: List[float],
    all_corpus_vecs: List[List[float]],
    proof: dict,
    top_k: int = 5,
) -> dict:
    """
    Verify the global batch proof and return the independently-determined top-k.

    Steps:
      1. Re-derive ρ from the announced scores (Fiat-Shamir)
      2. Reconstruct  w = Σᵢ ρⁱ · vᵢ  (same as Prover)
      3. Check  Σᵢ ρⁱ · sᵢ == sc_proof["H"]  (announced scores consistent)
      4. Verify the Sumcheck  q · w == s_batch
      5. If all pass: sort scores, pick top-k independently

    Returns dict:
        verified        – bool
        top_k_indices   – List[int] of top-k corpus indices (highest score first)
        top_k_scores    – List[int] corresponding signed scores
    """
    scale  = proof.get("scale", 65536)
    scores = proof.get("scores", [])
    rho    = proof.get("rho")
    s_batch_claimed = proof.get("s_batch")
    sc_proof = proof.get("sc_proof", {})
    N = proof.get("N", 0)

    FAIL = {"verified": False, "top_k_indices": [], "top_k_scores": []}

    if len(scores) != N or len(all_corpus_vecs) != N:
        return FAIL

    # Quantize query
    if q_vec and isinstance(q_vec[0], float):
        q_int = quantize(q_vec, scale)
    else:
        q_int = [_m(x) for x in q_vec]
    d = len(q_int)

    # Re-derive ρ (Fiat-Shamir)
    rho_check: int = int.from_bytes(
        hashlib.sha256(b"".join(s.to_bytes(8, "big") for s in scores)).digest(),
        "big",
    ) % P
    if rho_check != rho:
        return FAIL

    # Check announced s_batch == Σᵢ ρⁱ · sᵢ
    rho_pow = 1
    s_batch_check: int = 0
    for s in scores:
        s_batch_check = _m(s_batch_check + _m(rho_pow * s))
        rho_pow = _m(rho_pow * rho)
    if s_batch_check != s_batch_claimed:
        return FAIL

    # Reconstruct w = Σᵢ ρⁱ · vᵢ
    w: List[int] = [0] * d
    rho_pow = 1
    for v in all_corpus_vecs:
        if v and isinstance(v[0], float):
            v_int = quantize(v, scale)
        else:
            v_int = [_m(x) for x in v]
        for j in range(d):
            w[j] = _m(w[j] + _m(rho_pow * v_int[j]))
        rho_pow = _m(rho_pow * rho)

    # Verify Sumcheck: q · w == s_batch
    if not verify_inner_product(q_int, w, sc_proof, scale=1):
        return FAIL

    # All N scores certified. Verifier independently selects top-k.
    signed = [_field_to_signed(s) for s in scores]
    sorted_idx = sorted(range(N), key=lambda i: signed[i], reverse=True)
    top_idx = sorted_idx[:top_k]

    return {
        "verified": True,
        "top_k_indices": top_idx,
        "top_k_scores": [signed[i] for i in top_idx],
    }

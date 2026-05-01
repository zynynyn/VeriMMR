#ifndef RESCALING_CUH
#define RESCALING_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh" 
#include "tlookup.cuh"
#include "proof.cuh"

class Rescaling {
public:
    uint scaling_factor;
    tLookupRange tl_rem; // table for remainder
    Rescaling decomp(const FrTensor& X, FrTensor& rem);
    FrTensor *rem_tensor_ptr;

    Rescaling(uint scaling_factor);
    FrTensor operator()(const FrTensor& X);
    // proof: sumcheck polynomials + final tLookup claim appended here.
    vector<Claim> prove(const FrTensor& X, const FrTensor& X_, vector<Polynomial>& proof);

    // Constraint Fusion (zkGPT §5): fused proof for X →[/sf]→ X_ →[/sf]→ X__.
    // Replaces two prove() tLookup range checks with one, halving that overhead.
    // Caller: invoke operator() on both rs_outer and *this before calling this.
    void prove_chain_with(const Rescaling& rs_outer,
                          const FrTensor& X, const FrTensor& X_, const FrTensor& X__,
                          vector<Polynomial>& proof);

    ~Rescaling();
};

#endif // RESCALING_CUH
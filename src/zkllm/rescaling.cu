#include "rescaling.cuh"

Rescaling::Rescaling(uint scaling_factor): scaling_factor(scaling_factor), tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor), rem_tensor_ptr(nullptr)
{
}

// void decomp(const FrTensor& X, FrTensor& sign, FrTensor& abs, FrTensor& rem, FrTensor& rem_ind);
KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr, long scaling_factor, uint N)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        long hsf = scaling_factor >> 1;
        long x = scalar_to_long(in_ptr[tid]);
        long temp = (x + hsf) % scaling_factor;
        long x_rem = (temp < 0 ? temp + scaling_factor : temp) - hsf;
        long x_rescaled = (x - x_rem) / scaling_factor;
        out_ptr[tid] = long_to_scalar(x_rescaled);
        rem_ptr[tid] = long_to_scalar(x_rem);
    }
}

FrTensor Rescaling::operator()(const FrTensor& X)
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor out(X.size);
    uint block_size = 256;
    rescaling_kernel<<<(X.size + block_size - 1) / block_size, block_size>>>(X.gpu_data, out.gpu_data, rem_tensor_ptr->gpu_data, scaling_factor, X.size);
    cudaDeviceSynchronize();

    return out;
}

Rescaling::~Rescaling()
{
    if (rem_tensor_ptr) delete rem_tensor_ptr;
}

vector<Claim> Rescaling::prove(const FrTensor& X, const FrTensor& X_, vector<Polynomial>& proof)
{
    if (X.size != X_.size)
        throw std::runtime_error("Error: the size of X and X_ should be the same.");

    auto u = random_vec(ceilLog2(X.size));
    auto v = random_vec(ceilLog2(X.size));
    auto rand_temp = random_vec(2);

    auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});
    auto m = tl_rem.prep(rem);

    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u))
        throw std::runtime_error("Error: the rem is not correct.");

    auto c = tl_rem.prove(rem, m, rand_temp[0], rand_temp[1], u, v, proof);
    proof.push_back(Polynomial(c));

    cout << "Rescaling proof complete." << endl;
    return {};
}

void Rescaling::prove_chain_with(const Rescaling& rs_outer,
                                  const FrTensor& X, const FrTensor& X_, const FrTensor& X__,
                                  vector<Polynomial>& proof)
{
    // Constraint Fusion (zkGPT §5.2): instead of two separate tLookup range checks
    //   rs_inner.prove(X_, X__)  — range-check rem_inner = X_ − X__ * sf
    //   rs_outer.prove(X,  X_)  — range-check rem_outer = X  − X_  * sf
    // we concatenate [rem_inner || rem_outer] and run ONE tLookup prove.
    // Both remainders live in [−sf/2, sf/2), same table → valid combined check.

    if (!this->rem_tensor_ptr || !rs_outer.rem_tensor_ptr)
        throw std::runtime_error(
            "prove_chain_with: call operator() on both rescalings before prove_chain_with");

    auto& rem_inner = *this->rem_tensor_ptr;
    auto& rem_outer = *rs_outer.rem_tensor_ptr;

    // Concatenate remainders: [rem_inner | rem_outer]
    FrTensor rem_all(2 * X.size);
    cudaMemcpy(rem_all.gpu_data,
               rem_inner.gpu_data, X.size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(rem_all.gpu_data + X.size,
               rem_outer.gpu_data, X.size * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    // Pad to next power-of-2 (consistent with prove())
    auto rem_padded = rem_all.pad({rem_all.size});
    auto m = tl_rem.prep(rem_padded);

    auto u   = random_vec(ceilLog2(rem_padded.size));
    auto v   = random_vec(ceilLog2(rem_padded.size));
    auto rnd = random_vec(2);
    auto c = tl_rem.prove(rem_padded, m, rnd[0], rnd[1], u, v, proof);
    proof.push_back(Polynomial(c));

    cout << "Rescaling chain proof complete (fused 2× rem)." << endl;
}

#include "commitment.cuh"
#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

// ── G1 MLE without unmont ─────────────────────────────────────────────────────
// G1_me_step (g1-tensor.cu) does blstrs__scalar__Scalar_unmont(u) before
// G1Jacobian_mul.  That would compute unmont(u)×G instead of u×G.
// Python _eval_g1_ml uses the raw integer bits of u directly for scalar mul,
// so we must also skip the unmont here to match Python's answer.
KERNEL void g1_mle_raw_step(const G1Jacobian_t* in, G1Jacobian_t* out,
                             Fr_t u, uint in_n, uint out_n)
{
    uint gid = GET_GLOBAL_ID();
    if (gid >= out_n) return;
    uint g0 = 2 * gid, g1 = 2 * gid + 1;
    if (g1 < in_n)
        out[gid] = blstrs__g1__G1Affine_add(in[g0],
                       G1Jacobian_mul(blstrs__g1__G1Affine_add(in[g1], G1Jacobian_minus(in[g0])), u));
    else if (g0 < in_n)
        out[gid] = blstrs__g1__G1Affine_add(in[g0], G1Jacobian_minus(G1Jacobian_mul(in[g0], u)));
    else
        out[gid] = blstrs__g1__G1Affine_ZERO;
}

// Returns the G1 MLE evaluation at u_out (raw bits, no unmont).
G1Jacobian_t g1_mle_raw(const G1TensorJacobian& com, const vector<Fr_t>& u_out)
{
    uint n = com.size;
    G1Jacobian_t* cur;
    cudaMalloc(&cur, n * sizeof(G1Jacobian_t));
    cudaMemcpy(cur, com.gpu_data, n * sizeof(G1Jacobian_t), cudaMemcpyDeviceToDevice);

    for (Fr_t u : u_out) {
        uint new_n = (n + 1) / 2;
        G1Jacobian_t* nxt;
        cudaMalloc(&nxt, new_n * sizeof(G1Jacobian_t));
        g1_mle_raw_step<<<(new_n + G1NumThread - 1) / G1NumThread, G1NumThread>>>(cur, nxt, u, n, new_n);
        cudaDeviceSynchronize();
        cudaFree(cur);
        cur = nxt;
        n = new_n;
    }
    G1Jacobian_t result;
    cudaMemcpy(&result, cur, sizeof(G1Jacobian_t), cudaMemcpyDeviceToHost);
    cudaFree(cur);
    return result;
}

// ── G1 equality check ─────────────────────────────────────────────────────────
// Two projective Jacobian points are equal iff their difference is the identity.
// In blstrs Jacobian coords, the identity has z = 0.
KERNEL void g1_eq_kernel(G1Jacobian_t a, G1Jacobian_t b, bool* out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    G1Jacobian_t d = blstrs__g1__G1Affine_add(a, G1Jacobian_minus(b));
    bool z_zero = true;
    for (int i = 0; i < 12; ++i)
        if (d.z.val[i] != 0) { z_zero = false; break; }
    *out = z_zero;
}

bool h_g1_eq(G1Jacobian_t a, G1Jacobian_t b)
{
    bool* d_res;
    cudaMalloc(&d_res, sizeof(bool));
    g1_eq_kernel<<<1, 1>>>(a, b, d_res);
    cudaDeviceSynchronize();
    bool h_res;
    cudaMemcpy(&h_res, d_res, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_res);
    return h_res;
}

// ── Fr standard-form arithmetic (for fold check scalars) ─────────────────────
// blstrs Scalar_mul is Montgomery mul; to multiply standard-form a×b mod p we
// convert both to Montgomery, multiply, then convert back.
DEVICE Fr_t fr_mul_std(Fr_t a, Fr_t b)
{
    Fr_t am = blstrs__scalar__Scalar_mont(a);
    Fr_t bm = blstrs__scalar__Scalar_mont(b);
    return blstrs__scalar__Scalar_unmont(blstrs__scalar__Scalar_mul(am, bm));
}

DEVICE Fr_t fr_sub_std(Fr_t a, Fr_t b)
{
    return blstrs__scalar__Scalar_sub(a, b);   // works in standard form
}

// ── IPA fold check kernel ─────────────────────────────────────────────────────
// C_new = (1-u)² L0 + u(1-u) C + u² L1   (k rounds)
// Final check: C_k == w_final × g_final
KERNEL void ipa_fold_kernel(G1Jacobian_t C_init,
                             const G1Jacobian_t* L0s, const G1Jacobian_t* L1s,
                             const Fr_t* u_in, G1Jacobian_t g_final, Fr_t w_final,
                             bool* result, uint k)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const Fr_t ONE = {1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    G1Jacobian_t C = C_init;
    for (uint i = 0; i < k; ++i) {
        Fr_t u   = u_in[i];
        Fr_t omu = fr_sub_std(ONE, u);
        Fr_t omu2  = fr_mul_std(omu, omu);
        Fr_t u_omu = fr_mul_std(u, omu);
        Fr_t u2    = fr_mul_std(u, u);
        C = blstrs__g1__G1Affine_add(
                blstrs__g1__G1Affine_add(G1Jacobian_mul(L0s[i], omu2),
                                          G1Jacobian_mul(C,      u_omu)),
                G1Jacobian_mul(L1s[i], u2));
    }
    G1Jacobian_t expected = G1Jacobian_mul(g_final, w_final);
    G1Jacobian_t diff = blstrs__g1__G1Affine_add(C, G1Jacobian_minus(expected));
    bool z_zero = true;
    for (int i = 0; i < 12; ++i)
        if (diff.z.val[i] != 0) { z_zero = false; break; }
    *result = z_zero;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Usage: verify-ipa <proof_file> <commitment_file>" << endl;
        return 2;
    }

    // ── Read proof file ──────────────────────────────────────────────────────
    ifstream f(argv[1], ios::binary);
    if (!f) { cerr << "Cannot open: " << argv[1] << endl; return 2; }

    uint32_t magic, k, com_log;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x49504100u) { cerr << "Bad magic in " << argv[1] << endl; return 2; }
    f.read(reinterpret_cast<char*>(&k), 4);
    f.read(reinterpret_cast<char*>(&com_log), 4);

    G1Jacobian_t C_init;
    f.read(reinterpret_cast<char*>(&C_init), sizeof(G1Jacobian_t));

    vector<Fr_t> u_out(com_log), u_in(k);
    for (auto& v : u_out) f.read(reinterpret_cast<char*>(&v), sizeof(Fr_t));
    for (auto& v : u_in)  f.read(reinterpret_cast<char*>(&v), sizeof(Fr_t));

    vector<G1Jacobian_t> L0s(k), L1s(k);
    for (uint i = 0; i < k; ++i) {
        f.read(reinterpret_cast<char*>(&L0s[i]), sizeof(G1Jacobian_t));
        f.read(reinterpret_cast<char*>(&L1s[i]), sizeof(G1Jacobian_t));
    }
    G1Jacobian_t g_final;
    Fr_t w_final;
    f.read(reinterpret_cast<char*>(&g_final), sizeof(G1Jacobian_t));
    f.read(reinterpret_cast<char*>(&w_final), sizeof(Fr_t));
    f.close();

    // ── Binding check ────────────────────────────────────────────────────────
    // Evaluate the commitment's G1 MLE at u_out using raw bits (matches Python).
    string com_path(argv[2]);
    G1TensorJacobian com(com_path);
    G1Jacobian_t C_expected = (com_log == 0) ? com(0u) : g1_mle_raw(com, u_out);
    bool binding_ok = h_g1_eq(C_init, C_expected);

    // ── Fold check ───────────────────────────────────────────────────────────
    G1Jacobian_t *d_L0s, *d_L1s;
    Fr_t *d_u_in;
    bool *d_fold_ok;
    cudaMalloc(&d_L0s,    k * sizeof(G1Jacobian_t));
    cudaMalloc(&d_L1s,    k * sizeof(G1Jacobian_t));
    cudaMalloc(&d_u_in,   k * sizeof(Fr_t));
    cudaMalloc(&d_fold_ok, sizeof(bool));
    cudaMemcpy(d_L0s,   L0s.data(),  k * sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L1s,   L1s.data(),  k * sizeof(G1Jacobian_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_in,  u_in.data(), k * sizeof(Fr_t),          cudaMemcpyHostToDevice);

    ipa_fold_kernel<<<1, 1>>>(C_init, d_L0s, d_L1s, d_u_in, g_final, w_final, d_fold_ok, k);
    cudaDeviceSynchronize();

    bool fold_ok;
    cudaMemcpy(&fold_ok, d_fold_ok, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_L0s); cudaFree(d_L1s); cudaFree(d_u_in); cudaFree(d_fold_ok);

    // ── Output ───────────────────────────────────────────────────────────────
    cout << "{\"fold_ok\":" << (fold_ok ? "true" : "false")
         << ",\"binding_ok\":" << (binding_ok ? "true" : "false") << "}" << endl;

    return (fold_ok && binding_ok) ? 0 : 1;
}

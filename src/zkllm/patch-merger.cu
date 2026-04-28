#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "tlookup.cuh"
#include <string>

// PatchMerger 证明
//
// 结构（jina-v4 / Qwen2.5-VL）：
//   input  (n_patches, vit_dim)
//   → RMSNorm(vit_dim) → normed (n_patches, vit_dim)
//   → spatial merge: reshape to (n_patches/4, merged_dim=vit_dim*4)
//   → fc1: Linear(merged_dim, merged_dim) + GELU
//   → fc2: Linear(merged_dim, out_dim)
//   → output (n_patches/4, out_dim)
//
// 调用方式：
//   patch-merger <input_file> <n_patches> <vit_dim> <merged_dim> <out_dim>
//                <workdir> <prefix> <output_file>
//
// 权重文件命名：
//   workdir/patchmerger_layernorm.weight-pp.bin  (全局共用)
//   workdir/patchmerger_fc1.weight-pp.bin
//   workdir/patchmerger_fc2.weight-pp.bin
//   workdir/{prefix}-patchmerger_layernorm.weight-int.bin
//   workdir/{prefix}-patchmerger_layernorm.weight-commitment.bin
//   workdir/{prefix}-patchmerger_fc1.weight-int.bin
//   workdir/{prefix}-patchmerger_fc1.weight-commitment.bin
//   workdir/{prefix}-patchmerger_fc2.weight-int.bin
//   workdir/{prefix}-patchmerger_fc2.weight-commitment.bin
//
// GELU lookup 参数（与 gen_gelu_table.py 对应）：
//   low = -(1<<18), len = 1<<19, fc1_rescale = 1<<20
//   (适配 n_patches ≤ 256 即 D_padded = 2^19 ≤ table_size)

int main(int argc, char* argv[])
{
    string input_file = argv[1];
    uint n_patches    = std::stoi(argv[2]);
    uint vit_dim      = std::stoi(argv[3]);
    uint merged_dim   = std::stoi(argv[4]);   // = vit_dim * group_size (e.g. 5120)
    uint out_dim      = std::stoi(argv[5]);   // LLM embed dim (e.g. 2048)
    string workdir    = argv[6];
    string prefix     = argv[7];
    string output_file = argv[8];
    // argv[9]: optional rms_inv path (default: rms_inv_temp.bin)
    string rms_inv_path = (argc > 9) ? string(argv[9]) : "rms_inv_temp.bin";

    uint group_size    = merged_dim / vit_dim;   // 4 for jina-v4
    uint n_merged      = n_patches / group_size; // n_patches/4 output tokens

    // ── RMSNorm ──────────────────────────────────────────────────────────────
    auto norm_weight = create_weight(
        workdir + "/patchmerger_layernorm.weight-pp.bin",
        workdir + "/" + prefix + "-patchmerger_layernorm.weight-int.bin",
        workdir + "/" + prefix + "-patchmerger_layernorm.weight-commitment.bin",
        1, vit_dim
    );
    zkFC g(1, vit_dim, norm_weight.weight);
    Rescaling rs_norm1(1 << 16), rs_norm2(1 << 16);

    FrTensor X = FrTensor::from_int_bin(input_file);            // (n_patches × vit_dim)
    FrTensor rms_inv = FrTensor::from_int_bin(rms_inv_path);    // (n_patches,)

    auto g_inv_rms  = g(rms_inv);
    auto g_inv_rms_ = rs_norm1(g_inv_rms);
    auto normed     = g_inv_rms_ * X;
    auto normed_    = rs_norm2(normed);   // (n_patches × vit_dim)

    // ── Spatial merge: reshape (n_patches, vit_dim) → (n_merged, merged_dim) ─
    // The FrTensor is already laid out contiguously, so this is a free reshape.
    // No new kernel needed; just interpret the same buffer with different dims.
    // normed_.size == n_patches * vit_dim == n_merged * merged_dim ✓
    FrTensor& merged = normed_;   // alias: size = n_merged * merged_dim

    // ── FC1: Linear(merged_dim → merged_dim) + GELU ──────────────────────────
    auto fc1_weight = create_weight(
        workdir + "/patchmerger_fc1.weight-pp.bin",
        workdir + "/" + prefix + "-patchmerger_fc1.weight-int.bin",
        workdir + "/" + prefix + "-patchmerger_fc1.weight-commitment.bin",
        merged_dim, merged_dim
    );
    zkFC fc1(merged_dim, merged_dim, fc1_weight.weight);
    // Two-stage rescaling: 1<<16 then 1<<4 = total 1<<20, matching GELU table SCALE_IN=4096.
    // Single 1<<20 stage would require N=1<<20 > D_padded=524288, violating D%N==0.
    Rescaling fc1_rs_a(1 << 16);  // stage 1: scale 65536^2 → 65536
    Rescaling fc1_rs_b(1 << 4);   // stage 2: scale 65536 → 4096 = GELU SCALE_IN

    auto fc1_out  = fc1(merged);
    auto fc1_mid  = fc1_rs_a(fc1_out);   // at scale 65536
    auto fc1_out_ = fc1_rs_b(fc1_mid);   // at scale 4096 (fits GELU table range [-64,64])

    // GELU lookup: low=-(1<<18), len=1<<19
    // tLookupRangeMapping::prove handles non-power-of-2 D automatically.
    FrTensor gelu_values = FrTensor::from_int_bin("gelu-table.bin");
    tLookupRangeMapping gelu(-(1 << 18), 1 << 19, gelu_values);

    auto temp_rand = random_vec(3);
    auto gelu_u    = random_vec(ceilLog2(fc1_out_.size));
    auto gelu_v    = random_vec(ceilLog2(fc1_out_.size));
    vector<Polynomial> gelu_proof;

    auto p = gelu(fc1_out_);
    auto& gelu_out = p.first;
    auto& gelu_m   = p.second;

    // ── FC2: Linear(merged_dim → out_dim) ────────────────────────────────────
    auto fc2_weight = create_weight(
        workdir + "/patchmerger_fc2.weight-pp.bin",
        workdir + "/" + prefix + "-patchmerger_fc2.weight-int.bin",
        workdir + "/" + prefix + "-patchmerger_fc2.weight-commitment.bin",
        merged_dim, out_dim
    );
    zkFC fc2(merged_dim, out_dim, fc2_weight.weight);
    Rescaling fc2_rescale(1 << 16);

    auto fc2_out  = fc2(gelu_out);
    auto fc2_out_ = fc2_rescale(fc2_out);

    fc2_out_.save_int(output_file);

    // ── 证明 (逆序) ───────────────────────────────────────────────────────────
    fc2_rescale.prove(fc2_out, fc2_out_);
    verifyWeightClaim(fc2_weight, fc2.prove(gelu_out, fc2_out)[0],
        workdir + "/" + prefix + "-patchmerger_fc2-ipa-proof.bin");

    gelu.prove(fc1_out_, gelu_out, gelu_m,
               temp_rand[0], temp_rand[1], temp_rand[2],
               gelu_u, gelu_v, gelu_proof);
    cout << "GELU proof complete." << endl;

    fc1_rs_b.prove(fc1_mid, fc1_out_);
    fc1_rs_a.prove(fc1_out, fc1_mid);
    verifyWeightClaim(fc1_weight, fc1.prove(merged, fc1_out)[0],
        workdir + "/" + prefix + "-patchmerger_fc1-ipa-proof.bin");

    // ── RMSNorm 证明 ──────────────────────────────────────────────────────────
    rs_norm2.prove(normed, normed_);
    hadamard_product_sumcheck(g_inv_rms_, X,
        random_vec(ceilLog2(normed.size)), random_vec(ceilLog2(normed.size)));
    rs_norm1.prove(g_inv_rms, g_inv_rms_);
    verifyWeightClaim(norm_weight, g.prove(rms_inv, g_inv_rms)[0],
        workdir + "/" + prefix + "-patchmerger_layernorm-ipa-proof.bin");

    cout << "PatchMerger proof complete. ("
         << n_patches << " patches → " << n_merged << " tokens × " << out_dim << ")" << endl;
    return 0;
}

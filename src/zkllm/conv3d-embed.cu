#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

// Conv3d Patch Embedding 证明
//
// jina-v4 (Qwen2.5-VL) 的 ViT patch embedding：
//   Conv3d(in=3, out=1280, kernel=(2,14,14), stride=(2,14,14), bias=False)
//   因 stride = kernel（非重叠），等价于矩阵乘法：
//     X_flat (n_patches, patch_dim=1176) × W^T (patch_dim, out_dim=1280)
//
// 调用方式：
//   conv3d-embed <patches_file> <n_patches> <patch_dim> <out_dim> <workdir> <prefix> <output_file>
//
// 权重文件命名：
//   workdir/conv3d_embed.weight-pp.bin     (ppgen size=patch_dim)
//   workdir/{prefix}-conv3d_embed.weight-int.bin
//   workdir/{prefix}-conv3d_embed.weight-commitment.bin
//   → IPA proof: workdir/{prefix}-conv3d_embed-ipa-proof.bin

int main(int argc, char* argv[])
{
    string patches_file = argv[1];
    uint n_patches      = std::stoi(argv[2]);
    uint patch_dim      = std::stoi(argv[3]);   // 1176 = 3*2*14*14
    uint out_dim        = std::stoi(argv[4]);   // 1280
    string workdir      = argv[5];
    string prefix       = argv[6];
    string output_file  = argv[7];

    auto embed_weight = create_weight(
        workdir + "/conv3d_embed.weight-pp.bin",
        workdir + "/" + prefix + "-conv3d_embed.weight-int.bin",
        workdir + "/" + prefix + "-conv3d_embed.weight-commitment.bin",
        patch_dim, out_dim
    );

    zkFC embed_layer(patch_dim, out_dim, embed_weight.weight);
    Rescaling rescale(1 << 16);

    // patches: (n_patches, patch_dim) — Python 预先用 im2col 展开
    FrTensor patches = FrTensor::from_int_bin(patches_file);

    auto embed_out  = embed_layer(patches);
    auto embed_out_ = rescale(embed_out);

    embed_out_.save_int(output_file);

    { vector<Polynomial> rs_proof; rescale.prove(embed_out, embed_out_, rs_proof); }
    verifyWeightClaim(embed_weight, embed_layer.prove(patches, embed_out)[0],
        workdir + "/" + prefix + "-conv3d_embed-ipa-proof.bin");

    cout << "Conv3d embed proof complete. ("
         << n_patches << " patches × " << patch_dim << " → " << out_dim << ")" << endl;
    return 0;
}

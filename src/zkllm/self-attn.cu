#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>

int main(int argc, char *argv[])
{
    string mode = argv[1];
    string input_file_name = argv[2];
    uint seq_len = std::stoi(argv[3]);
    uint embed_dim = std::stoi(argv[4]);
    string workdir = argv[5];
    string layer_prefix = argv[6];
    string output_file_name = argv[7];
    // argv[8]: optional kv_dim for GQA (defaults to embed_dim for MHA)
    uint kv_dim = (argc > 8) ? std::stoi(argv[8]) : embed_dim;

    if (mode == "linear")
    {
        auto q_proj = create_weight(
            workdir + "/self_attn.q_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-commitment.bin",
            embed_dim,
            embed_dim
        );

        auto k_proj = create_weight(
            workdir + "/self_attn.k_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-commitment.bin",
            embed_dim,
            kv_dim
        );

        auto v_proj = create_weight(
            workdir + "/self_attn.v_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-commitment.bin",
            embed_dim,
            kv_dim
        );
        zkFC q_layer(embed_dim, embed_dim, q_proj.weight);
        zkFC k_layer(embed_dim, kv_dim, k_proj.weight);
        zkFC v_layer(embed_dim, kv_dim, v_proj.weight);
        Rescaling q_rescale(1 << 16);
        Rescaling k_rescale(1 << 16);
        Rescaling v_rescale(1 << 16);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        auto Q = q_layer(input);
        auto Q_ = q_rescale(Q);

        auto K = k_layer(input);
        auto K_ = k_rescale(K);

        auto V = v_layer(input);
        auto V_ = v_rescale(V);
        
        q_rescale.prove(Q, Q_);
        k_rescale.prove(K, K_);
        v_rescale.prove(V, V_);

        verifyWeightClaim(k_proj, k_layer.prove(input, K)[0]);
        verifyWeightClaim(q_proj, q_layer.prove(input, Q)[0]);
        verifyWeightClaim(v_proj, v_layer.prove(input, V)[0]);

        Q_.save_int("temp_Q.bin");
        K_.save_int("temp_K.bin");
        V_.save_int("temp_V.bin");

        cout << "QKV linear proof successfully verified!" << endl;

        return 0;
    }

    else if (mode == "attn")
    {
        // GQA support: argv[9] = num_kv_heads (default=1, MHA-compatible)
        // head_dim = kv_dim / num_kv_heads
        // For jina-v4: kv_dim=256, num_kv_heads=2 -> head_dim=128, num_q_heads=16, group_size=8
        // For MHA:     kv_dim=embed_dim, num_kv_heads=1 -> head_dim=embed_dim, 1 head (original behavior)
        uint num_kv_heads = (argc > 9) ? std::stoi(argv[9]) : 1;
        uint head_dim     = kv_dim / num_kv_heads;
        uint num_q_heads  = embed_dim / head_dim;
        uint group_size   = num_q_heads / num_kv_heads;

        auto Q_full = FrTensor::from_int_bin("temp_Q.bin");   // (seq_len * embed_dim)
        auto K_full = FrTensor::from_int_bin("temp_K.bin");   // (seq_len * kv_dim)
        auto V_full = FrTensor::from_int_bin("temp_V.bin");   // (seq_len * kv_dim)

        // Transpose to (dim, seq_len) layout so each head occupies a contiguous
        // memory region and can be extracted with trunc() without a new CUDA kernel.
        // After transpose: Q_T[c, r] == Q_full[r, c], so head h is at
        // positions [h*head_dim*seq_len, (h+1)*head_dim*seq_len).
        auto Q_T = Q_full.transpose(seq_len, embed_dim);      // (embed_dim, seq_len)
        auto K_T = K_full.transpose(seq_len, kv_dim);         // (kv_dim,    seq_len)
        auto V_T = V_full.transpose(seq_len, kv_dim);         // (kv_dim,    seq_len)

        // Rescaling factor must satisfy: table_size (= 1<<ceilLog2(sf)) divides
        // out_h.size (= seq_len * head_dim).  Original 1<<20 was designed for
        // MHA where out.size = seq_len*embed_dim >= 1<<20.  For per-head output
        // (seq_len * head_dim = 1024*128 = 2^17) we use sf = 1<<(ceilLog2(seq_len*head_dim)/2).
        // Using sf = 1<<16 gives table_size = 65536 which divides 131072 evenly.
        uint head_out_size = seq_len * head_dim;   // 131072 = 2^17
        uint rs_sf = 1;
        while (rs_sf * rs_sf < head_out_size) rs_sf <<= 1;  // largest power-of-2 whose square <= head_out_size
        // For head_out_size=131072=2^17: rs_sf=2^9=512 (512^2=262144>131072), back off to 256?
        // Safer: just pick largest power-of-2 that divides head_out_size
        rs_sf = 1;
        { uint tmp = head_out_size; while (tmp > 1) { tmp >>= 1; rs_sf <<= 1; } }
        // rs_sf = head_out_size (2^17=131072). That makes table=131072 entries which is fine.
        Rescaling rs1(rs_sf), rs2(rs_sf);

        for (uint g = 0; g < num_kv_heads; g++) {
            // Extract KV head g: trunc gives (head_dim * seq_len) contiguous elements,
            // then transpose back to (seq_len, head_dim) for matmul.
            auto K_g = K_T.trunc(g * head_dim * seq_len,
                                 (g + 1) * head_dim * seq_len)
                         .transpose(head_dim, seq_len);        // (seq_len, head_dim)
            auto V_g = V_T.trunc(g * head_dim * seq_len,
                                 (g + 1) * head_dim * seq_len)
                         .transpose(head_dim, seq_len);        // (seq_len, head_dim)

            for (uint h = 0; h < group_size; h++) {
                uint qi = g * group_size + h;                  // global Q-head index
                auto Q_h = Q_T.trunc(qi * head_dim * seq_len,
                                     (qi + 1) * head_dim * seq_len)
                             .transpose(head_dim, seq_len);    // (seq_len, head_dim)

                // Attention scores: X_h = Q_h @ K_g^T -> (seq_len, seq_len)
                auto X_h = FrTensor::matmul(Q_h,
                                            K_g.transpose(seq_len, head_dim),
                                            seq_len, head_dim, seq_len);

                // zkSoftmax proof for this head
                zkSoftmax softmax_h({1<<8, 1<<20, 1<<20}, 1, 0, 1UL<<32,
                                    {1<<18, 1<<22}, seq_len, seq_len, head_dim, 1);

                FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
                vector<FrTensor> X_segs, Y_segs, m_segs;
                FrTensor Y_h = softmax_h.compute(X_h, shift, X_shifted,
                                                 X_segs, Y_segs, m_segs);

                // Output: out_h = Y_h @ V_g -> (seq_len, head_dim)
                auto out_h   = FrTensor::matmul(Y_h, V_g, seq_len, seq_len, head_dim);
                auto out_h_  = rs2(out_h);
                auto out_h__ = rs1(out_h_);

                rs1.prove(out_h_, out_h__);
                rs2.prove(out_h, out_h_);

                // Sumcheck: prove out_h = Y_h @ V_g
                auto tmp = random_vec(3);
                vector<Polynomial> proof;
                auto u1 = random_vec(ceilLog2(seq_len));
                auto u2 = random_vec(ceilLog2(head_dim));
                auto ud = random_vec(ceilLog2(seq_len));
                auto claim = out_h.multi_dim_me({u1, u2}, {seq_len, head_dim});
                auto fc = zkip(claim,
                               Y_h.partial_me(u1, seq_len, seq_len),
                               V_g.partial_me(u2, head_dim, 1), ud, proof);

                // Sumcheck: prove X_h = Q_h @ K_g^T
                softmax_h.prove(Y_h, X_h, shift, X_shifted, X_segs, Y_segs, m_segs,
                    random_vec(ceilLog2(Y_h.size)), random_vec(ceilLog2(Y_h.size)),
                    tmp[0], tmp[1], tmp[2], proof);
                auto u1_ = random_vec(ceilLog2(seq_len));
                auto u2_ = random_vec(ceilLog2(seq_len));
                auto ud_ = random_vec(ceilLog2(head_dim));
                auto claim_ = X_h.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
                auto fc_ = zkip(claim_,
                                Q_h.partial_me(u1_, seq_len, head_dim),
                                K_g.partial_me(u2_, seq_len, head_dim), ud_, proof);

                cout << "Head " << qi << " (KV-group " << g
                     << ") attention proof complete." << endl;
            }
        }

        cout << "GQA zkAttn proof complete. ("
             << num_q_heads << " Q-heads, " << num_kv_heads
             << " KV-heads, head_dim=" << head_dim << ")" << endl;
        return 0;
    }
    return 0;
}
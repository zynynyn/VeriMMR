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
        
        // Batch tLookup prove for q/k/v rescaling remainders (all sf=1<<16).
        // One merged tLookup call replaces three separate ones.
        {
            uint qsz = Q.size, ksz = K.size, vsz = V.size;
            uint total = qsz + ksz + vsz;
            FrTensor all_rems(total);
            cudaMemcpy(all_rems.gpu_data,          q_rescale.rem_tensor_ptr->gpu_data,
                       qsz * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(all_rems.gpu_data + qsz,    k_rescale.rem_tensor_ptr->gpu_data,
                       ksz * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(all_rems.gpu_data + qsz + ksz, v_rescale.rem_tensor_ptr->gpu_data,
                       vsz * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            auto rem_p = all_rems.pad({total});
            Rescaling rs_b(1 << 16);
            auto m   = rs_b.tl_rem.prep(rem_p);
            auto u   = random_vec(ceilLog2(rem_p.size));
            auto v   = random_vec(ceilLog2(rem_p.size));
            auto rnd = random_vec(2);
            vector<Polynomial> rs_proof;
            rs_proof.push_back(Polynomial(rs_b.tl_rem.prove(rem_p, m, rnd[0], rnd[1], u, v, rs_proof)));
            cout << "Linear batch rescaling prove: q+k+v D=" << rem_p.size << endl;
        }

        // Batch prove for k/q/v: all three share the same input
        auto kqv_claims = zkFC::prove_batch(input,
            {{&k_layer, &K}, {&q_layer, &Q}, {&v_layer, &V}});
        verifyWeightClaim(k_proj, kqv_claims[0],
            workdir + "/" + layer_prefix + "-self_attn.k_proj-ipa-proof.bin");
        verifyWeightClaim(q_proj, kqv_claims[1],
            workdir + "/" + layer_prefix + "-self_attn.q_proj-ipa-proof.bin");
        verifyWeightClaim(v_proj, kqv_claims[2],
            workdir + "/" + layer_prefix + "-self_attn.v_proj-ipa-proof.bin");

        string tp = workdir + "/" + layer_prefix + "-";
        Q_.save_int(tp + "temp_Q.bin");
        K_.save_int(tp + "temp_K.bin");
        V_.save_int(tp + "temp_V.bin");

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
        // argv[10]/[11]: optional KV-group range [g_start, g_end) for parallel head splitting
        uint g_start = (argc > 10) ? std::stoi(argv[10]) : 0;
        uint g_end   = (argc > 11) ? std::stoi(argv[11]) : num_kv_heads;
        // argv[12]: n_wins > 0 → cross-window batch mode (eliminates per-window subprocess fork).
        // Each window's Q/K/V are read from {workdir}/{layer_prefix}-win{w}-temp_Q/K/V.bin.
        // n_wins == 0 → original single-window behaviour (load from {layer_prefix}-temp_Q/K/V.bin).
        uint n_wins = (argc > 12) ? std::stoi(argv[12]) : 0;

        string tp = workdir + "/" + layer_prefix + "-";

        // Rescaling factor: largest power-of-2 dividing (seq_len * head_dim).
        uint head_out_size = seq_len * head_dim;
        uint rs_sf = 1;
        { uint tmp = head_out_size; while (tmp % 2 == 0) { tmp >>= 1; rs_sf <<= 1; } }

        // Softmax parameters depend only on seq_len — precompute once.
        uint seq_sq = seq_len * seq_len;
        vector<uint>   softmax_bs;
        vector<double> softmax_thetas;
        if (seq_sq > (1U << 12)) {
            // Full attention (seq≥1024, e.g. 1024 or 4096): K=3, bs={256,2^20,2^20}.
            // bs={256,1M,1M} covers [0,2^32) for any seq: v2 actual range [0,16) << 1M table.
            // Avoids degenerate 16M-entry tables that the else branch would produce for seq=4096.
            softmax_bs     = {1U<<8, 1U<<20, 1U<<20};
            softmax_thetas = {double(1<<18), double(1<<22)};
        } else {
            // Window attention (seq≤64): K=4, 256×seq²³≥2^44 covers real activations.
            softmax_bs     = {256U, seq_sq, seq_sq, seq_sq};
            softmax_thetas = {double(1<<18), double(1<<22), double(1<<22)};
        }

        // Construct zkSoftmax ONCE: compute() and prove() do not mutate the table members
        // (tLookup::prep is stateless w.r.t. the table), so the same object is safe to
        // reuse across all total_wins × group_size × num_kv_heads head-level proofs.
        zkSoftmax softmax_shared(softmax_bs, 1, 0, 1UL<<32,
                                 softmax_thetas, seq_len, seq_len, head_dim, 1);

        uint total_wins = (n_wins == 0) ? 1u : n_wins;

        // Cross-head/window tLookup batch prove:
        // Accumulate all rescaling remainders into one tensor, then prove once.
        // For 640 heads (40 wins × 16 Q-heads): D_merged=2^23=8M, vs 640 × D=16K separately.
        // GPU occupancy: 3.1% per small kernel → ~100% for merged (39× fewer launches).
        uint per_head_rem     = seq_len * head_dim;  // = out_h.size
        uint total_q_in_range = (g_end - g_start) * group_size;
        uint total_rem_elems  = total_wins * total_q_in_range * 2 * per_head_rem;
        FrTensor all_rems(total_rem_elems);
        uint rem_offset = 0;
        // Shared rs1/rs2 — operator() only writes rem_tensor_ptr (overwritten each head), safe.
        Rescaling rs1(rs_sf), rs2(rs_sf);

        // Pre-allocate merged softmax tLookup segment tensors.
        // Each head contributes seq_sq = seq_len² elements per segment.
        // batch_prove_segs() proves all segments once after the head loop.
        uint per_head_seg   = seq_len * seq_len;
        uint total_segs_raw = total_wins * total_q_in_range * per_head_seg;
        uint K_sm = softmax_shared.get_K();
        uint L_sm = softmax_shared.get_L();
        // Guard: full-att blocks (seq=4096) → total_segs_raw=256M → 32GB OOM.
        // Window blocks (seq=64) → total_segs_raw=2.6M → safe.
        // Threshold 32M covers window + LLM-full-att (seq=1024, 16M), excludes jina full-att.
        const bool use_batch_segs = (total_segs_raw <= (1U << 25));
        vector<FrTensor> merged_X_segs_sm, merged_Y_segs_sm;
        if (use_batch_segs) {
            for (uint k = 0; k < K_sm; ++k) merged_X_segs_sm.emplace_back(total_segs_raw);
            for (uint k = L_sm; k < K_sm; ++k) merged_Y_segs_sm.emplace_back(total_segs_raw);
        }
        vector<uint> sm_x_off(K_sm, 0u), sm_y_off(K_sm - L_sm, 0u);

        for (uint w = 0; w < total_wins; w++) {
            // Load Q/K/V: window-batch mode reads per-window files split by verify_vit.py.
            string load_tp;
            if (n_wins == 0) {
                load_tp = tp;
            } else {
                load_tp = workdir + "/" + layer_prefix + "-win" + to_string(w) + "-";
            }
            auto Q_full = FrTensor::from_int_bin(load_tp + "temp_Q.bin");
            auto K_full = FrTensor::from_int_bin(load_tp + "temp_K.bin");
            auto V_full = FrTensor::from_int_bin(load_tp + "temp_V.bin");

            // Transpose to (dim, seq_len) layout so each head is a contiguous region.
            auto Q_T = Q_full.transpose(seq_len, embed_dim);
            auto K_T = K_full.transpose(seq_len, kv_dim);
            auto V_T = V_full.transpose(seq_len, kv_dim);

            for (uint g = g_start; g < g_end; g++) {
                auto K_g = K_T.trunc(g * head_dim * seq_len,
                                     (g + 1) * head_dim * seq_len)
                             .transpose(head_dim, seq_len);
                auto V_g = V_T.trunc(g * head_dim * seq_len,
                                     (g + 1) * head_dim * seq_len)
                             .transpose(head_dim, seq_len);

                for (uint h = 0; h < group_size; h++) {
                    uint qi = g * group_size + h;
                    auto Q_h = Q_T.trunc(qi * head_dim * seq_len,
                                         (qi + 1) * head_dim * seq_len)
                                 .transpose(head_dim, seq_len);

                    auto X_h = FrTensor::matmul(Q_h,
                                                K_g.transpose(seq_len, head_dim),
                                                seq_len, head_dim, seq_len);

                    FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
                    vector<FrTensor> X_segs, Y_segs, m_segs;
                    FrTensor Y_h = softmax_shared.compute(X_h, shift, X_shifted,
                                                          X_segs, Y_segs, m_segs);

                    auto out_h   = FrTensor::matmul(Y_h, V_g, seq_len, seq_len, head_dim);
                    auto out_h_  = rs2(out_h);
                    auto out_h__ = rs1(out_h_);

                    // Accumulate softmax tLookup segments for batch prove.
                    if (use_batch_segs) {
                        for (uint k = 0; k < K_sm; ++k) {
                            cudaMemcpy(merged_X_segs_sm[k].gpu_data + sm_x_off[k],
                                       X_segs[k].gpu_data,
                                       per_head_seg * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                            sm_x_off[k] += per_head_seg;
                        }
                        for (uint k = L_sm; k < K_sm; ++k) {
                            cudaMemcpy(merged_Y_segs_sm[k - L_sm].gpu_data + sm_y_off[k - L_sm],
                                       Y_segs[k - L_sm].gpu_data,
                                       per_head_seg * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                            sm_y_off[k - L_sm] += per_head_seg;
                        }
                    }

                    // Accumulate rem_inner (rs1) and rem_outer (rs2) into all_rems.
                    // Batch tLookup prove runs once after all heads (see below).
                    cudaMemcpy(all_rems.gpu_data + rem_offset,
                               rs1.rem_tensor_ptr->gpu_data,
                               per_head_rem * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                    rem_offset += per_head_rem;
                    cudaMemcpy(all_rems.gpu_data + rem_offset,
                               rs2.rem_tensor_ptr->gpu_data,
                               per_head_rem * sizeof(Fr_t), cudaMemcpyDeviceToDevice);
                    rem_offset += per_head_rem;

                    auto tmp = random_vec(3);
                    vector<Polynomial> proof;
                    auto u1 = random_vec(ceilLog2(seq_len));
                    auto u2 = random_vec(ceilLog2(head_dim));
                    auto ud = random_vec(ceilLog2(seq_len));
                    auto claim = out_h.multi_dim_me({u1, u2}, {seq_len, head_dim});
                    auto fc = zkip(claim,
                                   Y_h.partial_me(u1, seq_len, seq_len),
                                   V_g.partial_me(u2, head_dim, 1), ud, proof);

                    // Softmax tLookup segment proves deferred to batch_prove_segs() below.
                    softmax_shared.prove_no_segs(Y_h, X_h, shift, X_shifted, X_segs, Y_segs, m_segs,
                        random_vec(ceilLog2(Y_h.size)), random_vec(ceilLog2(Y_h.size)),
                        tmp[0], tmp[1], tmp[2], proof);
                    // Full-att fallback: D=seq² already large → prove segs per head immediately.
                    if (!use_batch_segs) {
                        vector<Polynomial> head_segs_proof;
                        softmax_shared.batch_prove_segs(X_segs, Y_segs, head_segs_proof);
                    }
                    auto u1_ = random_vec(ceilLog2(seq_len));
                    auto u2_ = random_vec(ceilLog2(seq_len));
                    auto ud_ = random_vec(ceilLog2(head_dim));
                    auto claim_ = X_h.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
                    auto fc_ = zkip(claim_,
                                    Q_h.partial_me(u1_, seq_len, head_dim),
                                    K_g.partial_me(u2_, seq_len, head_dim), ud_, proof);

                    cout << "Win " << w << " Head " << qi << " attn proof complete." << endl;
                }
            }
        }

        // ONE tLookup prove for all accumulated rescaling remainders (cross-head/window fusion).
        // D_merged = next_pow2(total_rem_elems); N = rs_sf (power of 2); D_merged % N == 0.
        {
            auto rem_padded = all_rems.pad({total_rem_elems});
            Rescaling rs_batch(rs_sf);
            auto m   = rs_batch.tl_rem.prep(rem_padded);
            auto u_b = random_vec(ceilLog2(rem_padded.size));
            auto v_b = random_vec(ceilLog2(rem_padded.size));
            auto rnd_b = random_vec(2);
            vector<Polynomial> proof_batch;
            auto rem_c = rs_batch.tl_rem.prove(rem_padded, m, rnd_b[0], rnd_b[1], u_b, v_b, proof_batch);
            proof_batch.push_back(Polynomial(rem_c));
            cout << "Batch tLookup rem: "
                 << total_wins << " wins × " << total_q_in_range << " heads × 2 rems"
                 << " → D=" << rem_padded.size << ", N=" << rs_sf
                 << ", rounds=" << ceilLog2(rem_padded.size) << "." << endl;
        }

        // Window blocks: ONE batch of K tLookup proves for all softmax segments.
        // Full-att blocks: segs proven per head above; skip here.
        if (use_batch_segs) {
            vector<Polynomial> sm_batch_proof;
            softmax_shared.batch_prove_segs(merged_X_segs_sm, merged_Y_segs_sm, sm_batch_proof);
            cout << "Batch softmax tLookup: "
                 << total_wins << " wins × " << total_q_in_range << " heads × "
                 << K_sm << " segs → D_raw=" << total_segs_raw
                 << " (pad→" << (1U << ceilLog2(total_segs_raw)) << ")." << endl;
        }

        cout << "GQA zkAttn proof complete. ("
             << num_q_heads << " Q-heads, " << num_kv_heads
             << " KV-heads, head_dim=" << head_dim
             << ", wins=" << total_wins
             << ", g=" << g_start << ".." << g_end << ")" << endl;
        return 0;
    }

    else if (mode == "o_proj")
    {
        // 证明 output projection: attn_out × W_o -> proj_out
        // argv[2]=input_file, argv[3]=seq_len, argv[4]=embed_dim,
        // argv[5]=workdir, argv[6]=layer_prefix, argv[7]=output_file
        auto o_proj = create_weight(
            workdir + "/self_attn.o_proj.weight-pp.bin",
            workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-int.bin",
            workdir + "/" + layer_prefix + "-self_attn.o_proj.weight-commitment.bin",
            embed_dim, embed_dim
        );
        zkFC o_layer(embed_dim, embed_dim, o_proj.weight);
        Rescaling o_rescale(1 << 16);

        FrTensor input = FrTensor::from_int_bin(input_file_name);
        auto O  = o_layer(input);
        auto O_ = o_rescale(O);

        { vector<Polynomial> rs_proof; o_rescale.prove(O, O_, rs_proof); }
        verifyWeightClaim(o_proj, o_layer.prove(input, O)[0],
            workdir + "/" + layer_prefix + "-self_attn.o_proj-ipa-proof.bin");

        O_.save_int(output_file_name);
        cout << "o_proj proof complete." << endl;
        return 0;
    }

    return 0;
}
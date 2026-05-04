#include "commitment.cuh"
#include "fr-tensor.cuh"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

// open-ipa: generate IPA oracle opening proof for a 1D vector
//
// Usage:
//   ./open-ipa <pp_file> <vec.bin> <u_file> <out_proof.bin> <vec_dim>
//
// <vec.bin>: auto-detected format by file size:
//   file_size == vec_dim * 4  → int32 format  (FrTensor::from_int_bin)
//   file_size == vec_dim * 32 → Fr standard form (FrTensor(filename), 32 B/elem)
//   Any other size is an error.
//
// Fr standard form: 8 × uint32 little-endian = BLS12-381 scalar field element.
// Used when the vector components are P_FR-range values (e.g. Sumcheck aggregation
// w = Σᵢ ρⁱ·vᵢ mod P_FR), which cannot be represented as int32.
// Both formats produce the same internal FrTensor standard-form representation
// and feed identically into commit_int / commitment.open.
//
// <u_file>: Sumcheck challenge r* (ell Fr_t values, 32 B each, standard form)
// <vec_dim>: embedding dimension (e.g. 2048)

int main(int argc, char* argv[])
{
    if (argc != 6) {
        cerr << "Usage: open-ipa <pp_file> <vec.bin> <u_file> <out_proof.bin> <vec_dim>" << endl;
        return 1;
    }

    string pp_file  = argv[1];
    string vec_file = argv[2];
    string u_file   = argv[3];
    string out_file = argv[4];
    uint   vec_dim  = (uint)stoi(argv[5]);

    // Detect input format from file size
    size_t file_sz = 0;
    {
        ifstream sz_f(vec_file, ios::ate | ios::binary);
        if (!sz_f) {
            cerr << "ERROR: cannot open vec_file: " << vec_file << endl;
            return 1;
        }
        file_sz = (size_t)sz_f.tellg();
    }

    FrTensor param_raw = (file_sz == (size_t)vec_dim * sizeof(int))
        ? FrTensor::from_int_bin(vec_file)   // int32: embeds as standard-form Fr_t
        : FrTensor(vec_file);                 // 32-byte Fr: loads raw Fr_t as-is

    Commitment generator(pp_file);
    auto param_padded = param_raw.pad({1u, vec_dim});
    auto com = generator.commit_int(param_padded);

    // Read Sumcheck challenge vector r* (standard-form Fr_t, 32 bytes each)
    vector<Fr_t> u_vec;
    {
        ifstream uf(u_file, ios::binary);
        if (!uf) {
            cerr << "ERROR: cannot open u_file: " << u_file << endl;
            return 1;
        }
        Fr_t v;
        while (uf.read(reinterpret_cast<char*>(&v), sizeof(Fr_t))) {
            u_vec.push_back(v);
        }
    }
    if (u_vec.empty()) {
        cerr << "ERROR: u_file is empty: " << u_file << endl;
        return 1;
    }

    generator.open(param_padded, com, u_vec, out_file);
    cout << "oracle proof written: " << out_file << endl;
    return 0;
}

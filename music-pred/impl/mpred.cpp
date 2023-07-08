#include <map>
#include <string>
#include <vector>
#include "hdc.h"
#include "ladispute-short.h"
#include "basic_encoding.h"

#define N 10240

BSC encode_relative_note(
    int *base_note,
    int *note
) {
    int base_note_id = base_note[1];
    if (note[0] > 1) {
        // chord
        std::vector<BSC&> note_vecs;
        for (int i = 0; i < note[0]; i++) {
            int diff = note[i + 1] - base_note_id;
            BSC note_vec = BSC(N, basis_melody[diff]);
        }
        BSC &res = note_vecs[0];
        for (int i = 1; i < note_vecs.size(); i++) {
            res = res * note_vecs[i];
        }
        return res;
    } else {
        // single note
        int diff = note[1] - base_note_id;
        BSC note_vec = BSC(N, basis_melody[diff]);
        return note_vec;
    }
}

BSC encode_note(
    int *note
) {
    if (note[0] > 1) {
        // chord
        std::vector<BSC&> note_vecs;
        for (int i = 0; i < note[0]; i++) {
            int diff = note[i + 1];
            BSC note_vec = BSC(N, basis_melody[diff]);
        }
        BSC &res = note_vecs[0];
        for (int i = 1; i < note_vecs.size(); i++) {
            res = res * note_vecs[i];
        }
        return res;
    } else {
        // single note
        int diff = note[1];
        BSC note_vec = BSC(N, basis_melody[diff]);
        return note_vec;
    }
}

BSC encode_note_hist(int *notes, unsigned int n) {
    std::vector<BSC> note_vecs;
    int *last_note = notes;
    for (int i = 1; i < n; i++) {
        int *note = last_note + 1 + last_note[0];
        BSC note_vec = encode_relative_note(
            notes,
            note
        );
        note_vecs.push_back(note_vec << (n - i - 1));
    }
    BSC &res = note_vecs[0];
    for (int i = 1; i < note_vecs.size(); i++) {
        res = res * note_vecs[i];
    }

}

void generate_song(
    std::map<std::string, BSC> &notemem,
    unsigned int top_k,
    unsigned int n_gen_notes
) {
    
}
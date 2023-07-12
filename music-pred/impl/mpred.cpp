#include <map>
#include <string>
#include <vector>
#include "hdc.h"
#include "ladispute-short.h"
#include "basic_encoding.h"
#include "itemmem.h"
#include <chrono>

#define N 10240

BSC encode_relative_note(
    int *base_note,
    int *note
) {
    int base_note_id = base_note[1];
    if (note[0] > 1) {
        // chord
        std::vector<BSC> note_vecs;
        for (int i = 0; i < note[0]; i++) {
            int diff = note[i + 1] - base_note_id;
            BSC note_vec = BSC(N, basis_melody[diff]);
            note_vecs.push_back(note_vec);
        }
        BSC res = note_vecs[0];
        for (size_t i = 1; i < note_vecs.size(); i++) {
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
        std::vector<BSC> note_vecs;
        for (int i = 0; i < note[0]; i++) {
            int pitch = (note[i + 1] + 3) % 12;
            BSC note_vec = BSC(N, basis_pitch[pitch]);
            note_vecs.push_back(note_vec);
        }
        BSC res = note_vecs[0];
        for (size_t i = 1; i < note_vecs.size(); i++) {
            res = res * note_vecs[i];
        }
        return res;
    } else {
        // single note
        int pitch = (note[1] + 3) % 12;
        BSC note_vec = BSC(N, basis_pitch[pitch]);
        return note_vec;
    }
}

BSC encode_note_hist(int *notes, int n) {
    std::vector<BSC> note_vecs;
    int *last_note = notes;
    for (int i = 1; i < n; i++) {
        int *note = last_note + 1 + last_note[0];
        BSC note_vec = encode_relative_note(
            notes,
            note
        );
        note_vecs.push_back(note_vec << (n - i - 1));
        last_note = note;
    }
    BSC res = note_vecs[0];
    for (size_t i = 1; i < note_vecs.size(); i++) {
        res = res * note_vecs[i];
    }
    BSC notehv = encode_note(notes);
    return res * notehv;
}

int itemmem_lookup(BSC &note_hist) {
    int min_dist = N;
    int min_id = -1;
    for (auto &item : note_itemmem) {
        BSC mem = BSC(N, item.second);
        int dist = note_hist.hamming_distance(mem);
        if (dist < min_dist) {
            min_dist = dist;
            min_id = item.first;
        }
    }
    printf("min_dist: %d\n", min_dist);
    printf("min_id: %d\n", min_id);
    // printf("note_hist:\n");
    // note_hist.print();
    return min_id;
}

int next_note(int *s) {
    return 1 + s[0];
}

void generate_song(
    unsigned int top_k,
    unsigned int n_user_notes,
    unsigned int n_gen_notes
) {
    size_t stride = 5;
    size_t idx = 0;
    size_t correct = 0;
    size_t total_pred = 0;
    printf("Generating song...\n");
    size_t avg_lat = 0;
    // iterate over song notes
    while (idx < song_len) {
        // skip user notes
        for (size_t i = 0; i < n_user_notes; i++) {
            idx += next_note(song + idx);
            if (idx >= song_len) {
                break; // end of song
            }
        }
        if (idx >= song_len) {
            break; // end of song
        }

        printf("Generating note %lu: [%d, %d]\n", idx, song[idx], song[idx + 1]);

        // generate n_gen_notes notes
        for (size_t i = 0; i < n_gen_notes; i++) {
            // get original note
            size_t orig_note_idx = idx;
            for (size_t j = 0; j < stride; j++) {
                orig_note_idx += next_note(song + orig_note_idx);
                if (orig_note_idx >= song_len) {
                    break; // end of song
                }
            }
            int *orig_note = song + orig_note_idx;

            // latency measurement
            auto start = std::chrono::high_resolution_clock::now();

            // encode note history
            BSC note_hist = encode_note_hist(song + idx, stride);
            // printf("Original note: %lu [%d, %d]\n", orig_note_idx, song[orig_note_idx], song[orig_note_idx + 1]);
            // lookup in itemmem
            int top_note = itemmem_lookup(note_hist);
            // BSC fifty_five = BSC(N, note_itemmem[55]);
            // int top_note = itemmem_lookup(fifty_five);

            // latency measurement
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            printf("Latency: %lu us\n", duration.count());
            avg_lat += duration.count();
            
            int orig_note_id = orig_note[1];
            correct += (top_note == orig_note_id);
            total_pred++;
            printf("Top note: %d, orig note: %d\n", top_note, orig_note_id);
            // update idx
            idx += next_note(&song[idx]);
        }
    }
    printf("Correct: %lu\n", correct);
    printf("Total pred: %lu\n", total_pred);
    printf("Accuracy: %f\n", (double)correct / total_pred);
    printf("Average latency: %lu us\n", avg_lat / total_pred);
}

int main() {
    generate_song(
        1,
        6,
        1
    );
    // BSC basis = BSC(N, basis_pitch[0]);
    // basis.print();
    // printf("permute by 1\n");
    // basis = basis << 1;
    // basis.print();
    // printf("permute by 2\n");
    // basis = basis << 1;
    // basis.print();
    return 0;
}
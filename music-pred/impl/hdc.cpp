#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <assert.h>
#include <immintrin.h>
#include <bit>
#include "hdc.h"

BSC::BSC(unsigned int dim, bool rand_init) {
    if (dim % 256 != 0) {
        printf("Error: dim must be a multiple of 256 for efficiency considerations\n");
        exit(1);
    }
    this->dim = dim;
    this->words = dim / 32;
    this->vec_size = dim / 256;
    this->data = new unsigned int[this->words];
    if (rand_init) {
        this->rand_init();
    }
}

BSC::~BSC() {
    delete[] this->data;
}

void BSC::rand_init() {
    for (unsigned int i = 0; i < this->words; i++) {
        this->data[i] = 0;
        this->data[i] |= rand() % (1 << 8);
        this->data[i] |= (rand() % (1 << 8)) << 8;
        this->data[i] |= (rand() % (1 << 8)) << 16;
        this->data[i] |= (rand() % (1 << 8)) << 24;
    }
}

BSC BSC::copy() {
    BSC bsc = BSC(this->dim);
    std::memcpy(this->data, bsc.data, this->words * sizeof(unsigned int));
    return bsc;
}

BSC BSC::permute(unsigned int perm) {
    BSC bsc = BSC(this->dim);
    // bit shift left by perm
    unsigned int word_ofs = perm / 32;
    unsigned int bit_ofs = perm % 32;
    unsigned int bit_ofs_inv = 32 - bit_ofs;
    for (unsigned int i = 0; i < this->words; i++) {
        bsc.data[i] = this->data[(i + word_ofs) % this->words] << bit_ofs;
        bsc.data[i] |= this->data[(i + word_ofs + 1) % this->words] >> bit_ofs_inv;
    }
    return bsc;
}

BSC BSC::bind(BSC &other) {
    assert(this->dim == other.dim);
    BSC res = BSC(this->dim);
    for (unsigned int i = 0; i < this->vec_size; i++) {
        __m256i vec_a = _mm256_loadu_si256((__m256i_u *) &this->data[i * 8]);
        __m256i vec_b = _mm256_loadu_si256((__m256i_u *) &other.data[i * 8]);
        __m256i vec_c = _mm256_xor_si256(vec_a, vec_b);
        _mm256_storeu_si256((__m256i_u *) &res.data[i * 8], vec_c);
    }
    return res;
}

unsigned int BSC::hamming_distance(BSC &other) {
    assert(this->dim == other.dim);
    unsigned int dist = 0;
    for (unsigned int i = 0; i < this->words; i++) {
        dist += std::__popcount(this->data[i] ^ other.data[i]);
    }
    return dist;
}

void BSC::print() {
    for (unsigned int i = 0; i < this->words; i++) {
        // print binary representation of this->data[i]
        for (unsigned int j = 0; j < 32; j++) {
            printf("%d", (this->data[i] >> (31 - j)) & 1);
        }
        printf(" ");
    }
    printf("\n");
}

#ifndef HDC_H
#define HDC_H

#define BITS_TO_BYTES(x) (((x) + 7) / 8)
#define BITS_TO_WORDS(x) (((x) + 31) / 32)

class BSC {
public:
    unsigned int dim;
    unsigned int words;
    unsigned int vec_size;
    unsigned int *data;

    BSC(unsigned int dim, bool rand_init = false);
    ~BSC();
    BSC copy();
    BSC permute(unsigned int perm);
    BSC bind(BSC &other);
    unsigned int hamming_distance(BSC &other);
    void print();

    BSC operator *(BSC &other) {
        return this->bind(other);
    }

    BSC operator <<(unsigned int perm) {
        return this->permute(perm);
    }

private:
    void rand_init();
};


#endif // HDC_H
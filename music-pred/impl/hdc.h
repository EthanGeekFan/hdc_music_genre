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

    /**
     * @brief Copy the vector
     * 
     * @return BSC copied vector
     */
    BSC copy();

    /**
     * @brief Permute the vector by perm bytes
     * 
     * @param perm 
     * @return BSC permuted vector
     */
    BSC permute(unsigned int perm);

    /**
     * @brief Bind with other vector
     * 
     * @param other 
     * @return BSC bound vector
     */
    BSC bind(BSC &other);

    /**
     * @brief Calculate the Hamming distance between two vectors
     * 
     * @param other 
     * @return unsigned int Hamming distance
     */
    unsigned int hamming_distance(BSC &other);

    /**
     * @brief Print the vector
     * 
     */
    void print();

    /**
     * @brief Binding operator
     * 
     */
    BSC operator *(BSC &other) {
        return this->bind(other);
    }

    /**
     * @brief Permutation operator
     * 
     */
    BSC operator <<(unsigned int perm) {
        return this->permute(perm);
    }

private:

    /**
     * @brief Initialize the vector with random values
     * 
     */
    void rand_init();
};


#endif // HDC_H
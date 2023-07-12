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

    BSC(unsigned int dim, unsigned int *data = NULL);
    ~BSC();

    /**
     * @brief Assign the vector with data
     * 
     * @param data 
     */
    void assign(unsigned int *data);

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

    /**
     * @brief Initialize the vector with random values
     * 
     */
    void rand_init();
    
private:

    /**
     * @brief Whether the vector needs to be freed
     * 
     */
    bool need_free;

};


#endif // HDC_H
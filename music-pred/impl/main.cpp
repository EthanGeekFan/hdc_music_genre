#include <stdio.h>
#include <stdlib.h>
#include "hdc.h"

int main() {
    int dim = 1024;
    srand(123);
    BSC a = BSC(dim);
    a.data[0] = 0b101;
    BSC b = BSC(dim);
    b.data[0] = 0b110;
    BSC c = a * b;
    a.print();
    printf("+\n");
    b.print();
    printf("=\n");
    c.print();
    printf("Permutation:\n");
    for (int i = 0; i < 32; i++) {
        BSC d = a << 32 + i;
        d.print();
    }
    printf("Hamming distance: %d\n", a.hamming_distance(b));
    printf("Random vector:\n");
    BSC e = BSC(dim, true);
    e.print();
    BSC zero = BSC(dim);
    BSC f = BSC(dim, true);
    printf("Hamming distance: %d\n", e.hamming_distance(f));
    return 0;
}
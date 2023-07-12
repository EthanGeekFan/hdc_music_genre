#include <stdio.h>
#include <stdlib.h>
#include "hdc.h"

int test() {
    int dim = 1024;
    srand(123);
    BSC a = BSC(dim);
    a.rand_init();
    // a.data[0] = 0b101;
    BSC b = BSC(dim);
    b.rand_init();
    // b.data[0] = 0b110;
    BSC c = a * b;
    a.print();
    printf("+\n");
    b.print();
    printf("=\n");
    c.print();
    BSC d = a * c;
    printf("a * c =\n");
    d.print();
    printf("should similar to b: %d\n", b.hamming_distance(d));
    printf("Permutation:\n");
    for (int i = 0; i < 4; i++) {
        BSC d = a << i;
        d.print();
    }
    printf("Hamming distance: %d\n", a.hamming_distance(b));
    printf("Random vector:\n");
    BSC e = BSC(dim);
    e.print();
    BSC zero = BSC(dim);
    BSC f = BSC(dim);
    printf("Hamming distance: %d\n", e.hamming_distance(f));
    return 0;
}

// int main() {
//     test();
//     return 0;
// }
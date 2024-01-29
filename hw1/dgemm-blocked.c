#include <stdlib.h>
#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 41
#endif

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

void micro_kernel(int lda, double *A, double *B, double *C){
    __m256d res0; __m256d res1; __m256d res2; __m256 res3;

    res0 = _mm256_load_pd(C + 0 * lda);
    for(int k = 0; k < 4; k++) res0 = _mm256_add_pd(res0, _mm256_mul_pd(_mm256_load_pd(D + k * lda), _mm256_load_pd(B + 0 * lda)));
    _mm256_store_pd(C + 0 * lda, res0);

    res1 = _mm256_load_pd(C + 1 * lda);
    for(int k = 0; k < 4; k++) res0 = _mm256_add_pd(res0, _mm256_mul_pd(_mm256_load_pd(D + k * lda), _mm256_load_pd(B + 1 * lda)));
    _mm256_store_pd(C + 1 * lda, res1);

    res2 = _mm256_load_pd(C + 2 * lda);
    for(int k = 0; k < 4; k++) res0 = _mm256_add_pd(res0, _mm256_mul_pd(_mm256_load_pd(D + k * lda), _mm256_load_pd(B + 2 * lda)));
    _mm256_store_pd(C + 2 * lda, res2);

    res3 = _mm256_load_pd(C + 3 * lda);
    for(int k = 0; k < 4; k++) res0 = _mm256_add_pd(res0, _mm256_mul_pd(_mm256_load_pd(D + k * lda), _mm256_load_pd(B + 3 * lda)));
    _mm256_store_pd(C + 3 * lda, res3);
}

/*
 * Copy the transpose of A to D
*/
void copy_transpose(int lda, double *A, double *D){
    for(int i = 0; i < lda; i++) for(int j = 0; j < lda; j++){
        D[j + i * lda] = A[i + j * lda];
    }
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    double *D = (*double) malloc(sizeof(double) * lda * lda);
    copy_transpose(lda, A, D);
    // For each block-row of A
    for (int i = 0; i < lda; i += KERNEL_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += KERNEL_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += KERNEL_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                // int M = min(BLOCK_SIZE, lda - i);
                // int N = min(BLOCK_SIZE, lda - j);
                // int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                micro_kernel(lda, D, B, C);
            }
        }
    }
    // // For each block-row of A
    // for (int i = 0; i < lda; i += BLOCK_SIZE) {
    //     // For each block-column of B
    //     for (int j = 0; j < lda; j += BLOCK_SIZE) {
    //         // Accumulate block dgemms into block of C
    //         for (int k = 0; k < lda; k += BLOCK_SIZE) {
    //             // Correct block dimensions if block "goes off edge of" the matrix
    //             int M = min(BLOCK_SIZE, lda - i);
    //             int N = min(BLOCK_SIZE, lda - j);
    //             int K = min(BLOCK_SIZE, lda - k);
    //             // Perform individual block dgemm
    //             do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
    //         }
    //     }
    // }
}

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef L2_SIZE
#define L2_SIZE 41
#endif

#ifndef L1_SIZE
#define L1_SIZE 20
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef double v4df __attribute__ ((vector_size (32), aligned(1)));

/* 
 *The micro kernel, which computes 4x4 block at a time. 
*/
void micro_kernel(int lda, int K, double *A, double *B, double *C){

    // Four columns
    double *c0 = C;
    double *c1 = C + lda;
    double *c2 = C + lda * 2;
    double *c3 = C + lda * 3;

    // Load each column with 4 elements
    v4df v0 = *(v4df*) c0;
    v4df v1 = *(v4df*) c1;
    v4df v2 = *(v4df*) c2;
    v4df v3 = *(v4df*) c3;

    // Compute
    for(int i = 0; i < K; i++){
        v4df a0 = *(v4df*) A;
        // v4df a0 = {A[0], A[lda], A[2 * lda], A[3 * lda]};
        v4df b0 = {B[0], B[0], B[0], B[0]};
        v4df b1 = {B[lda], B[lda], B[lda], B[lda]};
        v4df b2 = {B[lda * 2], B[lda * 2], B[lda * 2], B[lda * 2]};
        v4df b3 = {B[lda * 3], B[lda * 3], B[lda * 3], B[lda * 3]};
        A += lda; B += 1;

        v0 = v0 + a0 * b0;
        v1 = v1 + a0 * b1;
        v2 = v2 + a0 * b2;
        v3 = v3 + a0 * b3;
    }

    // Store
    *(v4df*) c0 = v0;
    *(v4df*) c1 = v1;
    *(v4df*) c2 = v2;
    *(v4df*) c3 = v3;
}

/*
 * Copy the transpose of A to D.
*/
void transpose(int lda, double *A, double *D){
    for(int i = 0; i < lda; ++i) for(int j = 0; j < lda; ++j){
        D[i + j * lda] = A[j + i * lda];
    }
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // Transpose A so that it's row-major
    // double *D = (double*) malloc(sizeof(double) * lda * lda);
    // transpose(lda, A, D);
    // A = D;

    // Perform micro kernel on 4x4 blocks
    for (int i = 0; i < M - M % 4; i += 4) {
        for (int j = 0; j < N - N % 4; j += 4) {
            // Compute C(i,j) with micro kernel
            micro_kernel(lda, K, A + i, B + j * lda, C + i + j * lda);
        }
    }
    // Process edge cases seperately
    if(M % 4 > 0){
        for(int i = M - M % 4; i < M; i++){
            for(int j = 0; j < N; j++){
                double cij = C[i + j * lda];
                for (int k = 0; k < K; ++k) {
                    cij += A[i + k * lda] * B[k + j * lda];
                }
                C[i + j * lda] = cij;
            }
        }
    }
    if(N % 4 > 0){
        for(int i = 0; i < M - M % 4; i++){
            for(int j = N - N % 4; j < N; j++){
                double cij = C[i + j * lda];
                for (int k = 0; k < K; ++k) {
                    cij += A[i + k * lda] * B[k + j * lda];
                }
                C[i + j * lda] = cij;
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {   
    // Loop reordering
    // For each block-row of A
    for (int k = 0; k < lda; k += L2_SIZE) {
        for (int i = 0; i < lda; i += L2_SIZE) {
            // For each block-column of B
            for (int j = 0; j < lda; j += L2_SIZE) {
                // Accumulate block dgemms into block of C
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(L2_SIZE, lda - i);
                int N = min(L2_SIZE, lda - j);
                int K = min(L2_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

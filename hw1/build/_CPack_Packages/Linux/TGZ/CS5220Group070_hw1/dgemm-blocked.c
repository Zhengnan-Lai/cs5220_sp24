#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef L2_SIZE
#define L2_SIZE 256

#endif

#ifndef L1_SIZE
#define L1_SIZE 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef double v4df __attribute__ ((vector_size (32), aligned(1)));

/* 
 *The micro kernel, which computes 8x4 block at a time. 
*/
void micro_kernel(int lda, int K, double *A, double *B, double *C){

    // Eight columns
    double *c0 = C;
    double *c4 = C + 4;
    double *c1 = C + lda;
    double *c5 = C + lda + 4;
    double *c2 = C + lda * 2;
    double *c6 = C + lda * 2 + 4;
    double *c3 = C + lda * 3;
    double *c7 = C + lda * 3 + 4;

    // Load each column with 4 elements
    v4df v0 = *(v4df*) c0;
    v4df v4 = *(v4df*) c4;
    v4df v1 = *(v4df*) c1;
    v4df v5 = *(v4df*) c5;
    v4df v2 = *(v4df*) c2;
    v4df v6 = *(v4df*) c6;
    v4df v3 = *(v4df*) c3;
    v4df v7 = *(v4df*) c7;

    // Compute
    for(int i = 0; i < K; i++){
        v4df a0 = *(v4df*) A;
        v4df a1 = *(v4df*) (A+4);
        A += lda; 

        v4df b0 = {B[0], B[0], B[0], B[0]};
        v4df b1 = {B[lda], B[lda], B[lda], B[lda]};
        v4df b2 = {B[lda * 2], B[lda * 2], B[lda * 2], B[lda * 2]};
        v4df b3 = {B[lda * 3], B[lda * 3], B[lda * 3], B[lda * 3]};
        B += 1;

        v0 = v0 + a0 * b0;
        v1 = v1 + a0 * b1;
        v2 = v2 + a0 * b2;
        v3 = v3 + a0 * b3;
        v4 = v4 + a1 * b0;
        v5 = v5 + a1 * b1;
        v6 = v6 + a1 * b2;
        v7 = v7 + a1 * b3;     
    }

    // Store
    *(v4df*) c0 = v0;
    *(v4df*) c1 = v1;
    *(v4df*) c2 = v2;
    *(v4df*) c3 = v3;
    *(v4df*) c4 = v4;
    *(v4df*) c5 = v5;
    *(v4df*) c6 = v6;
    *(v4df*) c7 = v7;
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block_L1(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // Transpose A so that it's row-major
    // double *D = (double*) malloc(sizeof(double) * lda * lda);
    // transpose(lda, A, D);
    // A = D;

    // Perform micro kernel on 4x4 blocks
    for (int i = 0; i < M - M % 8; i += 8) {
        for (int j = 0; j < N - N % 4; j += 4) {
            // Compute C(i,j) with micro kernel
            micro_kernel(lda, K, A + i, B + j * lda, C + i + j * lda);
        }
    }
    // Process edge cases seperately
    if(M % 8 > 0){
        for(int i = M - M % 8; i < M; i++){
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
        for(int i = 0; i < M - M % 8; i++){
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

static void do_block_L2(int lda, int M, int N, int K, double *A, double *B, double *C){
    for (int i = 0; i < M; i += L1_SIZE) {
        // For each block-column of B
        for (int j = 0; j < N; j += L1_SIZE) {
            for (int k = 0; k < K; k += L1_SIZE) {
            // Accumulate block dgemms into block of C
            // Correct block dimensions if block "goes off edge of" the matrix
                int R = min(L1_SIZE, M - i);
                int S = min(L1_SIZE, N - j);
                int T = min(L1_SIZE, K - k);
                // Perform individual block dgemm
                do_block_L1(lda, R, S, T, A + i + k * lda, B + k + j * lda, C + i + j * lda);
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
                do_block_L2(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

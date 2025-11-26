#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>

#include "utils.h"

__global__ void small_vector_add(const float* A, const float* B, float* C, int N) {
    int idx = THREAD_CORD_2_LINEAR_IDX(threadIdx, blockIdx, blockDim, gridDim);
    if (idx > N) {
        LOG_ERROR_DEVICE("Index out of bounds: %d", idx);
    } else {
        C[idx] = A[idx] + B[idx];
    }
}

int gen_and_add_vec(int n=1024) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    std::srand(0);
    h_A = static_cast<float*>(std::malloc(n * sizeof(float)));
    h_B = static_cast<float*>(std::malloc(n * sizeof(float)));
    h_C = static_cast<float*>(std::malloc(n * sizeof(float)));
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_C[i] = 0.0f;
    }
    HANDLE_ERROR(cudaMalloc(&d_A, n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_B, n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_C, n * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));
    int blockSize = 64;
    int numBlocks = n / blockSize + (n % blockSize == 0 ? 0 : 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    small_vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "Element Num = " << n << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);
    // Verify results
    int check_cnt = 10;
    std::cerr << "Verifying results..." << std::endl;
    for (int i = 0; i < check_cnt && i < n; i++) {
        std::cerr << "Checking element " << i << std::endl;
        std::cerr << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Result verification failed at element "<< i << std::endl;
            return -1;
        }
    } 

    for (int i = n - check_cnt >= 0 ? n - check_cnt : 0; i < n; i++) {
        std::cerr << "Checking element " << i << std::endl;
        std::cerr << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Result verification failed at element "<< i << std::endl;
            return -1;
        }
    } 
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
    
}

int sample_vector_adds() {
    assert(gen_and_add_vec(100) == 0);
    assert(gen_and_add_vec(1000) == 0);
    assert(gen_and_add_vec(10000) == 0);
    assert(gen_and_add_vec(100000) == 0);
    assert(gen_and_add_vec(1000000) == 0);
    return 0;   
}

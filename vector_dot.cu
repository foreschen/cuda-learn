#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include "utils.h"

__global__ void dot(const float* A, const float* B, float* C, int N) {
    // __shared__ float cache[blockDim.x * blockDim.y * blockDim.z];
    extern __shared__ float cache[]; // max block size is 1024
    int idx = THREAD_CORD_2_LINEAR_IDX(threadIdx, blockIdx, blockDim, gridDim);
    int idx_in_block = idx % (blockDim.x * blockDim.y * blockDim.z);
    int idx_in_grid = idx / (blockDim.x * blockDim.y * blockDim.z);
    cache[idx_in_block] = 0.0f;
    __syncthreads();
    // LOG_ERROR_DEVICE("Thread %d in block %d processing global index %d", idx_in_block, idx_in_grid, idx);
    if (idx > N) {
        LOG_ERROR_DEVICE("Index out of bounds: %d", idx);
    } else {
        // C[idx] = A[idx] + B[idx];
        cache[idx_in_block] = A[idx] * B[idx];
        // LOG_ERROR_DEVICE("Dot value at block %d, global %d\n, A=%f, B=%f, mul=%f",
        //     idx_in_block, idx, A[idx], B[idx], cache[idx_in_block]);
    }
    __syncthreads();

    // the block thread size must be power of 2
    int stride = blockDim.x * blockDim.y * blockDim.z / 2;
    while (stride != 0) {
        if (idx_in_block < stride) {
            cache[idx_in_block] += cache[idx_in_block + stride];
        }
        __syncthreads();
        stride /= 2;
    }

    if (idx_in_block == 0) {
        C[idx_in_grid] = cache[0];
    }
}

int gen_and_dot_vec(int n=1024) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    std::srand(0);
    h_A = static_cast<float*>(std::malloc(n * sizeof(float)));
    h_B = static_cast<float*>(std::malloc(n * sizeof(float)));
    //h_C = static_cast<float*>(std::malloc(n * sizeof(float)));
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
        // h_C[i] = 0.0f;
    }
    HANDLE_ERROR(cudaMalloc(&d_A, n * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_B, n * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice));
    int blockSize = 64;

    h_C = static_cast<float*>(std::malloc(blockSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_C, blockSize * sizeof(float)));

    int numBlocks = n / blockSize + (n % blockSize == 0 ? 0 : 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dot<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "Element Num = " << n << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    cudaMemcpy(h_C, d_C, blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    // Verify results
    int check_block_cnt = 10;
    std::cerr << "Verifying results..." << std::endl;
    for (int i = 0; i < check_block_cnt && i * blockSize < n; i++) {
        std::cerr << "Checking block " << i << std::endl;
        float curr_block_sum = 0.0f;
        for (int j = 0; j < blockSize && i * blockSize + j < n; j++) {
            std::cerr << h_A[i * blockSize + j] << " * " << h_B[i * blockSize + j] << " = " << h_C[i] << std::endl;
            curr_block_sum += h_A[i * blockSize + j] * h_B[i * blockSize + j];
        }
        std::cout << "Block " << i << " cpu sum: " << curr_block_sum << ", GPU result: " << h_C[i] << std::endl;
        if (fabs(h_C[i] - curr_block_sum) > 1e-5) {
            std::cerr << "Result verification failed at block "<< i << std::endl;
            return -1;
        }
    }

    std::cerr << "Verifying last results..." << std::endl;
    for (int i = blockSize - 1; i >= blockSize - check_block_cnt && i >= 0; i--) {
        std::cerr << "Checking block " << i << std::endl;
        float curr_block_sum = 0.0f;
        for (int j = 0; j < blockSize && i * blockSize + j < n; j++) {
            std::cerr << h_A[i * blockSize + j] << " * " << h_B[i * blockSize + j] << " = " << h_C[i] << std::endl;
            curr_block_sum += h_A[i * blockSize + j] * h_B[i * blockSize + j];
        }
        std::cout << "Block " << i << " cpu sum: " << curr_block_sum << ", GPU result: " << h_C[i] << std::endl;
        if (fabs(h_C[i] - curr_block_sum) > 1e-5) {
            std::cerr << "Result verification failed at block "<< i << std::endl;
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

int sample_vector_dot() {
    assert(gen_and_dot_vec(640) == 0);
    assert(gen_and_dot_vec(1000) == 0);
    assert(gen_and_dot_vec(10000) == 0);
    assert(gen_and_dot_vec(100000) == 0);
    assert(gen_and_dot_vec(1000000) == 0);
    return 0;   
}
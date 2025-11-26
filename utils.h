#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>

static __device__ int THREAD_CORD_2_LINEAR_IDX(dim3 blockCord,
                                     dim3 gridCord,
                                     dim3 blockDim,
                                     dim3 gridDim
                                    ) {
    int idxInBlock = blockCord.x + 
                        blockCord.y * blockDim.x + 
                        blockCord.z * blockDim.x * blockDim.y;
    int idxInGrid = gridCord.x + 
                    gridCord.y * gridDim.x + 
                    gridCord.z * gridDim.x * gridDim.y;
    int idx = idxInGrid * blockDim.x * blockDim.y * blockDim.z + idxInBlock;
    return idx;
}

static void PrintStackTrace() {
    void *array[10];
    size_t size;
    char **strings;
    
    size = backtrace(array, 10);
    strings = backtrace_symbols(array, size);
    
    printf("Stack trace (%zu frames):\n", size);
    for (size_t i = 0; i < size; i++) {
        printf("  [%zu] %s\n", i, strings[i]);
    }
    
    free(strings);
}

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
    if (err != cudaSuccess) {
        printf("[ERROR]: %s in %s at line %d\n", 
               cudaGetErrorString(err), file, line);
        // PrintStackTrace();
        exit(EXIT_FAILURE);
    }
}

#define LOG_ERROR_DEVICE(fmt, ...) do { \
    printf("[ERROR]: " fmt " (%s:%d)\n", ##__VA_ARGS__, __FILE__, __LINE__); \
} while(0)


#define LOG_INFO(fmt, ...) do { \
    fprintf(stdout, "[INFO]: " fmt " (%s:%d)\n", ##__VA_ARGS__, __FILE__, __LINE__); \
} while(0)

#define LOG_ERROR(fmt, ...) do { \
    fprintf(stderr, "[ERROR]: " fmt " (%s:%d)\n", ##__VA_ARGS__, __FILE__, __LINE__); \
} while(0)

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define HANDLE_NULL(a) {if (a == NULL) { \
                            printf("Host memory failed in %s at line %d\n", \
                                   __FILE__, __LINE__); \
                            exit(EXIT_FAILURE);}}
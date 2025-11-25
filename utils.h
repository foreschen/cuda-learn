#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <unistd.h>

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
        PrintStackTrace();
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
#define HANDLE_NULL(a) {if (a == NULL) { \
                            printf("Host memory failed in %s at line %d\n", \
                                   __FILE__, __LINE__); \
                            PrintStackTrace(); \
                            exit(EXIT_FAILURE);}}
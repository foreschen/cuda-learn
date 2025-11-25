#include <cuda_runtime.h>
#include <iostream>

static __global__ void kernel(void)
{

}

int hello() {
    kernel<<<dim3(1, 1, 4), dim3(5, 1, 1)>>>();
    std::cout << "Hello, CUDA!" << std::endl;
    return 0;
}
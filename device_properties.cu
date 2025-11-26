#include <cuda_runtime.h>
#include <iostream>
int device_properties() {
    int devnum;
    cudaGetDeviceCount(&devnum);
    std::cout << "Number of CUDA devices: " << devnum << std::endl;

    for (int i = 0; i < devnum; ++i) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "Device " << i << ": " << devProp.name << std::endl;
        std::cout << "  Total Global Memory: " << devProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Shared Memory Per block: " << devProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  MP infos: " << devProp.multiProcessorCount << " MPs" << std::endl;
        printf( "Threads in warp:  %d\n", devProp.warpSize );
        printf( "Max threads per block:  %d\n",
                    devProp.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    devProp.maxThreadsDim[0], devProp.maxThreadsDim[1],
                    devProp.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    devProp.maxGridSize[0], devProp.maxGridSize[1],
                    devProp.maxGridSize[2] );
        
    }
    return 0;
}
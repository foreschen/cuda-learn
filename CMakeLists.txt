cmake_minimum_required(VERSION 3.18)

project(cuda-learn LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 添加可执行文件
file(GLOB SOURCES "*.cu")
add_executable(run_all utils.h mylib.h ${SOURCES})

# 设置 CUDA 架构（根据你的 GPU 调整）
set_target_properties(run_all PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

# 链接 CUDA 运行时库
target_link_libraries(run_all ${CUDA_LIBRARIES})

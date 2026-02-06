# set(CMAKE_PREFIX_PATH "/home/mr_zlc/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
# 查找当前目录下的所有 .cu 文件
file(GLOB CUDA_SOURCES "*.cu")
# 循环遍历所有 .cu 文件，为每个文件创建独立的目标
foreach(CU_FILE ${CUDA_SOURCES})
    # 提取文件名（不包括路径）
    get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${CU_FILE})
    target_link_libraries(${EXE_NAME} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    
    # 如果是 Debug 模式，添加调试选项
    # 需要调试再开，不然影响运行速度和内存
    # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #     target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    # endif()
endforeach()

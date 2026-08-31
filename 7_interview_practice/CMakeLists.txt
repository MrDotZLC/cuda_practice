# find_package(BLAS REQUIRED)
# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)
# 查找当前目录下的所有 .cu 文件
file(GLOB CURRENT_CU  "*.cu")
file(GLOB REDUCTION_CU "reduction_and_scan/*.cu")
file(GLOB MATRIX_CU "matrix_operations/*.cu")
file(GLOB NORM_CU "normalization_and_activation/*.cu")
file(GLOB ATTENTION_CU "attention/*.cu")
file(GLOB MISC_CU "miscellaneous/*.cu")
set(CUDA_SOURCES
    ${CURRENT_CU}
    ${REDUCTION_CU}
    ${MATRIX_CU}
    ${NORM_CU}
    ${ATTENTION_CU}
    ${MISC_CU}
)
# file(GLOB_RECURSE CUDA_SOURCES "*.cu")
# 循环遍历所有 .cu 文件，为每个文件创建独立的目标
foreach(CU_FILE ${CUDA_SOURCES})
    # 提取文件名（不包括路径）
    get_filename_component(EXE_NAME ${CU_FILE} NAME_WE)
    add_executable(${EXE_NAME} ${CU_FILE})
    target_link_libraries(${EXE_NAME} PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
    
    # 如果是 Debug 模式，添加调试选项
    # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #     target_compile_options(${EXE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    # endif()
endforeach()

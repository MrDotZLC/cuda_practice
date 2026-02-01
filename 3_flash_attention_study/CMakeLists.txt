# cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "/home/mr_zlc/libtorch/share/cmake/Torch;${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(flash-attention flash-attention.cpp flash.cu)
target_link_libraries(flash-attention "${TORCH_LIBRARIES}")
set_property(TARGET flash-attention PROPERTY CXX_STANDARD 17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash-attention PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my-flash my-flash.cpp my-flash.cu)
target_link_libraries(my-flash "${TORCH_LIBRARIES}")
set_property(TARGET my-flash PROPERTY CXX_STANDARD 17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my-flash PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# # 查找当前目录下的所有 .cu 文件
# file(GLOB CPP_SOURCES "*.cpp")
# # 排除 main.cpp
# list(REMOVE_ITEM CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
# # 循环遍历所有 .cu 文件，为每个文件创建独立的目标
# foreach(CPP_FILE ${CPP_SOURCES})
#     # 提取文件名（不包括路径）
#     get_filename_component(EXE_NAME ${CPP_FILE} NAME_WE)
#     add_executable(${EXE_NAME} ${CPP_FILE})
#     target_link_libraries(${EXE_NAME} "${TORCH_LIBRARIES}")
#     set_property(TARGET ${EXE_NAME} PROPERTY CXX_STANDARD 17)
#     # 如果是 Debug 模式，添加调试选项
#     # if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     #     target_compile_options(${EXE_NAME} PRIVATE $<$<CONFIG:Debug>:-G>)
#     # endif()
# endforeach()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app>)
# endif (MSVC)
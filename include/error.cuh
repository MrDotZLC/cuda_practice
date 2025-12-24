#pragma once
#include <stdio.h>

#define CHECK_CUDA(call)                                       \
    do                                                    \
    {                                                     \
        const cudaError_t error_code = call;              \
        if (error_code != cudaSuccess)                    \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code: %d, reason: %s\n", error_code,  \
                   cudaGetErrorString(error_code));       \
            exit(1);                                      \
        }                                                 \
    } while (0)

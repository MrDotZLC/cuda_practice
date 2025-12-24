#include "error.cuh"
#include <cstdio>

const int NUM_REPEAtS = 10;
void timing(const double *x, const int N);
double reduce(const double *x, const int N);

int main() {
    const int N = 1e8;
    const int M = sizeof(double) * N;
    double *x = (double *)malloc(M);
    for (int n = 0; n < N; n++) {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
}

void timing(const double *x, const int N) {
    double sum = 0;

    for (int repeat = 0; repeat < NUM_REPEAtS; repeat++) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    printf("sum = %f.\n", sum);
}

double reduce(const double *x, const int N) {
    double sum = 0.0;
    for (int n = 0; n < N; n++) {
        sum += x[n];
    }
    return sum;
}
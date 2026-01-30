#include <torch/torch.h>
#include <iostream>
// #include "flash.cu"

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

int main() {
    const int batch_size = 16;
    const int n_head = 12;
    const int seq_len = 64;
    const int head_embd = 64;

    auto q = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto k = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();
    auto v = torch::randn({batch_size, n_head, seq_len, head_embd}).cuda();

    std::cout << "=== profiling manual attention ===" << std::endl;

    forward(q, k, v);

    return 0;
}
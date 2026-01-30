#include <vector>
#include <iostream>
#include <cmath>

std::vector<float> native_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f;
    for (int i = 0; i < src.size(); i++) {
        sum += std::exp(src[i]);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i]) / sum;
    }
    return dst;
}

std::vector<float> safe_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f, mx = -99999.f;
    for (float f : src) {
        mx = std::max(f, mx);
    }
    for (float f : src) {
        sum += std::exp(f - mx);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i] - mx) / sum;
    }
    return dst;
}

std::vector<float> online_softmax(const std::vector<float> src) {
    std::vector<float> dst(src.size());
    float sum = 0.f, mx = -99999.f, pre_mx = 0.0f;
    for (float f : src) {
        pre_mx = mx;
        mx = std::max(f, mx);
        sum = sum * std::exp(pre_mx - mx) + std::exp(f - mx);
    }
    for (int i = 0; i < src.size(); i++) {
        dst[i] = std::exp(src[i] - mx) / sum;
    }
    return dst;
}

float online_softmax_dot_product(const std::vector<float> src, const std::vector<float> value) {
    float dst = 0.f, l = 0.f, mx = -99999.f, pre_mx = -99999.f;
    for (float f : src) {
        mx = std::max(f, pre_mx);
        l = l * std::exp(pre_mx - mx) + std::exp(f - mx);
        pre_mx = mx;
    }
    for (int i = 0; i < src.size(); i++) {
        dst += std::exp(src[i] - mx) / l * value[i];
    }
    return dst;
}

float online_softmax_dot_product_perfect(const std::vector<float> src, const std::vector<float> value) {
    float dst = 0.f, l = 0.f, pre_l = 0.f, mx = -99999.f, pre_mx = -99999.f;
    for (int i = 0; i < src.size(); i++) {
        mx = std::max(src[i], pre_mx);
        l = pre_l * std::exp(pre_mx - mx) + std::exp(src[i] - mx);
        dst = (dst * std::exp(pre_mx - mx) * pre_l + std::exp(src[i] - mx) * value[i]) / l;
        pre_mx = mx;
        pre_l = l;
    }
    return dst;
}

int main() {
    std::vector<float> src = {1.2f, 2.5f, 4.61f, 10.85f, 48.12f};
    std::vector<float> dst = native_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst1 = safe_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::vector<float> dst2 = online_softmax(src);
    for (float f : dst) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

    std::vector<float> value = {3.1f, 6.42f, 5.161f, 4.85f, 7.12f};
    float dst3 = online_softmax_dot_product(src, value);
    std::cout << dst3 << std::endl;

    float dst4 = online_softmax_dot_product_perfect(src, value);
    std::cout << dst4 << std::endl;

    return 0;
}

// 输出：
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19 
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19 
// 0.0493478 0.181072 1.49352 765.966 1.17588e+19
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::sampling {

enum class TorchRandnPrecision {
    Float32,
    BFloat16,
};

void fill_torch_cuda_randn(
    float * output,
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_randn(
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision = TorchRandnPrecision::Float32,
    uint64_t start_index = 0);

void fill_torch_cuda_uniform(
    float * output,
    size_t count,
    uint64_t seed,
    uint64_t start_index = 0);

std::vector<float> generate_torch_cuda_uniform(size_t count, uint64_t seed, uint64_t start_index = 0);

float torch_cuda_tensor_iterator_exponential_element(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t call_index,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor);

}  // namespace engine::sampling

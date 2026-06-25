#include "engine/framework/sampling/torch_random.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace engine::sampling {
namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53U;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57U;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9U;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85U;
constexpr float kInvTwoPow32 = 2.3283064365386963e-10F;
constexpr float kInvTwoPow32TwoPi = 1.4629180792671596e-09F;

struct Philox4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
};

void mul_hi_lo(uint32_t lhs, uint32_t rhs, uint32_t & hi, uint32_t & lo) {
    const uint64_t product = static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs);
    lo = static_cast<uint32_t>(product);
    hi = static_cast<uint32_t>(product >> 32U);
}

Philox4 philox_round(Philox4 counter, uint32_t key0, uint32_t key1) {
    uint32_t hi0 = 0;
    uint32_t lo0 = 0;
    uint32_t hi1 = 0;
    uint32_t lo1 = 0;
    mul_hi_lo(kPhiloxM0, counter.x, hi0, lo0);
    mul_hi_lo(kPhiloxM1, counter.z, hi1, lo1);
    return Philox4{
        hi1 ^ counter.y ^ key0,
        lo1,
        hi0 ^ counter.w ^ key1,
        lo0,
    };
}

Philox4 philox_4x32_10(Philox4 counter, uint64_t seed) {
    uint32_t key0 = static_cast<uint32_t>(seed);
    uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
    for (int round = 0; round < 10; ++round) {
        counter = philox_round(counter, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return counter;
}

void box_muller(uint32_t uniform0, uint32_t uniform1, float & normal0, float & normal1) {
    const float radius_input =
        static_cast<float>(uniform0) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
    const float angle =
        static_cast<float>(uniform1) * kInvTwoPow32TwoPi + (kInvTwoPow32TwoPi * 0.5F);
    const float radius = std::sqrt(-2.0F * std::log(radius_input));
    normal0 = radius * std::sin(angle);
    normal1 = radius * std::cos(angle);
}

float torch_cuda_randn_element(uint64_t seed, uint64_t index) {
    const Philox4 counter{
        0U,
        0U,
        static_cast<uint32_t>(index),
        static_cast<uint32_t>(index >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    float normal0 = 0.0F;
    float normal1 = 0.0F;
    box_muller(random.x, random.y, normal0, normal1);
    return normal0;
}

float torch_cuda_uniform_element(uint64_t seed, uint64_t index) {
    const Philox4 counter{
        0U,
        0U,
        static_cast<uint32_t>(index),
        static_cast<uint32_t>(index >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    return static_cast<float>(random.x) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
}

float torch_cuda_uniform_tensor_iterator_element(
    uint64_t seed,
    uint64_t sequence,
    uint64_t offset_blocks,
    int component) {
    const Philox4 counter{
        static_cast<uint32_t>(offset_blocks),
        static_cast<uint32_t>(offset_blocks >> 32U),
        static_cast<uint32_t>(sequence),
        static_cast<uint32_t>(sequence >> 32U),
    };
    const Philox4 random = philox_4x32_10(counter, seed);
    uint32_t value = random.x;
    switch (component) {
    case 0:
        value = random.x;
        break;
    case 1:
        value = random.y;
        break;
    case 2:
        value = random.z;
        break;
    case 3:
        value = random.w;
        break;
    default:
        throw std::invalid_argument("torch CUDA TensorIterator uniform component is invalid");
    }
    return static_cast<float>(value) * kInvTwoPow32 + (kInvTwoPow32 * 0.5F);
}

float round_to_bfloat16(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    float rounded = 0.0F;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

}  // namespace

void fill_torch_cuda_randn(
    float * output,
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision,
    uint64_t start_index) {
    if (output == nullptr && count != 0) {
        throw std::invalid_argument("torch CUDA randn output pointer is null");
    }
    for (size_t index = 0; index < count; ++index) {
        float value = torch_cuda_randn_element(seed, start_index + static_cast<uint64_t>(index));
        if (precision == TorchRandnPrecision::BFloat16) {
            value = round_to_bfloat16(value);
        }
        output[index] = value;
    }
}

std::vector<float> generate_torch_cuda_randn(
    size_t count,
    uint64_t seed,
    TorchRandnPrecision precision,
    uint64_t start_index) {
    std::vector<float> output(count);
    fill_torch_cuda_randn(output.data(), output.size(), seed, precision, start_index);
    return output;
}

void fill_torch_cuda_uniform(float * output, size_t count, uint64_t seed, uint64_t start_index) {
    if (output == nullptr && count != 0) {
        throw std::invalid_argument("torch CUDA uniform output pointer is null");
    }
    for (size_t index = 0; index < count; ++index) {
        output[index] = torch_cuda_uniform_element(seed, start_index + static_cast<uint64_t>(index));
    }
}

std::vector<float> generate_torch_cuda_uniform(size_t count, uint64_t seed, uint64_t start_index) {
    std::vector<float> output(count);
    fill_torch_cuda_uniform(output.data(), output.size(), seed, start_index);
    return output;
}

float torch_cuda_tensor_iterator_exponential_element(
    uint64_t seed,
    uint64_t total_elements,
    uint64_t element_index,
    uint64_t call_index,
    int64_t multiprocessor_count,
    int64_t max_threads_per_multiprocessor) {
    if (total_elements == 0 || element_index >= total_elements) {
        throw std::invalid_argument("torch CUDA TensorIterator exponential element index is out of range");
    }
    if (multiprocessor_count <= 0 || max_threads_per_multiprocessor <= 0) {
        throw std::invalid_argument("torch CUDA TensorIterator exponential requires CUDA device properties");
    }
    constexpr uint64_t block_size = 256;
    constexpr uint64_t unroll_factor = 4;
    uint64_t grid = (total_elements + block_size - 1) / block_size;
    uint64_t blocks_per_sm = static_cast<uint64_t>(max_threads_per_multiprocessor) / block_size;
    if (blocks_per_sm == 0) {
        blocks_per_sm = 1;
    }
    const uint64_t grid_cap = static_cast<uint64_t>(multiprocessor_count) * blocks_per_sm;
    grid = std::max<uint64_t>(1, std::min(grid_cap, grid));
    const uint64_t stride = block_size * grid;
    const uint64_t counter_offset = ((total_elements - 1) / (stride * unroll_factor) + 1) * unroll_factor;
    const uint64_t chunk = element_index / stride;
    const int component = static_cast<int>(chunk % unroll_factor);
    const uint64_t loop_index = chunk / unroll_factor;
    const uint64_t sequence = element_index % stride;
    const uint64_t offset_blocks = call_index * (counter_offset / unroll_factor) + loop_index;
    const float uniform = torch_cuda_uniform_tensor_iterator_element(seed, sequence, offset_blocks, component);
    return -std::log(uniform);
}

}  // namespace engine::sampling

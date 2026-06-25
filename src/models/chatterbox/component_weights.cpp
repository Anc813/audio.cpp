#include "components/component_weights.h"

#include <random>

namespace engine::models::chatterbox::components {

uint64_t choose_seed(uint64_t seed) {
    if (seed != 0) {
        return seed;
    }
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32U) ^ static_cast<uint64_t>(rd());
}

std::vector<float> read_f32_tensor(
    const engine::assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    return source.require_f32(name, expected_shape);
}

CampplusEncoderWeights::Conv1dWeights load_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    CampplusEncoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.weight = read_f32_tensor(source, prefix + ".weight", {out_channels, in_channels, kernel});
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        conv.bias = read_f32_tensor(source, prefix + ".bias", {out_channels});
        conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return conv;
}

}  // namespace engine::models::chatterbox::components

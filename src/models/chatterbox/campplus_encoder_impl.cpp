#include "components/component_weights.h"

#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"

#include "ggml.h"

#include <chrono>
#include <memory>
#include <mutex>

namespace engine::models::chatterbox::components {
namespace {

using SteadyClock = std::chrono::steady_clock;

bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::ensure_backend_addressable_layout(ctx, input);
}

engine::core::TensorValue concat_along_axis(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs,
    int logical_axis) {
    auto output_shape = lhs.shape;
    output_shape.dims[logical_axis] += rhs.shape.dims[logical_axis];
    return engine::core::wrap_tensor(
        ggml_concat(
            ctx.ggml,
            lhs.tensor,
            rhs.tensor,
            engine::core::logical_axis_to_ggml_axis(lhs.shape.rank, logical_axis)),
        output_shape,
        lhs.type);
}

engine::core::TensorValue permute_tensor(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    std::array<int, engine::core::kMaxTensorRank> axes) {
    std::array<int, engine::core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    engine::core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;
    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        output_shape.dims[out_logical_axis] = input.shape.dims[in_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = engine::core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
    }
    return engine::core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

CampplusEncoderWeights::BatchNorm1dWeights load_bn1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    bool affine = true) {
    CampplusEncoderWeights::BatchNorm1dWeights bn;
    bn.affine = affine;
    if (affine) {
        bn.weight = read_f32_tensor(source, prefix + ".weight", {channels});
        bn.bias = read_f32_tensor(source, prefix + ".bias", {channels});
    }
    bn.running_mean = read_f32_tensor(source, prefix + ".running_mean", {channels});
    bn.running_var = read_f32_tensor(source, prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0f);
    std::vector<float> shift(static_cast<size_t>(channels), 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
        const size_t index = static_cast<size_t>(c);
        const float gamma = affine ? bn.weight[index] : 1.0f;
        const float beta = affine ? bn.bias[index] : 0.0f;
        const float inv = 1.0f / std::sqrt(bn.running_var[index] + 1.0e-5f);
        scale[index] = gamma * inv;
        shift[index] = beta - bn.running_mean[index] * gamma * inv;
    }
    bn.scale_tensor = store.make_f32(engine::core::TensorShape::from_dims({1, channels, 1}), std::move(scale));
    bn.shift_tensor = store.make_f32(engine::core::TensorShape::from_dims({1, channels, 1}), std::move(shift));
    return bn;
}

CampplusEncoderWeights::BatchNorm2dWeights load_bn2d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    CampplusEncoderWeights::BatchNorm2dWeights bn;
    bn.weight = read_f32_tensor(source, prefix + ".weight", {channels});
    bn.bias = read_f32_tensor(source, prefix + ".bias", {channels});
    bn.running_mean = read_f32_tensor(source, prefix + ".running_mean", {channels});
    bn.running_var = read_f32_tensor(source, prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels), 0.0f);
    std::vector<float> shift(static_cast<size_t>(channels), 0.0f);
    for (int64_t c = 0; c < channels; ++c) {
        const size_t index = static_cast<size_t>(c);
        const float inv = 1.0f / std::sqrt(bn.running_var[index] + 1.0e-5f);
        scale[index] = bn.weight[index] * inv;
        shift[index] = bn.bias[index] - bn.running_mean[index] * bn.weight[index] * inv;
    }
    bn.scale_tensor = store.make_f32(engine::core::TensorShape::from_dims({1, channels, 1, 1}), std::move(scale));
    bn.shift_tensor = store.make_f32(engine::core::TensorShape::from_dims({1, channels, 1, 1}), std::move(shift));
    return bn;
}

CampplusEncoderWeights::Conv2dWeights load_conv2d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    engine::assets::TensorStorageType weight_storage_type) {
    CampplusEncoderWeights::Conv2dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel_h = kernel_h;
    conv.kernel_w = kernel_w;
    conv.stride_h = stride_h;
    conv.stride_w = stride_w;
    conv.padding_h = padding_h;
    conv.padding_w = padding_w;
    conv.weight = read_f32_tensor(source, prefix + ".weight", {out_channels, in_channels, kernel_h, kernel_w});
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel_h, kernel_w});
    return conv;
}

CampplusEncoderWeights::Conv1dWeights fold_bn_after_conv1d(
    engine::core::BackendWeightStore & store,
    const CampplusEncoderWeights::Conv1dWeights & conv,
    const CampplusEncoderWeights::BatchNorm1dWeights & bn,
    engine::assets::TensorStorageType weight_storage_type) {
    if (conv.out_channels != static_cast<int64_t>(bn.running_var.size())) {
        throw std::runtime_error("fold_bn_after_conv1d out_channels mismatch");
    }

    CampplusEncoderWeights::Conv1dWeights folded = conv;
    folded.use_bias = true;
    folded.bias.assign(static_cast<size_t>(conv.out_channels), 0.0f);

    const int64_t kernel_elems = conv.in_channels * conv.kernel;
    for (int64_t oc = 0; oc < conv.out_channels; ++oc) {
        const float gamma = bn.affine ? bn.weight[static_cast<size_t>(oc)] : 1.0f;
        const float beta = bn.affine ? bn.bias[static_cast<size_t>(oc)] : 0.0f;
        const float inv = 1.0f / std::sqrt(bn.running_var[static_cast<size_t>(oc)] + 1.0e-5f);
        const float scale = gamma * inv;
        const float shift = beta - bn.running_mean[static_cast<size_t>(oc)] * scale;
        const size_t oc_offset = static_cast<size_t>(oc * kernel_elems);
        for (int64_t k = 0; k < kernel_elems; ++k) {
            folded.weight[oc_offset + static_cast<size_t>(k)] *= scale;
        }
        const float original_bias = conv.use_bias ? conv.bias[static_cast<size_t>(oc)] : 0.0f;
        folded.bias[static_cast<size_t>(oc)] = original_bias * scale + shift;
    }

    folded.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({folded.out_channels, folded.in_channels, folded.kernel}),
        weight_storage_type,
        folded.weight);
    folded.bias_tensor = store.make_f32(
        engine::core::TensorShape::from_dims({folded.out_channels}),
        folded.bias);
    return folded;
}

CampplusEncoderWeights::Conv2dWeights fold_bn_after_conv2d(
    engine::core::BackendWeightStore & store,
    const CampplusEncoderWeights::Conv2dWeights & conv,
    const CampplusEncoderWeights::BatchNorm2dWeights & bn,
    engine::assets::TensorStorageType weight_storage_type) {
    if (conv.out_channels != static_cast<int64_t>(bn.running_var.size())) {
        throw std::runtime_error("fold_bn_after_conv2d out_channels mismatch");
    }

    CampplusEncoderWeights::Conv2dWeights folded = conv;
    folded.use_bias = true;
    folded.bias.assign(static_cast<size_t>(conv.out_channels), 0.0f);

    const int64_t kernel_elems = conv.in_channels * conv.kernel_h * conv.kernel_w;
    for (int64_t oc = 0; oc < conv.out_channels; ++oc) {
        const float gamma = bn.weight[static_cast<size_t>(oc)];
        const float beta = bn.bias[static_cast<size_t>(oc)];
        const float inv = 1.0f / std::sqrt(bn.running_var[static_cast<size_t>(oc)] + 1.0e-5f);
        const float scale = gamma * inv;
        const float shift = beta - bn.running_mean[static_cast<size_t>(oc)] * scale;
        const size_t oc_offset = static_cast<size_t>(oc * kernel_elems);
        for (int64_t k = 0; k < kernel_elems; ++k) {
            folded.weight[oc_offset + static_cast<size_t>(k)] *= scale;
        }
        folded.bias[static_cast<size_t>(oc)] = shift;
    }

    folded.weight_tensor = store.make_from_f32(
        engine::core::TensorShape::from_dims({folded.out_channels, folded.in_channels, folded.kernel_h, folded.kernel_w}),
        weight_storage_type,
        folded.weight);
    folded.bias_tensor = store.make_f32(
        engine::core::TensorShape::from_dims({folded.out_channels}),
        folded.bias);
    return folded;
}

class CampplusBackendRunner {
public:
    struct RunResult {
        std::vector<float> dense;
        double upload_ms = 0.0;
        double ggml_ms = 0.0;
        double readback_ms = 0.0;
    };

    CampplusBackendRunner(
        int64_t frames,
        const CampplusEncoderWeights & weights)
        : execution_context_(*weights.execution_context) {
        ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context for campplus backend");
        }

        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = ggml_;
        ctx.module_instance_name = "campplus_backend";
        ctx.backend_type = execution_context_.backend_type();

        input_tensor_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 1, 80, frames}));

        const auto conv2d = [&](const engine::core::TensorValue & input,
                                int64_t height,
                                int64_t width,
                                const CampplusEncoderWeights::Conv2dWeights & conv) {
            const int64_t out_h = (height + 2 * conv.padding_h - conv.kernel_h) / conv.stride_h + 1;
            const int64_t out_w = (width + 2 * conv.padding_w - conv.kernel_w) / conv.stride_w + 1;
            auto output = engine::core::wrap_tensor(
                ggml_conv_2d(
                    ctx.ggml,
                    conv.weight_tensor.tensor,
                    contiguous(ctx, input).tensor,
                    static_cast<int>(conv.stride_w),
                    static_cast<int>(conv.stride_h),
                    static_cast<int>(conv.padding_w),
                    static_cast<int>(conv.padding_h),
                    1,
                    1),
                engine::core::TensorShape::from_dims({1, conv.out_channels, out_h, out_w}),
                GGML_TYPE_F32);
            if (conv.use_bias) {
                const auto bias = engine::core::reshape_tensor(
                    ctx,
                    conv.bias_tensor,
                    engine::core::TensorShape::from_dims({1, conv.out_channels, 1, 1}));
                output = engine::core::wrap_tensor(
                    ggml_add(ctx.ggml, output.tensor, bias.tensor),
                    output.shape,
                    GGML_TYPE_F32);
            }
            return output;
        };

        const auto conv1d = [&](const engine::core::TensorValue & input,
                                int64_t in_channels,
                                int64_t time_steps,
                                const CampplusEncoderWeights::Conv1dWeights & conv) {
            if (conv.kernel == 1 && conv.stride == 1 && conv.padding == 0 && conv.dilation == 1) {
                auto weight = engine::core::reshape_tensor(
                    ctx,
                    conv.weight_tensor,
                    engine::core::TensorShape::from_dims({conv.out_channels, conv.in_channels}));
                auto btc = permute_tensor(ctx, input, {0, 2, 1});
                auto flat = engine::core::reshape_tensor(
                    ctx,
                    contiguous(ctx, btc),
                    engine::core::TensorShape::from_dims({time_steps, in_channels}));
                auto projected = engine::core::wrap_tensor(
                    ggml_mul_mat(ctx.ggml, weight.tensor, flat.tensor),
                    engine::core::TensorShape::from_dims({time_steps, conv.out_channels}),
                    GGML_TYPE_F32);
                if (conv.use_bias) {
                    auto bias = engine::core::reshape_tensor(
                        ctx,
                        conv.bias_tensor,
                        engine::core::TensorShape::from_dims({1, conv.out_channels}));
                    projected = engine::core::wrap_tensor(
                        ggml_add(ctx.ggml, projected.tensor, bias.tensor),
                        projected.shape,
                        GGML_TYPE_F32);
                }
                auto projected_btc = engine::core::reshape_tensor(
                    ctx,
                    projected,
                    engine::core::TensorShape::from_dims({1, time_steps, conv.out_channels}));
                return contiguous(ctx, permute_tensor(ctx, projected_btc, {0, 2, 1}));
            }
            const int64_t out_frames =
                (time_steps + 2 * conv.padding - conv.dilation * (conv.kernel - 1) - 1) / conv.stride + 1;
            auto output = engine::core::wrap_tensor(
                ggml_conv_1d(
                    ctx.ggml,
                    conv.weight_tensor.tensor,
                    contiguous(ctx, input).tensor,
                    static_cast<int>(conv.stride),
                    static_cast<int>(conv.padding),
                    static_cast<int>(conv.dilation)),
                engine::core::TensorShape::from_dims({1, conv.out_channels, out_frames}),
                GGML_TYPE_F32);
            if (conv.use_bias) {
                auto bias = engine::core::reshape_tensor(
                    ctx,
                    conv.bias_tensor,
                    engine::core::TensorShape::from_dims({1, conv.out_channels, 1}));
                output = engine::core::wrap_tensor(
                    ggml_add(ctx.ggml, output.tensor, bias.tensor),
                    output.shape,
                    GGML_TYPE_F32);
            }
            return output;
        };

        const auto batch_norm_1d = [&](const engine::core::TensorValue & input,
                                       const CampplusEncoderWeights::BatchNorm1dWeights & bn) {
            auto scaled = engine::core::wrap_tensor(
                ggml_mul(ctx.ggml, input.tensor, bn.scale_tensor.tensor),
                input.shape,
                GGML_TYPE_F32);
            return engine::core::wrap_tensor(
                ggml_add(ctx.ggml, scaled.tensor, bn.shift_tensor.tensor),
                input.shape,
                GGML_TYPE_F32);
        };

        const auto relu = [&](const engine::core::TensorValue & input) {
            return engine::core::wrap_tensor(ggml_relu(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
        };

        const auto as_bct = [&](const engine::core::TensorValue & input4d, int64_t channels, int64_t time_steps) {
            return engine::core::reshape_tensor(
                ctx,
                contiguous(ctx, input4d),
                engine::core::TensorShape::from_dims({1, channels, time_steps}));
        };

        const auto as_bchw = [&](const engine::core::TensorValue & input3d, int64_t channels, int64_t time_steps) {
            return engine::core::reshape_tensor(
                ctx,
                input3d,
                engine::core::TensorShape::from_dims({1, channels, 1, time_steps}));
        };

        const auto avg_pool_repeat = [&](const engine::core::TensorValue & input,
                                         int64_t channels,
                                         int64_t time_steps,
                                         int64_t seg_len) {
            const int64_t seg_frames = (time_steps + seg_len - 1) / seg_len;
            const int64_t padded_time = seg_frames * seg_len;
            auto input4d = as_bchw(input, channels, time_steps);
            auto padded = engine::core::wrap_tensor(
                ggml_pad(ctx.ggml, input4d.tensor, static_cast<int>(padded_time - time_steps), 0, 0, 0),
                engine::core::TensorShape::from_dims({1, channels, 1, padded_time}),
                GGML_TYPE_F32);
            auto pooled = engine::core::wrap_tensor(
                ggml_pool_2d(ctx.ggml, padded.tensor, GGML_OP_POOL_AVG, static_cast<int>(seg_len), 1, static_cast<int>(seg_len), 1, 0, 0),
                engine::core::TensorShape::from_dims({1, channels, 1, seg_frames}),
                GGML_TYPE_F32);
            std::vector<float> correction_values(static_cast<size_t>(seg_frames), 1.0f);
            for (int64_t seg = 0; seg < seg_frames; ++seg) {
                const int64_t repeat_len = std::min<int64_t>(seg_len, time_steps - seg * seg_len);
                correction_values[static_cast<size_t>(seg)] =
                    static_cast<float>(seg_len) / static_cast<float>(repeat_len);
            }
            auto correction = writer_.make_f32_tensor(
                ctx,
                engine::core::TensorShape::from_dims({1, 1, 1, seg_frames}),
                correction_values);
            auto ratio = engine::core::wrap_tensor(
                ggml_mul(ctx.ggml, pooled.tensor, correction.tensor),
                pooled.shape,
                GGML_TYPE_F32);
            auto expanded_padded = engine::core::wrap_tensor(
                ggml_interpolate(
                    ctx.ggml,
                    contiguous(ctx, ratio).tensor,
                    padded_time,
                    1,
                    channels,
                    1,
                    GGML_SCALE_MODE_NEAREST),
                engine::core::TensorShape::from_dims({1, channels, 1, padded_time}),
                GGML_TYPE_F32);
            if (padded_time == time_steps) {
                return as_bct(expanded_padded, channels, time_steps);
            }
            auto * trimmed_view = ggml_view_4d(
                ctx.ggml,
                expanded_padded.tensor,
                time_steps,
                1,
                channels,
                1,
                expanded_padded.tensor->nb[1],
                expanded_padded.tensor->nb[2],
                expanded_padded.tensor->nb[3],
                0);
            auto trimmed = engine::core::wrap_tensor(
                trimmed_view,
                engine::core::TensorShape::from_dims({1, channels, 1, time_steps}),
                GGML_TYPE_F32);
            return as_bct(trimmed, channels, time_steps);
        };

        const auto global_avg = [&](const engine::core::TensorValue & input, int64_t channels, int64_t time_steps) {
            (void) time_steps;
            return engine::core::wrap_tensor(
                ggml_mean(ctx.ggml, input.tensor),
                engine::core::TensorShape::from_dims({1, channels, 1}),
                GGML_TYPE_F32);
        };

        const auto stats_pool = [&](const engine::core::TensorValue & input, int64_t channels, int64_t time_steps) {
            auto mean = global_avg(input, channels, time_steps);
            auto centered = engine::core::wrap_tensor(
                ggml_sub(ctx.ggml, input.tensor, mean.tensor),
                input.shape,
                GGML_TYPE_F32);
            auto squared = engine::core::wrap_tensor(
                ggml_mul(ctx.ggml, centered.tensor, centered.tensor),
                input.shape,
                GGML_TYPE_F32);
            auto var = global_avg(squared, channels, time_steps);
            if (time_steps > 1) {
                auto correction = writer_.make_f32_tensor(
                    ctx,
                    engine::core::TensorShape::from_dims({1, 1, 1}),
                    std::vector<float>{static_cast<float>(time_steps) / static_cast<float>(time_steps - 1)});
                var = engine::core::wrap_tensor(
                    ggml_mul(ctx.ggml, var.tensor, correction.tensor),
                    var.shape,
                    GGML_TYPE_F32);
            }
            auto stddev = engine::core::wrap_tensor(ggml_sqrt(ctx.ggml, var.tensor), var.shape, GGML_TYPE_F32);
            auto stats = concat_along_axis(ctx, mean, stddev, 1);
            return stats;
        };

        auto x4 = input_tensor_;
        x4 = relu(conv2d(x4, 80, frames, weights.head_conv1_folded));

        auto run_res_block = [&](const engine::core::TensorValue & input,
                                 int64_t height,
                                 int64_t width,
                                 const CampplusEncoderWeights::BasicResBlockWeights & block) {
            const int64_t out_h = (height + 2 * block.conv1.padding_h - block.conv1.kernel_h) / block.conv1.stride_h + 1;
            auto main = relu(conv2d(input, height, width, block.conv1_folded));
            main = conv2d(main, out_h, width, block.conv2_folded);
            auto shortcut = input;
            if (block.use_shortcut) {
                shortcut = conv2d(input, height, width, block.shortcut_conv_folded);
            }
            auto added = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, main.tensor, shortcut.tensor),
                main.shape,
                GGML_TYPE_F32);
            return relu(added);
        };

        int64_t height = 80;
        x4 = run_res_block(x4, height, frames, weights.head_layer1[0]); height = (height + 1) / 2;
        x4 = run_res_block(x4, height, frames, weights.head_layer1[1]);
        x4 = run_res_block(x4, height, frames, weights.head_layer2[0]); height = (height + 1) / 2;
        x4 = run_res_block(x4, height, frames, weights.head_layer2[1]);
        x4 = relu(conv2d(x4, height, frames, weights.head_conv2_folded));
        height = (height + 1) / 2;

        int64_t channels = 32 * height;
        int64_t time_steps = frames;
        auto x = as_bct(x4, channels, time_steps);

        x = relu(conv1d(x, channels, time_steps, weights.tdnn_linear_folded));
        channels = 128;
        time_steps = (time_steps + 1) / 2;

        const int block_layers[3] = {12, 24, 16};
        for (int block_index = 0; block_index < 3; ++block_index) {
            for (int layer_index = 0; layer_index < block_layers[block_index]; ++layer_index) {
                const auto & layer = weights.blocks[static_cast<size_t>(block_index)].layers[static_cast<size_t>(layer_index)];
                auto h = relu(batch_norm_1d(x, layer.nonlinear1_bn));
                h = conv1d(h, channels, time_steps, layer.linear1);
                h = relu(batch_norm_1d(h, layer.nonlinear2_bn));
                auto local = conv1d(h, 128, time_steps, layer.cam_layer.linear_local);
                auto avg = avg_pool_repeat(h, 128, time_steps, 100);
                auto global = global_avg(h, 128, time_steps);
                auto context = engine::core::wrap_tensor(
                    ggml_add(ctx.ggml, avg.tensor, global.tensor),
                    avg.shape,
                    GGML_TYPE_F32);
                auto gate = relu(conv1d(context, 128, time_steps, layer.cam_layer.linear1));
                auto sigmoid = engine::core::wrap_tensor(
                    ggml_sigmoid(ctx.ggml, conv1d(gate, 64, time_steps, layer.cam_layer.linear2).tensor),
                    engine::core::TensorShape::from_dims({1, 32, time_steps}),
                    GGML_TYPE_F32);
                auto out = engine::core::wrap_tensor(
                    ggml_mul(ctx.ggml, local.tensor, sigmoid.tensor),
                    local.shape,
                    GGML_TYPE_F32);
                x = concat_along_axis(ctx, x, out, 1);
                channels += 32;
            }
            x = relu(batch_norm_1d(x, weights.transits[static_cast<size_t>(block_index)].nonlinear_bn));
            x = conv1d(x, channels, time_steps, weights.transits[static_cast<size_t>(block_index)].linear);
            channels /= 2;
        }

        x = relu(batch_norm_1d(x, weights.out_nonlinear_bn));
        auto pooled = stats_pool(x, channels, time_steps);
        auto dense = conv1d(pooled, channels * 2, 1, weights.dense_linear_folded);
        output_tensor_ = dense;

        graph_ = ggml_new_graph_custom(ggml_, 65536, false);
        ggml_build_forward_expand(graph_, output_tensor_.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_.backend());
        if (buffer_ == nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
            throw std::runtime_error("failed to allocate backend tensors for campplus graph");
        }
        writer_.flush();
    }

    ~CampplusBackendRunner() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
        }
    }

    RunResult run(const std::vector<float> & input) {
        std::lock_guard<std::mutex> lock(run_mutex_);
        const auto upload_started = SteadyClock::now();
        engine::core::write_tensor_f32(input_tensor_, input);
        const double upload_ms = engine::debug::elapsed_ms(upload_started);
        const auto ggml_started = SteadyClock::now();
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for campplus graph");
        }
        const double ggml_ms = engine::debug::elapsed_ms(ggml_started);
        const auto readback_started = SteadyClock::now();
        auto dense = engine::core::read_tensor_f32(output_tensor_.tensor);
        const double readback_ms = engine::debug::elapsed_ms(readback_started);
        return RunResult{std::move(dense), upload_ms, ggml_ms, readback_ms};
    }

private:
    const engine::core::ExecutionContext & execution_context_;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_tensor_;
    engine::core::TensorValue output_tensor_;
    engine::core::DeferredTensorWriter writer_;
    std::mutex run_mutex_;
};

}  // namespace

class CampplusInferenceSessionCache;

CampplusEncoderOutputs compute_campplus_embedding_from_features(
    CampplusInferenceSessionCache & cache,
    const CampplusEncoderWeights & weights,
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims,
    engine::core::BackendConfig backend = {});

class CampplusInferenceSessionCache {
public:
    struct State;

    CampplusInferenceSessionCache();
    ~CampplusInferenceSessionCache();
    CampplusInferenceSessionCache(CampplusInferenceSessionCache &&) noexcept;
    CampplusInferenceSessionCache & operator=(CampplusInferenceSessionCache &&) noexcept;

    CampplusInferenceSessionCache(const CampplusInferenceSessionCache &) = delete;
    CampplusInferenceSessionCache & operator=(const CampplusInferenceSessionCache &) = delete;

private:
    std::unique_ptr<State> state_;

    friend CampplusEncoderOutputs compute_campplus_embedding_from_features(
        CampplusInferenceSessionCache & cache,
        const CampplusEncoderWeights & weights,
        const std::vector<float> & features,
        int64_t frames,
        int64_t dims,
        engine::core::BackendConfig backend);
};

struct CampplusInferenceSessionCache::State {
    std::mutex mutex;
    std::shared_ptr<CampplusBackendRunner> runner;
    const CampplusEncoderWeights * weights = nullptr;
    int64_t frames = 0;
    engine::core::BackendConfig backend;
};

std::shared_ptr<CampplusBackendRunner> get_campplus_runner(
    CampplusInferenceSessionCache::State & cache,
    int64_t frames,
    const CampplusEncoderWeights & weights,
    const engine::core::BackendConfig & backend) {
    std::lock_guard<std::mutex> lock(cache.mutex);
    if (!cache.runner ||
        cache.weights != &weights ||
        cache.frames != frames ||
        !same_backend(cache.backend, backend)) {
        if (!same_backend(weights.execution_context->config(), backend)) {
            throw std::runtime_error("CAMPPlus backend does not match uploaded weight backend");
        }
        cache.runner = std::make_shared<CampplusBackendRunner>(frames, weights);
        cache.weights = &weights;
        cache.frames = frames;
        cache.backend = backend;
    }
    return cache.runner;
}

CampplusInferenceSessionCache::CampplusInferenceSessionCache()
    : state_(std::make_unique<State>()) {}

CampplusInferenceSessionCache::~CampplusInferenceSessionCache() = default;
CampplusInferenceSessionCache::CampplusInferenceSessionCache(CampplusInferenceSessionCache &&) noexcept = default;
CampplusInferenceSessionCache & CampplusInferenceSessionCache::operator=(CampplusInferenceSessionCache &&) noexcept = default;

static std::shared_ptr<CampplusInferenceSessionCache> make_campplus_inference_session_cache() {
    return std::make_shared<CampplusInferenceSessionCache>();
}

static std::shared_ptr<const CampplusEncoderWeights> load_campplus_encoder_weights(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<CampplusEncoderWeights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.campplus.weights",
        512ull * 1024ull * 1024ull);

    weights->head_conv1 = load_conv2d(*weights->store, *source, "speaker_encoder.head.conv1", 32, 1, 3, 3, 1, 1, 1, 1, weight_storage_type);
    weights->head_bn1 = load_bn2d(*weights->store, *source, "speaker_encoder.head.bn1", 32);
    weights->head_conv1_folded = fold_bn_after_conv2d(*weights->store, weights->head_conv1, weights->head_bn1, weight_storage_type);
    weights->head_conv2 = load_conv2d(*weights->store, *source, "speaker_encoder.head.conv2", 32, 32, 3, 3, 2, 1, 1, 1, weight_storage_type);
    weights->head_bn2 = load_bn2d(*weights->store, *source, "speaker_encoder.head.bn2", 32);
    weights->head_conv2_folded = fold_bn_after_conv2d(*weights->store, weights->head_conv2, weights->head_bn2, weight_storage_type);

    auto load_res_block = [&](const std::string & prefix, bool use_shortcut, int64_t stride_h) {
        CampplusEncoderWeights::BasicResBlockWeights block;
        block.conv1 = load_conv2d(*weights->store, *source, prefix + ".conv1", 32, 32, 3, 3, stride_h, 1, 1, 1, weight_storage_type);
        block.bn1 = load_bn2d(*weights->store, *source, prefix + ".bn1", 32);
        block.conv1_folded = fold_bn_after_conv2d(*weights->store, block.conv1, block.bn1, weight_storage_type);
        block.conv2 = load_conv2d(*weights->store, *source, prefix + ".conv2", 32, 32, 3, 3, 1, 1, 1, 1, weight_storage_type);
        block.bn2 = load_bn2d(*weights->store, *source, prefix + ".bn2", 32);
        block.conv2_folded = fold_bn_after_conv2d(*weights->store, block.conv2, block.bn2, weight_storage_type);
        block.use_shortcut = use_shortcut;
        if (use_shortcut) {
            block.shortcut_conv = load_conv2d(*weights->store, *source, prefix + ".shortcut.0", 32, 32, 1, 1, stride_h, 1, 0, 0, weight_storage_type);
            block.shortcut_bn = load_bn2d(*weights->store, *source, prefix + ".shortcut.1", 32);
            block.shortcut_conv_folded = fold_bn_after_conv2d(*weights->store, block.shortcut_conv, block.shortcut_bn, weight_storage_type);
        }
        return block;
    };
    weights->head_layer1 = {
        load_res_block("speaker_encoder.head.layer1.0", true, 2),
        load_res_block("speaker_encoder.head.layer1.1", false, 1),
    };
    weights->head_layer2 = {
        load_res_block("speaker_encoder.head.layer2.0", true, 2),
        load_res_block("speaker_encoder.head.layer2.1", false, 1),
    };

    weights->tdnn_linear = load_conv1d(*weights->store, *source, "speaker_encoder.xvector.tdnn.linear", 128, 320, 5, 2, 2, 1, false, weight_storage_type);
    weights->tdnn_bn = load_bn1d(*weights->store, *source, "speaker_encoder.xvector.tdnn.nonlinear.batchnorm", 128);
    weights->tdnn_linear_folded = fold_bn_after_conv1d(*weights->store, weights->tdnn_linear, weights->tdnn_bn, weight_storage_type);

    const int block_layers[3] = {12, 24, 16};
    const int block_input_channels[3] = {128, 256, 512};
    weights->blocks.resize(3);
    weights->transits.resize(3);
    for (int block_index = 0; block_index < 3; ++block_index) {
        auto & block = weights->blocks[static_cast<size_t>(block_index)];
        for (int layer = 1; layer <= block_layers[block_index]; ++layer) {
            const int in_channels = block_input_channels[block_index] + (layer - 1) * 32;
            const std::string prefix =
                "speaker_encoder.xvector.block" + std::to_string(block_index + 1) + ".tdnnd" + std::to_string(layer);
            CampplusEncoderWeights::CAMDenseTDNNLayerWeights layer_weights;
            layer_weights.nonlinear1_bn = load_bn1d(*weights->store, *source, prefix + ".nonlinear1.batchnorm", in_channels);
            layer_weights.linear1 = load_conv1d(*weights->store, *source, prefix + ".linear1", 128, in_channels, 1, 1, 0, 1, false, weight_storage_type);
            layer_weights.nonlinear2_bn = load_bn1d(*weights->store, *source, prefix + ".nonlinear2.batchnorm", 128);
            layer_weights.cam_layer.linear_local =
                load_conv1d(*weights->store, *source, prefix + ".cam_layer.linear_local", 32, 128, 3, 1, block_index == 0 ? 1 : 2, block_index == 0 ? 1 : 2, false, weight_storage_type);
            layer_weights.cam_layer.linear1 =
                load_conv1d(*weights->store, *source, prefix + ".cam_layer.linear1", 64, 128, 1, 1, 0, 1, true, weight_storage_type);
            layer_weights.cam_layer.linear2 =
                load_conv1d(*weights->store, *source, prefix + ".cam_layer.linear2", 32, 64, 1, 1, 0, 1, true, weight_storage_type);
            block.layers.push_back(std::move(layer_weights));
        }
        const int transit_in = block_input_channels[block_index] + block_layers[block_index] * 32;
        const int transit_out = transit_in / 2;
        const std::string transit_prefix = "speaker_encoder.xvector.transit" + std::to_string(block_index + 1);
        weights->transits[static_cast<size_t>(block_index)].nonlinear_bn =
            load_bn1d(*weights->store, *source, transit_prefix + ".nonlinear.batchnorm", transit_in);
        weights->transits[static_cast<size_t>(block_index)].linear =
            load_conv1d(*weights->store, *source, transit_prefix + ".linear", transit_out, transit_in, 1, 1, 0, 1, false, weight_storage_type);
    }

    weights->out_nonlinear_bn = load_bn1d(*weights->store, *source, "speaker_encoder.xvector.out_nonlinear.batchnorm", 512);
    weights->dense_linear = load_conv1d(*weights->store, *source, "speaker_encoder.xvector.dense.linear", 192, 1024, 1, 1, 0, 1, false, weight_storage_type);
    weights->dense_bn = load_bn1d(*weights->store, *source, "speaker_encoder.xvector.dense.nonlinear.batchnorm", 192, false);
    weights->dense_linear_folded = fold_bn_after_conv1d(*weights->store, weights->dense_linear, weights->dense_bn, weight_storage_type);
    weights->store->upload();
    return weights;
}

CampplusEncoderOutputs compute_campplus_embedding_from_features(
    CampplusInferenceSessionCache & cache,
    const CampplusEncoderWeights & weights,
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims,
    engine::core::BackendConfig backend) {
    if (dims != 80) {
        throw std::runtime_error("CAMPPlus expects 80-dim features");
    }
    const int64_t batch = 1;
    auto x4 = std::vector<float>(static_cast<size_t>(batch * 1 * dims * frames), 0.0f);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t d = 0; d < dims; ++d) {
            x4[static_cast<size_t>(d * frames + t)] = features[static_cast<size_t>(t * dims + d)];
        }
    }
    auto runner = get_campplus_runner(*cache.state_, frames, weights, backend);
    auto run_result = runner->run(x4);
    CampplusEncoderOutputs outputs;
    outputs.embedding_size = 192;
    outputs.embedding.assign(run_result.dense.begin(), run_result.dense.begin() + 192);
    return outputs;
}

CampplusEncoderOutputs compute_campplus_embedding_from_features(
    const CampplusEncoderWeights & weights,
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims,
    engine::core::BackendConfig backend) {
    CampplusInferenceSessionCache cache;
    return compute_campplus_embedding_from_features(cache, weights, features, frames, dims, backend);
}

CampplusEncoderOutputs compute_campplus_embedding_from_audio(
    const CampplusEncoderWeights & weights,
    const runtime::AudioBuffer & audio,
    engine::core::BackendConfig backend) {
    const auto fbank = compute_campplus_fbank(audio);
    return compute_campplus_embedding_from_features(weights, fbank.features, fbank.frames, fbank.dims, backend);
}

}  // namespace engine::models::chatterbox::components

namespace engine::models::chatterbox {

struct CAMPPlusEncoderComponent::State {
    std::shared_ptr<components::CampplusInferenceSessionCache> cache;
};

CAMPPlusEncoderComponent CAMPPlusEncoderComponent::load_from_checkpoint(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    auto runtime_weights = components::load_campplus_encoder_weights(
        checkpoint_path,
        execution_context,
        weight_storage_type);
    auto weights = std::make_shared<CampplusEncoderComponentWeights>();
    weights->runtime_weights = std::move(runtime_weights);
    CAMPPlusEncoderComponent component(std::move(weights), execution_context);
    component.state_ = std::make_shared<State>(State{
        components::make_campplus_inference_session_cache(),
    });
    return component;
}

CAMPPlusEncoderComponent::CAMPPlusEncoderComponent(
    std::shared_ptr<const CampplusEncoderComponentWeights> weights,
    const engine::core::ExecutionContext & execution_context)
    : weights_(std::move(weights)), execution_context_(&execution_context) {}

const engine::core::BackendConfig & CAMPPlusEncoderComponent::backend() const noexcept {
    return execution_context_->config();
}

const std::shared_ptr<const CampplusEncoderComponentWeights> & CAMPPlusEncoderComponent::weights() const noexcept {
    return weights_;
}

SpeakerEncoderOutputs CAMPPlusEncoderComponent::embed_from_audio(const runtime::AudioBuffer & audio) const {
    const auto fbank = components::compute_campplus_fbank(audio);
    const auto outputs = components::compute_campplus_embedding_from_features(
        *state_->cache,
        *weights_->runtime_weights,
        fbank.features,
        fbank.frames,
        fbank.dims,
        execution_context_->config());
    SpeakerEncoderOutputs result;
    result.embedding = outputs.embedding;
    result.embedding_size = outputs.embedding_size;
    return result;
}

SpeakerEncoderOutputs CAMPPlusEncoderComponent::embed_from_features(
    const std::vector<float> & features,
    int64_t frames,
    int64_t dims) const {
    const auto outputs = components::compute_campplus_embedding_from_features(
        *state_->cache,
        *weights_->runtime_weights,
        features,
        frames,
        dims,
        execution_context_->config());
    SpeakerEncoderOutputs result;
    result.embedding = outputs.embedding;
    result.embedding_size = outputs.embedding_size;
    return result;
}

}  // namespace engine::models::chatterbox

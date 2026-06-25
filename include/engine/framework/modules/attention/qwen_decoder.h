#pragma once

#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <optional>
#include <vector>

struct ggml_cgraph;

namespace engine::modules {

struct QwenDecoderLayerConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
};

struct QwenMLPWeights {
    LinearWeights gate_proj;
    LinearWeights up_proj;
    LinearWeights down_proj;
};

struct QwenDecoderLayerWeights {
    NormWeights input_norm;
    AttentionWeights self_attention;
    NormWeights q_norm;
    NormWeights k_norm;
    NormWeights post_norm;
    QwenMLPWeights mlp;
};

struct QwenDecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

class QwenDecoderLayerModule {
public:
    explicit QwenDecoderLayerModule(QwenDecoderLayerConfig config);

    const QwenDecoderLayerConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    QwenDecoderLayerOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

    QwenDecoderLayerOutputs build_with_static_cache_tail(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & attention_mask) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    QwenDecoderLayerConfig config_;
};

struct QwenDecoderStackConfig {
    int64_t hidden_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t layers = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    ggml_prec attention_precision = GGML_PREC_F32;
    ggml_prec projection_precision = GGML_PREC_DEFAULT;
};

struct QwenDecoderStackWeights {
    std::vector<QwenDecoderLayerWeights> layers;
};

struct QwenDecoderStackLayerState {
    std::optional<core::TensorValue> key;
    std::optional<core::TensorValue> value;
};

struct QwenDecoderStackState {
    std::vector<QwenDecoderStackLayerState> layers;
};

struct QwenDecoderStackOutputs {
    core::TensorValue output;
    QwenDecoderStackState state;
};

class QwenDecoderStackModule {
public:
    explicit QwenDecoderStackModule(QwenDecoderStackConfig config);

    const QwenDecoderStackConfig & config() const noexcept;

    QwenDecoderStackOutputs build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const QwenDecoderStackWeights & weights,
        const std::optional<QwenDecoderStackState> & prefix_state = std::nullopt,
        const std::optional<core::TensorValue> & attention_mask = std::nullopt) const;

private:
    QwenDecoderStackConfig config_;
};

}  // namespace engine::modules

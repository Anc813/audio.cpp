#include "attention_internal.h"

namespace engine::modules {

using namespace attention::internal;

CrossAttentionModule::CrossAttentionModule(AttentionConfig config) : config_(config) {
    validate_attention_config(config_);
}

const AttentionConfig & CrossAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & CrossAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue CrossAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & query,
    const core::TensorValue & memory,
    const AttentionWeights & weights) const {
    return build_attention_impl(ctx, query, memory, config_, require_attention_weights(weights, config_.use_bias));
}

const core::ModuleSchema & CrossAttentionModule::static_schema() noexcept {
    return kCrossAttentionSchema;
}

}  // namespace engine::modules

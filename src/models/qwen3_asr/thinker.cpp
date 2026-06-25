#include "engine/models/qwen3_asr/thinker.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/decode_modules.h"


#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::qwen3_asr {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TextLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct ThinkerWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<TextLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

struct DecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

int64_t head_dim(const Qwen3ASRTextDecoderConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("Qwen3 ASR thinker attention config is invalid");
    }
    return config.head_dim;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const ThinkerWeights & weights,
    const Qwen3ASRTextDecoderConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * audio_embeddings,
    ggml_tensor * audio_positions,
    int64_t prompt_steps,
    int64_t audio_tokens) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, weights.token_embedding);
    if (audio_tokens > 0) {
        auto audio = core::wrap_tensor(
            audio_embeddings,
            core::TensorShape::from_dims({audio_tokens, config.hidden_size}),
            GGML_TYPE_F32);
        auto positions = core::wrap_tensor(
            audio_positions,
            core::TensorShape::from_dims({audio_tokens}),
            GGML_TYPE_I64);
        x = core::wrap_tensor(
            ggml_set_rows(ctx.ggml, x.tensor, audio.tensor, positions.tensor),
            x.shape,
            GGML_TYPE_F32);
    }
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.hidden_size}));
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    if (!core::has_backend_addressable_layout(q_heads.tensor) ||
        !core::has_backend_addressable_layout(k_heads.tensor) ||
        !core::has_backend_addressable_layout(v_heads.tensor)) {
        throw std::runtime_error("Qwen3 ASR flash attention expects contiguous Q/K/V heads");
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Qwen3 ASR thinker cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, dim}),
        GGML_TYPE_F32);
}

DecoderLayerOutputs decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const Qwen3ASRTextDecoderConfig & config,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.num_attention_heads * dim, config.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

DecoderLayerOutputs decoder_layer_with_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const Qwen3ASRTextDecoderConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask) {
    const int64_t dim = head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.num_attention_heads * dim, config.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto key_tail = cache_view(ctx, cache_key, scratch_slot, 1, config.num_key_value_heads, dim);
    auto value_tail = cache_view(ctx, cache_value, scratch_slot, 1, config.num_key_value_heads, dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail.tensor));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_value.shape.rank}).build(ctx, cache_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = flash_attention_from_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, 1, config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

ThinkerWeights load_weights(
    const Qwen3ASRAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.text_decoder;
    const auto & source = *assets.model_weights;
    ThinkerWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "qwen3_asr.thinker.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "thinker.model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    const int64_t dim = head_dim(config);
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "thinker.model.layers." + std::to_string(layer);
        TextLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_proj = weights.store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size});
        w.k_proj = weights.store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.v_proj = weights.store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.o_proj = weights.store->load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        w.gate_proj = weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.up_proj = weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.down_proj = weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "thinker.model.norm.weight", {config.hidden_size});
    weights.lm_head = weights.store->load_tensor(source, "thinker.lm_head.weight", storage_type, {config.output_size, config.hidden_size});
    weights.store->upload();
    return weights;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("Qwen3 ASR thinker cannot select from empty logits");
    }
    size_t best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

bool is_eos(const Qwen3ASRTextDecoderConfig & config, int32_t token) {
    return std::find(config.eos_token_ids.begin(), config.eos_token_ids.end(), static_cast<int64_t>(token)) !=
        config.eos_token_ids.end();
}

class ThinkerWeightsRuntime {
public:
    ThinkerWeightsRuntime(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<ThinkerWeights>(load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Qwen3 ASR thinker weights runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Qwen3 ASR thinker backend is not initialized");
        }
    }

    const Qwen3ASRAssets & assets() const noexcept {
        return *assets_;
    }

    const ThinkerWeights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const Qwen3ASRAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const ThinkerWeights> weights_;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<ThinkerWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t audio_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          audio_tokens_(audio_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker prefill requires positive prompt length");
        }
        if (audio_tokens_ < 0 || audio_tokens_ > prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker prefill graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        audio_embeddings_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, std::max<int64_t>(audio_tokens_, 1));
        audio_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(audio_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_,
            audio_embeddings_,
            audio_positions_,
            prompt_steps_,
            audio_tokens_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        for (const auto & layer : weights.layers) {
            auto out = decoder_layer(ctx, x, positions, layer, config);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modules::LinearModule({config.hidden_size, config.output_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3 ASR thinker prefill graph");
        }
        std::vector<int32_t> pos(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const ThinkerWeightsRuntime & runtime, int64_t prompt_steps, int64_t audio_tokens) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps && audio_tokens_ == audio_tokens;
    }

    PrefillOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & audio_embeddings,
        const std::vector<int32_t> & audio_positions) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill token id count mismatch");
        }
        if (static_cast<int64_t>(audio_embeddings.size()) != audio_tokens_ * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio embedding size mismatch");
        }
        if (static_cast<int64_t>(audio_positions.size()) != audio_tokens_) {
            throw std::runtime_error("Qwen3 ASR thinker prefill audio position count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        if (audio_tokens_ > 0) {
            std::vector<int64_t> positions(audio_positions.begin(), audio_positions.end());
            ggml_backend_tensor_set(
                audio_embeddings_,
                audio_embeddings.data(),
                0,
                audio_embeddings.size() * sizeof(float));
            ggml_backend_tensor_set(
                audio_positions_,
                positions.data(),
                0,
                positions.size() * sizeof(int64_t));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.output_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * audio_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class PromptClassificationGraph {
public:
    PromptClassificationGraph(
        std::shared_ptr<ThinkerWeightsRuntime> runtime,
        int64_t prompt_steps,
        int64_t audio_tokens,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps),
          audio_tokens_(audio_tokens) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker classification requires positive prompt length");
        }
        if (audio_tokens_ < 0 || audio_tokens_ > prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio token count is invalid");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker classification graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.classify", runtime_->backend_type()};
        token_ids_input_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        audio_embeddings_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, std::max<int64_t>(audio_tokens_, 1));
        audio_positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I64, std::max<int64_t>(audio_tokens_, 1));
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_input_,
            audio_embeddings_,
            audio_positions_,
            prompt_steps_,
            audio_tokens_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        for (const auto & layer : weights.layers) {
            x = decoder_layer(ctx, x, positions, layer, config).output;
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modules::LinearModule({config.hidden_size, config.output_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        auto token_ids = engine::sampling::GreedyDecodeModule().build(ctx, logits);
        token_ids_ = token_ids.tensor;
        ggml_set_output(token_ids_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, token_ids_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3 ASR thinker classification graph");
        }
        std::vector<int32_t> pos(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.classify_prompt_steps", prompt_steps_);
    }

    ~PromptClassificationGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const ThinkerWeightsRuntime & runtime, int64_t prompt_steps, int64_t audio_tokens) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps && audio_tokens_ == audio_tokens;
    }

    std::vector<int32_t> run(
        const std::vector<int32_t> & input_ids,
        const std::vector<float> & audio_embeddings,
        const std::vector<int32_t> & audio_positions) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (static_cast<int64_t>(input_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker classification token id count mismatch");
        }
        if (static_cast<int64_t>(audio_embeddings.size()) != audio_tokens_ * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio embedding size mismatch");
        }
        if (static_cast<int64_t>(audio_positions.size()) != audio_tokens_) {
            throw std::runtime_error("Qwen3 ASR thinker classification audio position count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_input_, input_ids.data(), 0, input_ids.size() * sizeof(int32_t));
        if (audio_tokens_ > 0) {
            std::vector<int64_t> positions(audio_positions.begin(), audio_positions.end());
            ggml_backend_tensor_set(
                audio_embeddings_,
                audio_embeddings.data(),
                0,
                audio_embeddings.size() * sizeof(float));
            ggml_backend_tensor_set(
                audio_positions_,
                positions.data(),
                0,
                positions.size() * sizeof(int64_t));
        }
        debug::timing_log_scalar("qwen3_asr.thinker.classify_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker classification graph compute failed");
        }
        std::vector<int32_t> token_ids(static_cast<size_t>(prompt_steps_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("qwen3_asr.thinker.classify_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return token_ids;
    }

private:
    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_input_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * audio_positions_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * token_ids_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<ThinkerWeightsRuntime> runtime, int64_t cache_steps, size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("Qwen3 ASR thinker decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Qwen3 ASR thinker decode graph context");
        }
        const auto & config = runtime_->assets().config.text_decoder;
        const auto & weights = runtime_->weights();
        const int64_t dim = head_dim(config);
        core::ModuleBuildContext ctx{ctx_.get(), "qwen3_asr.thinker.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token_id = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_id, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_ + 1, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        for (const auto & layer : weights.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, config.num_key_value_heads, dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_ + 1, config.num_key_value_heads, dim})));
            auto out = decoder_layer_with_static_cache_tail(
                ctx,
                graph_,
                x,
                positions,
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                attention_mask);
            x = out.output;
            key_sources_.push_back(ggml_view_1d(ctx_.get(), out.key.tensor, config.num_key_value_heads * dim, 0));
            value_sources_.push_back(ggml_view_1d(ctx_.get(), out.value.tensor, config.num_key_value_heads * dim, 0));
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_ + 1,
            config.num_key_value_heads * dim,
            std::move(cache_keys),
            std::move(cache_values));
        build_transfer_views(config.num_key_value_heads * dim);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        auto logits = modules::LinearModule({config.hidden_size, config.output_size, false})
                          .build(ctx, x, {weights.lm_head, std::nullopt});
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate Qwen3 ASR thinker decode graph");
        }
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_ + 1), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("qwen3_asr.thinker.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("qwen3_asr.thinker.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const ThinkerWeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    std::vector<float> run_step(int32_t token) {
        const auto & config = runtime_->assets().config.text_decoder;
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("Qwen3 ASR thinker decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_values_[static_cast<size_t>(i)] = visible;
        }
        attention_mask_values_[static_cast<size_t>(cache_steps_)] = visible;
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Qwen3 ASR thinker decode graph compute failed");
        }
        std::vector<float> logits(static_cast<size_t>(config.output_size));
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        const size_t dst_slot = static_cast<size_t>(step_cache_.valid_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        step_cache_.advance_after_direct_append(1);
        return logits;
    }

private:
    void build_transfer_views(int64_t step_elems) {
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(
                    ctx_.get(),
                    step_cache_.key_tensor(layer).tensor,
                    step_elems,
                    byte_offset));
                value_slot.push_back(ggml_view_1d(
                    ctx_.get(),
                    step_cache_.value_tensor(layer).tensor,
                    step_elems,
                    byte_offset));
            }
        }
    }

    std::shared_ptr<ThinkerWeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct Qwen3ASRThinkerRuntime::Impl {
    Impl(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<ThinkerWeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes) {}

    void validate_prompt_audio(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings) const {
        const auto & config = weights->assets().config.text_decoder;
        if (audio_embeddings.hidden_size != config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR audio embedding hidden size mismatch");
        }
        if (audio_embeddings.tokens != static_cast<int64_t>(prompt.audio_token_positions.size())) {
            throw std::runtime_error("Qwen3 ASR audio embedding token count does not match prompt placeholders");
        }
        if (static_cast<int64_t>(audio_embeddings.values.size()) != audio_embeddings.tokens * config.hidden_size) {
            throw std::runtime_error("Qwen3 ASR audio embedding value count mismatch");
        }
        for (const int32_t position : prompt.audio_token_positions) {
            if (position < 0 || position >= static_cast<int32_t>(prompt.input_ids.size())) {
                throw std::runtime_error("Qwen3 ASR audio placeholder position out of range");
            }
        }
    }

    Qwen3ASRGeneratedTokens generate(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings,
        const Qwen3ASRGenerationOptions & options) {
        const auto & config = weights->assets().config.text_decoder;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("Qwen3 ASR thinker prompt is empty");
        }
        if (options.max_new_tokens <= 0) {
            throw std::runtime_error("Qwen3 ASR max_new_tokens must be positive");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps + options.max_new_tokens > config.max_position_embeddings) {
            throw std::runtime_error("Qwen3 ASR thinker request exceeds max_position_embeddings");
        }
        auto timing_start = Clock::now();
        validate_prompt_audio(prompt, audio_embeddings);
        debug::timing_log_scalar("qwen3_asr.thinker.prompt_prepare_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (prefill_graph == nullptr || !prefill_graph->matches(*weights, prompt_steps, audio_embeddings.tokens)) {
            prefill_graph = std::make_unique<PrefillGraph>(
                weights,
                prompt_steps,
                audio_embeddings.tokens,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.prefill_prompt_steps", prompt_steps);
        }
        timing_start = Clock::now();
        auto prefill = prefill_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            prompt.audio_token_positions);
        debug::timing_log_scalar("qwen3_asr.thinker.prefill_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        const int64_t required_cache_steps = prompt_steps + options.max_new_tokens;
        if (decode_graph == nullptr || !decode_graph->can_run(*weights, required_cache_steps)) {
            decode_graph = std::make_unique<DecodeGraph>(weights, required_cache_steps, decode_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.decode.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.decode_cache_steps", required_cache_steps);
        }
        decode_graph->import_state(prefill.kv_state);

        Qwen3ASRGeneratedTokens out;
        std::vector<float> logits = std::move(prefill.logits);
        timing_start = Clock::now();
        for (int64_t step = 0; step < options.max_new_tokens; ++step) {
            const int32_t token = argmax_index(logits);
            if (is_eos(config, token)) {
                break;
            }
            out.token_ids.push_back(token);
            logits = decode_graph->run_step(token);
        }
        debug::timing_log_scalar("qwen3_asr.thinker.decode_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

    std::vector<int32_t> classify_prompt(
        const Qwen3ASRPrompt & prompt,
        const Qwen3ASRAudioEmbeddings & audio_embeddings) {
        const auto & config = weights->assets().config.text_decoder;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("Qwen3 ASR thinker classification prompt is empty");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps > config.max_position_embeddings) {
            throw std::runtime_error("Qwen3 ASR thinker classification request exceeds max_position_embeddings");
        }
        auto timing_start = Clock::now();
        validate_prompt_audio(prompt, audio_embeddings);
        debug::timing_log_scalar("qwen3_asr.thinker.classify_prompt_prepare_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (classification_graph == nullptr ||
            !classification_graph->matches(*weights, prompt_steps, audio_embeddings.tokens)) {
            classification_graph = std::make_unique<PromptClassificationGraph>(
                weights,
                prompt_steps,
                audio_embeddings.tokens,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("qwen3_asr.thinker.classify.graph.build_ms", 0.0);
            debug::trace_log_scalar("qwen3_asr.thinker.classify_prompt_steps", prompt_steps);
        }
        timing_start = Clock::now();
        auto token_ids = classification_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            prompt.audio_token_positions);
        debug::timing_log_scalar("qwen3_asr.thinker.classify_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return token_ids;
    }

    std::shared_ptr<ThinkerWeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
    std::unique_ptr<PromptClassificationGraph> classification_graph;
};

Qwen3ASRThinkerRuntime::Qwen3ASRThinkerRuntime(
    std::shared_ptr<const Qwen3ASRAssets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

Qwen3ASRThinkerRuntime::~Qwen3ASRThinkerRuntime() = default;

Qwen3ASRGeneratedTokens Qwen3ASRThinkerRuntime::generate(
    const Qwen3ASRPrompt & prompt,
    const Qwen3ASRAudioEmbeddings & audio_embeddings,
    const Qwen3ASRGenerationOptions & options) {
    return impl_->generate(prompt, audio_embeddings, options);
}

std::vector<int32_t> Qwen3ASRThinkerRuntime::classify_prompt(
    const Qwen3ASRPrompt & prompt,
    const Qwen3ASRAudioEmbeddings & audio_embeddings) {
    return impl_->classify_prompt(prompt, audio_embeddings);
}

}  // namespace engine::models::qwen3_asr

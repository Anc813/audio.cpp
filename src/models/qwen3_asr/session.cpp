#include "engine/models/qwen3_asr/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const Qwen3ASRAssets> require_assets(std::shared_ptr<const Qwen3ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 ASR session requires assets");
    }
    return assets;
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_audio_encoder_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error("qwen3_asr.audio_encoder_weight_type currently supports only native, f32, and f16");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

}  // namespace

Qwen3ASRSession::Qwen3ASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3ASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.audio_encoder_graph_arena_mb"}, 128ull * 1024ull * 1024ull)),
      thinker_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_weight_context_mb"}, 64ull * 1024ull * 1024ull)),
      audio_encoder_weight_storage_type_(option_weight_type(options, "qwen3_asr.audio_encoder_weight_type", engine::assets::TensorStorageType::Native)),
      thinker_weight_storage_type_(option_weight_type(
          options,
          "qwen3_asr.thinker_weight_type",
          option_weight_type(options, "qwen3_asr.weight_type", engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_, execution_context(), audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_),
      thinker_(
          assets_,
          execution_context(),
          thinker_prefill_graph_arena_bytes_,
          thinker_decode_graph_arena_bytes_,
          thinker_weight_context_bytes_,
          thinker_weight_storage_type_),
      prompt_builder_(tokenizer_),
      postprocessor_(tokenizer_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Qwen3 ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 ASR currently supports offline sessions");
    }
    validate_audio_encoder_weight_storage(audio_encoder_weight_storage_type_);
    validate_matmul_weight_storage(thinker_weight_storage_type_, "qwen3_asr.thinker_weight_type");
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("qwen3_asr.", 0) == 0 &&
            key != "qwen3_asr.audio_encoder_graph_arena_mb" &&
            key != "qwen3_asr.thinker_prefill_graph_arena_mb" &&
            key != "qwen3_asr.thinker_decode_graph_arena_mb" &&
            key != "qwen3_asr.thinker_weight_context_mb" &&
            key != "qwen3_asr.audio_encoder_weight_type" &&
            key != "qwen3_asr.thinker_weight_type" &&
            key != "qwen3_asr.weight_type") {
            throw std::runtime_error("unknown Qwen3 ASR session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

std::string Qwen3ASRSession::family() const {
    return "qwen3_asr";
}

runtime::VoiceTaskKind Qwen3ASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3ASRSession::run_mode() const {
    return task_.mode;
}

void Qwen3ASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Qwen3 ASR prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult Qwen3ASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 ASR run()");
    const auto wall_start = Clock::now();
    const auto asr_request = make_request(request);
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(asr_request.audio);
    const auto frontend_end = Clock::now();
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(asr_request, features.encoder_tokens);
    const auto prompt_end = Clock::now();
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto encoder_end = Clock::now();
    const auto thinker_start = Clock::now();
    const auto tokens = thinker_.generate(prompt, audio_embeddings, asr_request.generation);
    const auto thinker_end = Clock::now();
    const auto postprocess_start = Clock::now();
    const auto decoded = postprocessor_.decode(tokens, asr_request);
    const auto postprocess_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, decoded.language};
    result.word_timestamps = decoded.word_timestamps;
    const auto wall_end = Clock::now();
    debug::timing_log_scalar("qwen3_asr.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("qwen3_asr.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("qwen3_asr.audio_encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
    debug::timing_log_scalar("qwen3_asr.thinker_ms", engine::debug::elapsed_ms(thinker_start, thinker_end));
    debug::timing_log_scalar("qwen3_asr.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::trace_log_scalar("qwen3_asr.prompt_tokens", prompt.input_ids.size());
    debug::trace_log_scalar("qwen3_asr.audio_frames", features.frames);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    return result;
}

Qwen3ASRRequest Qwen3ASRSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Qwen3 ASR run() requires audio_input");
    }
    Qwen3ASRRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = assets_->config.max_new_tokens;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
        if (out.generation.max_new_tokens <= 0) {
            throw std::runtime_error("Qwen3 ASR max_tokens must be positive");
        }
    }
    if (const auto value = runtime::find_option(request.options, {"return_timestamps"})) {
        out.generation.return_timestamps = runtime::parse_bool_option(*value, "return_timestamps");
    }
    return out;
}

}  // namespace engine::models::qwen3_asr

#include "engine/models/voxcpm2/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::voxcpm2 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultTextChunkSize = 2048;

std::shared_ptr<const VoxCPM2Assets>
require_assets(std::shared_ptr<const VoxCPM2Assets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("VoxCPM2 session requires assets");
  }
  return assets;
}

void reject_enabled_denoise(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  const auto match = runtime::find_option_match(options, keys);
  if (match.has_value() &&
      runtime::parse_bool_option(match->value, match->key)) {
    throw std::runtime_error(
        "VoxCPM2 denoise is disabled in this implementation");
  }
}

void reject_denoiser_option(
    const std::unordered_map<std::string, std::string> &options,
    std::initializer_list<std::string_view> keys) {
  if (runtime::find_option_match(options, keys).has_value()) {
    throw std::runtime_error(
        "VoxCPM2 denoise is disabled in this implementation");
  }
}

bool audio_buffer_equal(const runtime::AudioBuffer &lhs,
                        const runtime::AudioBuffer &rhs) {
  return lhs.sample_rate == rhs.sample_rate && lhs.channels == rhs.channels &&
         lhs.samples == rhs.samples;
}

bool optional_audio_equal(const std::optional<runtime::AudioBuffer> &lhs,
                          const std::optional<runtime::AudioBuffer> &rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return false;
  }
  return !lhs.has_value() || audio_buffer_equal(*lhs, *rhs);
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type,
                             const char *option_name) {
  if (storage_type == engine::assets::TensorStorageType::Native ||
      storage_type == engine::assets::TensorStorageType::F32 ||
      storage_type == engine::assets::TensorStorageType::F16 ||
      storage_type == engine::assets::TensorStorageType::BF16 ||
      storage_type == engine::assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(std::string(option_name) +
                           " supports only native, f32, f16, bf16, and q8_0");
}

void parse_weight_type(
    const std::unordered_map<std::string, std::string> &options,
    const char *key, engine::assets::TensorStorageType &storage_type) {
  const auto it = options.find(key);
  if (it == options.end()) {
    return;
  }
  storage_type = engine::assets::parse_tensor_storage_type(it->second);
  validate_weight_storage(storage_type, key);
}

void validate_session_options(
    const std::unordered_map<std::string, std::string> &options) {
  for (const auto &[key, value] : options) {
    (void)value;
    if (key.rfind("voxcpm2.", 0) != 0) {
      continue;
    }
    if (key == "voxcpm2.weight_context_mb" ||
        key == "voxcpm2.text_embedding_graph_context_mb" ||
        key == "voxcpm2.lm_step_graph_context_mb" ||
        key == "voxcpm2.projection_graph_context_mb" ||
        key == "voxcpm2.local_encoder_graph_context_mb" ||
        key == "voxcpm2.dit_graph_context_mb" ||
        key == "voxcpm2.audiovae_weight_context_mb" ||
        key == "voxcpm2.audiovae_graph_context_mb" ||
        key == "voxcpm2.audiovae_encoder_graph_context_mb" ||
        key == "voxcpm2.audiovae_latent_capacity" ||
        key == "voxcpm2.audiovae_encoder_sample_capacity" ||
        key == "voxcpm2.weight_type" ||
        key == "voxcpm2.audiovae_weight_type" ||
        key == "voxcpm2.denoise" || key == "voxcpm2.load_denoiser") {
      continue;
    }
    throw std::runtime_error("unknown VoxCPM2 session option: " + key);
  }
}

int64_t product(const std::vector<int64_t> &values) {
  int64_t out = 1;
  for (const int64_t value : values) {
    if (value <= 0) {
      throw std::runtime_error("VoxCPM2 AudioVAE decoder rate is invalid");
    }
    out *= value;
  }
  return out;
}

} // namespace

VoxCPM2Session::VoxCPM2Session(runtime::TaskSpec task,
                               runtime::SessionOptions options,
                               std::shared_ptr<const VoxCPM2Assets> assets)
    : RuntimeSessionBase(options), task_(task),
      assets_(require_assets(std::move(assets))) {
  if (task_.mode != runtime::RunMode::Offline &&
      task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 only supports offline and streaming sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts) {
    throw std::runtime_error("VoxCPM2 only supports the Tts task");
  }

  reject_enabled_denoise(options.options, {"voxcpm2.denoise"});
  reject_enabled_denoise(options.options, {"voxcpm2.load_denoiser"});
  reject_denoiser_option(options.options, {"voxcpm2.denoiser"});
  validate_session_options(options.options);

  generator_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.weight_context_mb"},
      generator_config_.weight_context_bytes);
  generator_config_.text_embedding_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.text_embedding_graph_context_mb"},
          generator_config_.text_embedding_graph_context_bytes);
  generator_config_.lm_step_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.lm_step_graph_context_mb"},
      generator_config_.lm_step_graph_context_bytes);
  generator_config_.projection_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.projection_graph_context_mb"},
          generator_config_.projection_graph_context_bytes);
  generator_config_.local_encoder_graph_context_bytes =
      runtime::parse_size_mb_option(
          options.options, {"voxcpm2.local_encoder_graph_context_mb"},
          generator_config_.local_encoder_graph_context_bytes);
  generator_config_.dit_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.dit_graph_context_mb"},
      generator_config_.dit_graph_context_bytes);
  decoder_config_.weight_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_weight_context_mb"},
      decoder_config_.weight_context_bytes);
  decoder_config_.graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_graph_context_mb"},
      decoder_config_.graph_context_bytes);
  decoder_config_.encoder_graph_context_bytes = runtime::parse_size_mb_option(
      options.options, {"voxcpm2.audiovae_encoder_graph_context_mb"},
      decoder_config_.encoder_graph_context_bytes);
  decoder_config_.latent_frame_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm2.audiovae_latent_capacity"},
      decoder_config_.latent_frame_capacity);
  decoder_config_.encoder_sample_capacity = runtime::parse_positive_i64_option(
      options.options, {"voxcpm2.audiovae_encoder_sample_capacity"},
      decoder_config_.encoder_sample_capacity);
  parse_weight_type(options.options, "voxcpm2.weight_type",
                    generator_config_.weight_storage_type);
  parse_weight_type(options.options, "voxcpm2.audiovae_weight_type",
                    decoder_config_.weight_storage_type);

  generator_ = std::make_unique<VoxCPM2FeatureGeneratorRuntime>(
      assets_, execution_context(), generator_config_);
  decoder_ = std::make_unique<VoxCPM2AudioVAEDecoderRuntime>(
      assets_, execution_context(), decoder_config_);
}

std::string VoxCPM2Session::family() const { return "voxcpm2"; }

runtime::VoiceTaskKind VoxCPM2Session::task_kind() const { return task_.task; }

runtime::RunMode VoxCPM2Session::run_mode() const { return task_.mode; }

void VoxCPM2Session::prepare(
    const runtime::SessionPreparationRequest &request) {
  (void)request;
  mark_prepared();
}

runtime::TaskResult VoxCPM2Session::run(const runtime::TaskRequest &request) {
  require_prepared("VoxCPM2 run");
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("VoxCPM2 run requires an offline session");
  }
  validate_request(request);

  const auto wall_start = Clock::now();
  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
  const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
  const auto generation_options = generation_options_from_request(request);
  const auto prompt_text =
      runtime::find_option(request.options, {"voxcpm2.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM2EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);

  runtime::TaskResult result;
  double generator_ms = 0.0;
  double decoder_ms = 0.0;
  runtime::AudioBuffer merged_audio;
  for (const auto & chunk_request : chunk_requests) {
    const auto generator_start = Clock::now();
    const auto generated = generator_->generate(
        chunk_request.text_input->text, prompt, generation_options);
    generator_ms += engine::debug::elapsed_ms(generator_start, Clock::now());

    const auto decoder_start = Clock::now();
    auto audio = decoder_->decode_features(generated.decode_features,
                                           generated.decode_patches);
    if (generated.decode_trim_patches > 0) {
      const int64_t trim_samples =
          generated.decode_trim_patches * assets_->config.patch_size *
          product(assets_->config.audio_vae.decoder_rates);
      if (trim_samples > static_cast<int64_t>(audio.samples.size())) {
        throw std::runtime_error(
            "VoxCPM2 decoded continuation trim exceeds audio length");
      }
      audio.samples.erase(
          audio.samples.begin(),
          audio.samples.begin() + static_cast<std::ptrdiff_t>(trim_samples));
    }
    decoder_ms += engine::debug::elapsed_ms(decoder_start, Clock::now());
    runtime::append_audio_buffer(merged_audio, audio);
  }
  result.audio_output = std::move(merged_audio);

  const auto wall_end = Clock::now();
  debug::timing_log_scalar("voxcpm2.generator_ms", generator_ms);
  debug::timing_log_scalar("voxcpm2.audiovae_decoder_ms", decoder_ms);
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

runtime::TaskResult
VoxCPM2Session::run_streaming(const runtime::TaskRequest &request) {
  require_prepared("VoxCPM2 run_streaming");
  if (task_.mode != runtime::RunMode::Streaming) {
    throw std::runtime_error(
        "VoxCPM2 run_streaming requires a streaming session");
  }
  validate_request(request);

  const auto wall_start = Clock::now();
  auto generation_options = generation_options_from_request(request);
  const auto prompt_text =
      runtime::find_option(request.options, {"voxcpm2.prompt_text",
                                             "prompt_text", "reference_text"})
          .value_or("");
  std::optional<runtime::AudioBuffer> reference_audio;
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    reference_audio = *request.voice->speaker->audio;
  }
  const VoxCPM2EncodedPrompt *prompt =
      encoded_prompt_for_request(request.audio_input, prompt_text,
                                 reference_audio);

  const auto generator_start = Clock::now();
  const auto generated = generator_->generate_streaming(
      request.text_input->text, prompt, generation_options);
  const auto generator_end = Clock::now();

  const auto decoder_start = Clock::now();
  runtime::TaskResult result;
  runtime::AudioBuffer merged;
  merged.sample_rate = assets_->config.audio_vae.output_sample_rate;
  merged.channels = 1;
  result.named_audio_outputs.reserve(generated.chunks.size());
  for (size_t index = 0; index < generated.chunks.size(); ++index) {
    const auto &chunk = generated.chunks[index];
    auto audio = decoder_->decode_features(chunk.decode_features,
                                           chunk.decode_patches);
    if (index == 0) {
      merged.sample_rate = audio.sample_rate;
      merged.channels = audio.channels;
    } else if (audio.sample_rate != merged.sample_rate ||
               audio.channels != merged.channels) {
      throw std::runtime_error(
          "VoxCPM2 streaming decoder chunk format changed");
    }
    merged.samples.insert(merged.samples.end(), audio.samples.begin(),
                          audio.samples.end());
    runtime::NamedAudioBuffer named;
    named.id = "chunk_" + std::to_string(index);
    named.audio = std::move(audio);
    named.meta.insert_or_assign(
        "generated_patches", std::to_string(chunk.generated_patches));
    result.named_audio_outputs.push_back(std::move(named));
  }
  result.audio_output = std::move(merged);
  const auto decoder_end = Clock::now();

  const auto wall_end = Clock::now();
  debug::timing_log_scalar(
      "voxcpm2.generator_ms",
      engine::debug::elapsed_ms(generator_start, generator_end));
  debug::timing_log_scalar(
      "voxcpm2.audiovae_decoder_ms",
      engine::debug::elapsed_ms(decoder_start, decoder_end));
  debug::timing_log_scalar("voxcpm2.streaming_chunks",
                           static_cast<double>(generated.chunks.size()));
  debug::timing_log_scalar("session.wall_ms",
                           engine::debug::elapsed_ms(wall_start, wall_end));
  return result;
}

const VoxCPM2EncodedPrompt *VoxCPM2Session::encoded_prompt_for_request(
    const std::optional<runtime::AudioBuffer> &prompt_audio,
    const std::string &prompt_text,
    const std::optional<runtime::AudioBuffer> &reference_audio) {
  if (!prompt_audio.has_value() && !reference_audio.has_value()) {
    return nullptr;
  }
  const bool cache_hit =
      encoded_prompt_cache_.has_value() &&
      encoded_prompt_cache_->prompt_text == prompt_text &&
      optional_audio_equal(encoded_prompt_cache_->prompt_audio, prompt_audio) &&
      optional_audio_equal(encoded_prompt_cache_->reference_audio,
                           reference_audio);
  debug::trace_log_scalar("voxcpm2.prompt_cache.hit", cache_hit);
  if (cache_hit) {
    debug::timing_log_scalar("voxcpm2.prompt_encode_ms", 0.0);
    return &encoded_prompt_cache_->encoded;
  }

  const auto encode_start = Clock::now();
  EncodedPromptCacheEntry entry;
  entry.prompt_text = prompt_text;
  entry.prompt_audio = prompt_audio;
  entry.reference_audio = reference_audio;
  entry.encoded =
      decoder_->encode_prompt_audio(prompt_audio, prompt_text, reference_audio);
  encoded_prompt_cache_ = std::move(entry);
  debug::timing_log_scalar("voxcpm2.prompt_encode_ms",
                           engine::debug::elapsed_ms(encode_start));
  return &encoded_prompt_cache_->encoded;
}

VoxCPM2GenerationOptions VoxCPM2Session::generation_options_from_request(
    const runtime::TaskRequest &request) const {
  VoxCPM2GenerationOptions options;
  if (const auto value = runtime::parse_i64_option(
          request.options, {"voxcpm2.min_tokens", "min_tokens"})) {
    options.min_tokens = *value;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options, {"max_tokens", "voxcpm2.max_tokens"})) {
    options.max_tokens = *value;
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"num_inference_steps", "voxcpm2.num_inference_steps"})) {
    options.num_inference_steps = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options, {"guidance_scale", "voxcpm2.guidance_scale"})) {
    options.guidance_scale = *value;
  }
  if (const auto match = runtime::find_option_match(
          request.options, {"voxcpm2.retry_badcase", "retry_badcase"})) {
    options.retry_badcase =
        runtime::parse_bool_option(match->value, match->key);
  }
  if (const auto value = runtime::parse_i64_option(
          request.options,
          {"voxcpm2.retry_badcase_max_times", "retry_badcase_max_times"})) {
    options.retry_badcase_max_times = *value;
  }
  if (const auto value = runtime::parse_finite_float_option(
          request.options, {"voxcpm2.retry_badcase_ratio_threshold",
                            "retry_badcase_ratio_threshold"})) {
    options.retry_badcase_ratio_threshold = *value;
  }
  if (const auto value = runtime::parse_u32_option(request.options,
                                                   {"voxcpm2.seed", "seed"})) {
    options.seed = *value;
  }
  options.cfm_noise_file =
      runtime::find_option(request.options,
                           {"voxcpm2.cfm_noise_file", "cfm_noise_file"})
          .value_or("");
  if (options.min_tokens < 0) {
    throw std::runtime_error("VoxCPM2 min_tokens must be non-negative");
  }
  if (options.max_tokens < 0) {
    throw std::runtime_error("VoxCPM2 max_tokens must be non-negative");
  }
  if (options.max_tokens == 0) {
    options.max_tokens = assets_->config.max_length;
  }
  if (options.min_tokens > options.max_tokens) {
    throw std::runtime_error("VoxCPM2 min_tokens must not exceed max_tokens");
  }
  if (options.max_tokens > assets_->config.max_length) {
    throw std::runtime_error(
        "VoxCPM2 max_tokens exceeds model config max_length");
  }
  if (options.num_inference_steps <= 0) {
    throw std::runtime_error(
        "VoxCPM2 num_inference_steps must be positive");
  }
  if (options.guidance_scale < 0.0F) {
    throw std::runtime_error("VoxCPM2 guidance_scale must be non-negative");
  }
  if (options.retry_badcase_max_times <= 0) {
    throw std::runtime_error(
        "VoxCPM2 retry_badcase_max_times must be positive");
  }
  if (options.retry_badcase_ratio_threshold <= 0.0F) {
    throw std::runtime_error(
        "VoxCPM2 retry_badcase_ratio_threshold must be positive");
  }
  reject_enabled_denoise(request.options, {"voxcpm2.denoise", "denoise"});
  reject_enabled_denoise(request.options,
                         {"voxcpm2.load_denoiser", "load_denoiser"});
  reject_denoiser_option(request.options, {"voxcpm2.denoiser", "denoiser"});
  return options;
}

void VoxCPM2Session::validate_request(
    const runtime::TaskRequest &request) const {
  if (!request.text_input.has_value()) {
    throw std::runtime_error("VoxCPM2 requires text input");
  }
  if (request.text_input->text.empty()) {
    throw std::runtime_error("VoxCPM2 text input must not be empty");
  }
  if (request.voice.has_value()) {
    if (request.voice->style.has_value()) {
      throw std::runtime_error(
          "VoxCPM2 C++ session does not consume style conditions");
    }
    if (request.voice->speaker.has_value()) {
      const auto &speaker = *request.voice->speaker;
      if (speaker.cached_voice_id.has_value()) {
        throw std::runtime_error("VoxCPM2 C++ session requires speaker "
                                 "reference audio, not a cached voice id");
      }
      if (!speaker.audio.has_value()) {
        throw std::runtime_error(
            "VoxCPM2 C++ session speaker condition requires audio");
      }
    }
  }
  if (!request.input_artifacts.empty()) {
    throw std::runtime_error(
        "VoxCPM2 C++ session does not consume input artifacts");
  }
}

} // namespace engine::models::voxcpm2

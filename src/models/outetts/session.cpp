#include "engine/models/outetts/session.h"

#include "engine/framework/audio/fft.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_forced_aligner/session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace engine::models::outetts {
namespace {

assets::TensorStorageType requested_weight_type(
    const runtime::SessionOptions &options) {
  const auto it = options.options.find("outetts.weight_type");
  return it == options.options.end()
             ? assets::TensorStorageType::Native
             : assets::parse_tensor_storage_type(it->second);
}

bool has_quantized_clone_weights(const runtime::SessionOptions &options,
                                 const OuteTTSAssets &model_assets) {
  constexpr std::string_view probe =
      "model.layers.0.self_attn.q_proj.weight";
  const auto source_type = assets::tensor_storage_type_for_dtype(
      model_assets.model_weights->require_metadata(probe).dtype);
  const auto requested_type = assets::resolve_tensor_storage_type(
      *model_assets.model_weights, probe, requested_weight_type(options));
  return ggml_is_quantized(
             assets::ggml_type_for_tensor_storage(source_type)) ||
         ggml_is_quantized(
             assets::ggml_type_for_tensor_storage(requested_type));
}

assets::TensorStorageType clone_weight_type(
    const runtime::SessionOptions &options,
    const OuteTTSAssets &model_assets) {
  if (options.backend.type == core::BackendType::Cuda &&
      has_quantized_clone_weights(options, model_assets)) {
    // CUDA execution with quantized OuteTTS weights diverges over the long
    // reference-codec prompt used for cloning. The GGUF stays quantized on
    // disk; only the in-memory language-model tensors are expanded to F32.
    // F16 still produces phonetic but unintelligible speech from Q8 source
    // tensors on this route, while F32 matches the coherent CPU decode.
    debug::trace_log_scalar("outetts.cuda_clone_quantized_f32_fallback", true);
    return assets::TensorStorageType::F32;
  }
  return requested_weight_type(options);
}

OuteTTSGenerateOptions
generation_options(const runtime::TaskRequest &request,
                   const OuteTTSGenerationConfig &defaults,
                   bool voice_cloning,
                   bool quantized_cloning) {
  OuteTTSGenerateOptions out;
  out.temperature = voice_cloning ? 0.4F : defaults.temperature;
  out.repetition_penalty = defaults.repetition_penalty;
  out.repetition_window = defaults.repetition_window;
  out.top_k = voice_cloning ? 40 : defaults.top_k;
  out.top_p = voice_cloning ? 0.9F : defaults.top_p;
  out.min_p = voice_cloning ? 0.05F : defaults.min_p;
  if (const auto v = runtime::parse_i64_option(request.options, {"max_tokens"}))
    out.max_new_tokens = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"temperature"}))
    out.temperature = *v;
  if (const auto v = runtime::parse_finite_float_option(request.options,
                                                        {"repetition_penalty"}))
    out.repetition_penalty = *v;
  if (const auto v =
          runtime::parse_i64_option(request.options, {"repetition_window"}))
    out.repetition_window = *v;
  if (const auto v = runtime::parse_i64_option(request.options, {"top_k"}))
    out.top_k = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"top_p"}))
    out.top_p = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"min_p"}))
    out.min_p = *v;
  out.seed = runtime::parse_u32_option(request.options, {"seed"})
                 .value_or(voice_cloning
                               ? (quantized_cloning ? 42u : 4099u)
                               : runtime::random_u32_seed());
  if (out.max_new_tokens <= 0 || out.repetition_window < 0 || out.top_k < 0 ||
      out.temperature < 0.0F || out.repetition_penalty <= 0.0F ||
      out.top_p <= 0.0F || out.top_p > 1.0F || out.min_p < 0.0F ||
      out.min_p > 1.0F) {
    throw std::runtime_error("invalid OuteTTS generation options");
  }
  return out;
}

std::vector<std::string> split_words(const std::string &text) {
  std::istringstream input(text);
  std::vector<std::string> words;
  std::string word;
  while (input >> word)
    words.push_back(word);
  return words;
}

size_t utf8_length(const std::string &text) {
  return std::max<size_t>(
      1, static_cast<size_t>(
             std::count_if(text.begin(), text.end(), [](unsigned char value) {
               return (value & 0xc0) != 0x80;
             })));
}

OuteTTSVoiceFeatures audio_features(const std::vector<float> &samples,
                                    size_t begin, size_t end) {
  OuteTTSVoiceFeatures result;
  if (begin >= end || begin >= samples.size())
    return result;
  end = std::min(end, samples.size());
  double sum_sq = 0.0;
  for (size_t i = begin; i < end; ++i)
    sum_sq += static_cast<double>(samples[i]) * samples[i];
  const double rms = std::sqrt(sum_sq / static_cast<double>(end - begin));
  result.energy =
      static_cast<int>(std::clamp(std::lround(rms * 100.0), 0l, 100l));

  const size_t count = end - begin;
  std::vector<std::complex<float>> spectrum(count / 2u + 1u);
  const auto fft = engine::audio::get_real_fft_plan(count);
  fft->forward({count}, {static_cast<std::ptrdiff_t>(sizeof(float))},
               {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))}, 0,
               samples.data() + static_cast<ptrdiff_t>(begin),
               spectrum.data());
  double magnitude_sum = 1.0e-10;
  double weighted_frequency = 0.0;
  for (size_t bin = 0; bin < spectrum.size(); ++bin) {
    const double magnitude = std::abs(spectrum[bin]);
    magnitude_sum += magnitude;
    weighted_frequency += magnitude * static_cast<double>(bin) *
                          24000.0 / static_cast<double>(count);
  }
  result.spectral_centroid = static_cast<int>(std::clamp(
      std::lround(weighted_frequency / magnitude_sum / 12000.0 * 100.0),
      0l, 100l));

  if (count >= 400 && sum_sq >= 1.0e-8) {
    constexpr int frame_length = 400;
    constexpr int hop_length = 160;
    constexpr int min_lag = 24000 / 600;
    constexpr int max_lag_exclusive = 24000 / 75;
    const size_t pad =
        (frame_length - (count % hop_length)) % hop_length;
    const size_t padded_count = count + pad;
    const size_t frames = 1u + (padded_count - frame_length) / hop_length;
    double pitch_sum = 0.0;
    std::vector<double> windowed(frame_length);
    std::vector<double> autocorrelation(frame_length);
    for (size_t frame = 0; frame < frames; ++frame) {
      const size_t offset = begin + frame * hop_length;
      for (int i = 0; i < frame_length; ++i) {
        const size_t source = offset + static_cast<size_t>(i);
        const double sample = source < end ? samples[source] : 0.0;
        const double window =
            0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i /
                                 frame_length);
        windowed[static_cast<size_t>(i)] = sample * window;
      }
      for (int lag = 0; lag < frame_length; ++lag) {
        double value = 0.0;
        for (int i = 0; i + lag < frame_length; ++i)
          value += windowed[static_cast<size_t>(i)] *
                   windowed[static_cast<size_t>(i + lag)];
        autocorrelation[static_cast<size_t>(lag)] = value;
      }
      int best_lag = min_lag;
      for (int lag = min_lag + 1; lag < max_lag_exclusive; ++lag) {
        if (autocorrelation[static_cast<size_t>(lag)] >
            autocorrelation[static_cast<size_t>(best_lag)])
          best_lag = lag;
      }
      double frequency = 75.0;
      const double beta = autocorrelation[static_cast<size_t>(best_lag)];
      if (autocorrelation[0] > 1.0e-10 &&
          beta / autocorrelation[0] > 0.3) {
        const double alpha =
            autocorrelation[static_cast<size_t>(best_lag - 1)];
        const double gamma =
            autocorrelation[static_cast<size_t>(best_lag + 1)];
        const double delta = 0.5 * (alpha - gamma) /
                             (alpha - 2.0 * beta + gamma + 1.0e-8);
        frequency = std::clamp(24000.0 / (best_lag + delta), 75.0, 600.0);
      }
      pitch_sum += frequency;
    }
    const double average_pitch = pitch_sum / static_cast<double>(frames);
    result.pitch = static_cast<int>(std::clamp(
        std::lround((average_pitch - 75.0) / 525.0 * 100.0), 0l, 100l));
  }
  return result;
}

struct ReferenceAlignment {
  std::vector<runtime::WordTimestamp> words;
  int sample_rate = 16000;
};

OuteTTSVoiceProfile
make_voice_profile(OuteTTSDacDecoder::EncodedReference encoded,
                   std::string reference_text,
                   const ReferenceAlignment *alignment) {
  auto words = split_words(reference_text);
  if (words.empty())
    throw std::runtime_error("OuteTTS reference_text must not be empty");
  const size_t frame_count =
      std::min(encoded.codebook1.size(), encoded.codebook2.size());
  if (frame_count == 0)
    throw std::runtime_error(
        "OuteTTS DAC encoder produced no reference codec frames");
  if (alignment != nullptr && alignment->words.size() != words.size()) {
    words.clear();
    for (const auto &word : alignment->words)
      words.push_back(word.word);
  }
  if (words.size() > frame_count)
    words.resize(frame_count);
  std::vector<size_t> weights(words.size());
  size_t total_weight = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    weights[i] = utf8_length(words[i]);
    total_weight += weights[i];
  }

  OuteTTSVoiceProfile profile;
  profile.text = reference_text;
  profile.global_features =
      audio_features(encoded.samples, 0, encoded.samples.size());
  debug::trace_log_scalar("outetts.reference.global.energy",
                          profile.global_features.energy);
  debug::trace_log_scalar("outetts.reference.global.spectral_centroid",
                          profile.global_features.spectral_centroid);
  debug::trace_log_scalar("outetts.reference.global.pitch",
                          profile.global_features.pitch);
  size_t start = 0;
  size_t cumulative_weight = 0;
  for (size_t word_index = 0; word_index < words.size(); ++word_index) {
    size_t feature_begin = 0;
    size_t feature_end = 0;
    size_t end = 0;
    if (alignment != nullptr && word_index < alignment->words.size()) {
      const auto &span = alignment->words[word_index].span;
      const double begin_seconds =
          static_cast<double>(span.start_sample) / alignment->sample_rate;
      const double end_seconds =
          static_cast<double>(span.end_sample) / alignment->sample_rate;
      if (word_index == 0) {
        const int64_t aligned_start =
            static_cast<int64_t>(begin_seconds * 75.0) - 20;
        start = static_cast<size_t>(std::clamp<int64_t>(
            aligned_start, 0, static_cast<int64_t>(frame_count - 1)));
      }
      int64_t aligned_end = static_cast<int64_t>(end_seconds * 75.0);
      if (word_index + 1 == words.size())
        aligned_end += 20;
      end = static_cast<size_t>(std::clamp<int64_t>(
          aligned_end, static_cast<int64_t>(start + 1),
          static_cast<int64_t>(frame_count)));
      feature_begin = static_cast<size_t>(std::clamp<int64_t>(
          static_cast<int64_t>(begin_seconds * 24000.0), 0,
          static_cast<int64_t>(encoded.samples.size())));
      feature_end = static_cast<size_t>(std::clamp<int64_t>(
          static_cast<int64_t>(end_seconds * 24000.0),
          static_cast<int64_t>(feature_begin),
          static_cast<int64_t>(encoded.samples.size())));
    } else {
      cumulative_weight += weights[word_index];
      end = word_index + 1 == words.size()
                ? frame_count
                : (frame_count * cumulative_weight + total_weight / 2) /
                      total_weight;
      feature_begin = start * 320u;
      feature_end = std::min(encoded.samples.size(), end * 320u);
    }
    end = std::max(end, std::min(frame_count, start + 1));
    OuteTTSVoiceWord word;
    word.text = words[word_index];
    word.duration =
        std::round(static_cast<double>(end - start) / 75.0 * 100.0) / 100.0;
    word.features = audio_features(encoded.samples, feature_begin, feature_end);
    word.codebook1.assign(
        encoded.codebook1.begin() + static_cast<ptrdiff_t>(start),
        encoded.codebook1.begin() + static_cast<ptrdiff_t>(end));
    word.codebook2.assign(
        encoded.codebook2.begin() + static_cast<ptrdiff_t>(start),
        encoded.codebook2.begin() + static_cast<ptrdiff_t>(end));
    const std::string trace_prefix =
        "outetts.reference.word." + std::to_string(word_index);
    debug::trace_log_scalar(trace_prefix + ".text", word.text);
    debug::trace_log_scalar(trace_prefix + ".duration", word.duration);
    debug::trace_log_scalar(trace_prefix + ".energy", word.features.energy);
    debug::trace_log_scalar(trace_prefix + ".spectral_centroid",
                            word.features.spectral_centroid);
    debug::trace_log_scalar(trace_prefix + ".pitch", word.features.pitch);
    profile.words.push_back(std::move(word));
    start = end;
  }
  return profile;
}

std::optional<ReferenceAlignment> align_reference(
    const runtime::SessionOptions &options,
    const OuteTTSAssets &model_assets,
    const runtime::AudioBuffer &audio,
    const std::string &text,
    const std::string &language) {
  const auto model_path = runtime::find_option(
      options.options,
      {"outetts.aligner_model_path", "outetts.forced_aligner_model_path"});
  runtime::SessionOptions aligner_options;
  aligner_options.backend = options.backend;
  for (const auto &[key, value] : options.options) {
    if (key.rfind("qwen3_forced_aligner.", 0) == 0)
      aligner_options.options.emplace(key, value);
  }
  std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets>
      aligner_assets;
  if (model_path.has_value()) {
    aligner_assets = engine::models::qwen3_asr::load_qwen3_asr_assets(
        std::filesystem::path(*model_path), "qwen3_forced_aligner");
  } else {
    aligner_assets = model_assets.embedded_aligner;
  }
  if (aligner_assets == nullptr) {
    throw std::runtime_error(
        "OuteTTS voice cloning requires a GGUF with an embedded Qwen3 "
        "Forced Aligner or --session-option "
        "outetts.aligner_model_path=<path>");
  }
  engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession session(
      {runtime::VoiceTaskKind::Alignment, runtime::RunMode::Offline},
      std::move(aligner_options), aligner_assets);
  runtime::TaskRequest request;
  request.audio_input = audio;
  request.text_input = runtime::Transcript{text, language};
  request.options["audio_chunk_mode"] = "none";
  session.prepare(runtime::build_preparation_request(request));
  auto result = session.run(request);
  if (result.word_timestamps.empty())
    throw std::runtime_error("OuteTTS reference aligner returned no words");
  return ReferenceAlignment{std::move(result.word_timestamps),
                            aligner_assets->config.sample_rate};
}

const runtime::AudioBuffer *
reference_audio(const runtime::TaskRequest &request) {
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    return &*request.voice->speaker->audio;
  }
  return request.audio_input.has_value() ? &*request.audio_input : nullptr;
}

} // namespace

OuteTTSSession::OuteTTSSession(runtime::TaskSpec task,
                               runtime::SessionOptions options,
                               std::shared_ptr<const OuteTTSAssets> assets)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)),
      tokenizer_(assets_),
      dac_(assets_, execution_context(),
           runtime::parse_size_mb_option(options.options,
                                         {"outetts.dac_weight_context_mb"},
                                         1024ull * 1024ull * 1024ull),
           runtime::parse_size_mb_option(options.options,
                                         {"outetts.dac_graph_context_mb"},
                                         1536ull * 1024ull * 1024ull),
           assets::TensorStorageType::F32) {
  if (assets_ == nullptr)
    throw std::runtime_error("OuteTTS session requires assets");
  if ((task_.task != runtime::VoiceTaskKind::Tts &&
       task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
      task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "OuteTTS supports offline TTS and voice cloning only");
  }
}

OuteTTSLlamaRuntime &OuteTTSSession::llama(bool voice_cloning) {
  auto &runtime_slot = voice_cloning ? clone_llama_ : llama_;
  if (runtime_slot == nullptr) {
    const auto storage_type =
        voice_cloning ? clone_weight_type(options(), *assets_)
                      : requested_weight_type(options());
    runtime_slot = std::make_unique<OuteTTSLlamaRuntime>(
        assets_, options().backend.type, options().backend.device,
        std::max(1, options().backend.threads),
        runtime::parse_size_mb_option(options().options,
                                      {"outetts.llama_weight_context_mb"},
                                      4096ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options().options,
                                      {"outetts.constant_context_mb"},
                                      256ull * 1024ull * 1024ull),
        storage_type);
  }
  return *runtime_slot;
}

std::string OuteTTSSession::family() const { return "outetts"; }
runtime::VoiceTaskKind OuteTTSSession::task_kind() const { return task_.task; }
runtime::RunMode OuteTTSSession::run_mode() const { return task_.mode; }
void OuteTTSSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  voice_profile_.reset();
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    const auto reference_text =
        runtime::find_option(request.options, {"reference_text"}).value_or("");
    if (reference_text.empty()) {
      throw std::runtime_error(
          "OuteTTS voice cloning requires --reference-text");
    }
    const auto &audio = *request.voice->speaker->audio;
    const auto language = runtime::find_option(
                              request.options, {"reference_language"})
                              .value_or(request.text.has_value() &&
                                                !request.text->language.empty()
                                            ? request.text->language
                                            : "en");
    const auto alignment =
        align_reference(options(), *assets_, audio, reference_text, language);
    voice_profile_ = make_voice_profile(
        dac_.encode_reference(audio), reference_text,
        alignment.has_value() ? &*alignment : nullptr);
  }
  mark_prepared();
}

runtime::TaskResult OuteTTSSession::run(const runtime::TaskRequest &request) {
  require_prepared("OuteTTS run");
  if (!request.text_input.has_value() || request.text_input->text.empty()) {
    throw std::runtime_error("OuteTTS requires text input");
  }
  const auto *voice_audio = reference_audio(request);
  std::optional<OuteTTSVoiceProfile> request_profile;
  if (voice_audio != nullptr && !voice_profile_.has_value()) {
    const auto reference_text =
        runtime::find_option(request.options, {"reference_text"}).value_or("");
    if (reference_text.empty())
      throw std::runtime_error(
          "OuteTTS voice cloning requires --reference-text");
    const auto language = runtime::find_option(
                              request.options, {"reference_language"})
                              .value_or(request.text_input.has_value() &&
                                                !request.text_input->language.empty()
                                            ? request.text_input->language
                                            : "en");
    const auto alignment =
        align_reference(options(), *assets_, *voice_audio, reference_text,
                        language);
    request_profile = make_voice_profile(
        dac_.encode_reference(*voice_audio), reference_text,
        alignment.has_value() ? &*alignment : nullptr);
  }
  const OuteTTSVoiceProfile *profile =
      request_profile.has_value()
          ? &*request_profile
          : (voice_profile_.has_value() ? &*voice_profile_ : nullptr);
  if (task_.task == runtime::VoiceTaskKind::VoiceCloning &&
      profile == nullptr) {
    throw std::runtime_error(
        "OuteTTS voice cloning requires --voice-ref and --reference-text");
  }
  const auto prompt =
      profile != nullptr
          ? tokenizer_.build_clone_prompt(request.text_input->text, *profile)
          : tokenizer_.build_prompt(request.text_input->text);
  const bool quantized_cloning =
      profile != nullptr && has_quantized_clone_weights(options(), *assets_);
  auto generate_options = generation_options(
      request, assets_->generation, profile != nullptr, quantized_cloning);
  const auto generated = llama(profile != nullptr).generate(
      prompt, generate_options,
      tokenizer_.eos_id(), tokenizer_.audio_end_id());
  std::vector<int32_t> c1;
  std::vector<int32_t> c2;
  for (const int32_t token : generated)
    tokenizer_.append_audio_code(token, c1, c2);
  const size_t pairs = std::min(c1.size(), c2.size());
  c1.resize(pairs);
  c2.resize(pairs);
  if (pairs == 0) {
    std::string detail;
    for (size_t i = 0; i < std::min<size_t>(generated.size(), 12); ++i) {
      detail += (i == 0 ? "" : ",") + std::to_string(generated[i]);
    }
    throw std::runtime_error(
        "OuteTTS generated no complete DAC code pairs (tokens=" + detail + ")");
  }
  runtime::TaskResult result;
  result.audio_output = dac_.decode(c1, c2);
  return result;
}

} // namespace engine::models::outetts

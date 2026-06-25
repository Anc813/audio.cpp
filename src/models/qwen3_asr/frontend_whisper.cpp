#include "engine/models/qwen3_asr/frontend_whisper.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

constexpr int kMinInputSamples = 8000;
constexpr float kLogFloor = 1.0e-10F;

using Clock = std::chrono::steady_clock;

void validate_audio_input(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Qwen3 ASR audio sample_rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Qwen3 ASR audio channels must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Qwen3 ASR audio is empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Qwen3 ASR interleaved audio size is not divisible by channel count");
    }
}

std::vector<float> normalize_audio(const runtime::AudioBuffer & audio, int sample_rate) {
    validate_audio_input(audio);
    auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples,
        audio.sample_rate,
        audio.channels,
        sample_rate);
    engine::audio::normalize_peak_to_unit_range_and_clamp_in_place(mono);
    if (mono.size() < kMinInputSamples) {
        mono.resize(kMinInputSamples, 0.0F);
    }
    return mono;
}

}  // namespace

Qwen3ASRWhisperFrontend::Qwen3ASRWhisperFrontend(std::shared_ptr<const Qwen3ASRAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Qwen3 ASR Whisper frontend requires assets");
    }
    const auto & config = assets_->config.frontend;
    filterbank_ = engine::audio::MelFilterbank().build_sparse(
        engine::audio::MelFilterbankConfig{
            config.sample_rate,
            config.n_fft,
            config.feature_size,
            0.0F,
            static_cast<float>(config.sample_rate) / 2.0F,
            true,
        });
}

Qwen3ASRAudioFeatures Qwen3ASRWhisperFrontend::extract(const runtime::AudioBuffer & audio) const {
    const auto normalize_start = Clock::now();
    const auto & config = assets_->config.frontend;
    if (config.sample_rate <= 0 || config.feature_size <= 0 || config.hop_length <= 0 || config.n_fft <= 0) {
        throw std::runtime_error("Qwen3 ASR Whisper frontend config is invalid");
    }
    auto samples = normalize_audio(audio, config.sample_rate);
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    const auto normalize_end = Clock::now();

    const engine::audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Default,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto stft_start = Clock::now();
    auto magnitude = engine::audio::STFT().compute_magnitude(samples, window, 1, sample_count, stft_config);
    const auto stft_end = Clock::now();
    if (magnitude.shape.size() != 3) {
        throw std::runtime_error("Qwen3 ASR STFT returned unexpected rank");
    }
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    if (stft_frames <= 1) {
        throw std::runtime_error("Qwen3 ASR STFT produced too few frames");
    }
    const int64_t frames = stft_frames - 1;

    const auto mel_start = Clock::now();
    auto mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        freq_bins,
        stft_frames,
        frames,
        filterbank_);
    const auto mel_end = Clock::now();
    if (mel.shape.size() != 3 || mel.shape[1] != config.feature_size || mel.shape[2] != frames) {
        throw std::runtime_error("Qwen3 ASR mel frontend returned unexpected shape");
    }

    const auto log_start = Clock::now();
    float max_log = -INFINITY;
    for (float & value : mel.values) {
        value = std::log10(std::max(value, kLogFloor));
        max_log = std::max(max_log, value);
    }
    const float floor = max_log - 8.0F;
    for (float & value : mel.values) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    const auto log_end = Clock::now();

    Qwen3ASRAudioFeatures out;
    out.values = std::move(mel.values);
    out.attention_mask.assign(static_cast<size_t>(frames), 1);
    out.mel_bins = config.feature_size;
    out.frames = frames;
    out.encoder_tokens = qwen3_asr_audio_encoder_token_count(frames);
    debug::timing_log_scalar("qwen3_asr.frontend.normalize_ms", engine::debug::elapsed_ms(normalize_start, normalize_end));
    debug::timing_log_scalar("qwen3_asr.frontend.stft_ms", engine::debug::elapsed_ms(stft_start, stft_end));
    debug::timing_log_scalar("qwen3_asr.frontend.mel_ms", engine::debug::elapsed_ms(mel_start, mel_end));
    debug::timing_log_scalar("qwen3_asr.frontend.log_norm_ms", engine::debug::elapsed_ms(log_start, log_end));
    return out;
}

}  // namespace engine::models::qwen3_asr

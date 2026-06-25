#include "engine/framework/audio/waveform_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::audio {

void apply_preemphasis_in_place(float * mono_samples, size_t sample_count, float coefficient) {
    if (mono_samples == nullptr && sample_count > 0) {
        throw std::runtime_error("preemphasis requires a valid sample buffer");
    }
    if (coefficient == 0.0F || sample_count <= 1) {
        return;
    }
    for (size_t i = sample_count - 1; i > 0; --i) {
        mono_samples[i] -= coefficient * mono_samples[i - 1];
    }
}

void apply_preemphasis_in_place(std::vector<float> & mono_samples, float coefficient) {
    apply_preemphasis_in_place(mono_samples.data(), mono_samples.size(), coefficient);
}

std::vector<float> apply_preemphasis(std::vector<float> mono_samples, float coefficient) {
    apply_preemphasis_in_place(mono_samples, coefficient);
    return mono_samples;
}

std::vector<float> copy_or_zero_pad_samples_to_count(
    const std::vector<float> & samples,
    size_t output_sample_count) {
    std::vector<float> out(output_sample_count, 0.0F);
    const size_t copy_count = std::min(out.size(), samples.size());
    std::copy_n(samples.begin(), copy_count, out.begin());
    return out;
}

std::vector<float> zero_pad_samples_to_multiple(
    const std::vector<float> & samples,
    int64_t sample_multiple) {
    if (sample_multiple <= 0) {
        throw std::runtime_error("audio sample padding multiple must be positive");
    }
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    const int64_t multiples = (sample_count + sample_multiple - 1) / sample_multiple;
    const int64_t padded_samples = multiples * sample_multiple;
    std::vector<float> out = samples;
    out.resize(static_cast<size_t>(padded_samples), 0.0F);
    return out;
}

void truncate_samples_to_count(std::vector<float> & samples, size_t max_sample_count) {
    if (samples.size() > max_sample_count) {
        samples.resize(max_sample_count);
    }
}

void normalize_peak_to_unit_range_and_clamp_in_place(std::vector<float> & samples) {
    float peak = 0.0F;
    for (const float sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }
    if (peak > 1.0F) {
        for (float & sample : samples) {
            sample /= peak;
        }
    }
    for (float & sample : samples) {
        sample = std::clamp(sample, -1.0F, 1.0F);
    }
}

std::vector<float> reflect_pad_samples(
    const std::vector<float> & samples,
    int64_t left_pad_samples,
    int64_t right_pad_samples) {
    if (samples.empty()) {
        throw std::runtime_error("reflect padding requires non-empty samples");
    }
    if (left_pad_samples < 0 || right_pad_samples < 0) {
        throw std::runtime_error("reflect padding size is invalid for sample count");
    }
    if (samples.size() == 1 && (left_pad_samples > 0 || right_pad_samples > 0)) {
        throw std::runtime_error("reflect padding requires at least two samples when padding is non-zero");
    }
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    std::vector<float> out(static_cast<size_t>(sample_count + left_pad_samples + right_pad_samples), 0.0F);
    for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        int64_t index = i - left_pad_samples;
        while (index < 0 || index >= sample_count) {
            if (index < 0) {
                index = -index;
            }
            if (index >= sample_count) {
                index = 2 * sample_count - index - 2;
            }
        }
        out[static_cast<size_t>(i)] = samples[static_cast<size_t>(index)];
    }
    return out;
}

float root_mean_square_or_throw(const std::vector<float> & samples) {
    if (samples.empty()) {
        throw std::runtime_error("RMS input samples must not be empty");
    }
    double sum = 0.0;
    for (const float sample : samples) {
        sum += static_cast<double>(sample) * static_cast<double>(sample);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(samples.size())));
}

std::vector<int16_t> float_to_pcm16_clipped(
    const std::vector<float> & samples,
    Pcm16QuantizeMode quantize_mode) {
    std::vector<int16_t> pcm(samples.size(), 0);
    for (size_t i = 0; i < samples.size(); ++i) {
        const float clipped = std::clamp(samples[i] * 32768.0F, -32768.0F, 32767.0F);
        if (quantize_mode == Pcm16QuantizeMode::RoundToNearest) {
            pcm[i] = static_cast<int16_t>(std::lrint(clipped));
        } else {
            pcm[i] = static_cast<int16_t>(clipped);
        }
    }
    return pcm;
}

std::vector<float> pcm16_to_float_unit_range(const std::vector<int16_t> & pcm_samples) {
    std::vector<float> samples(pcm_samples.size(), 0.0F);
    for (size_t i = 0; i < pcm_samples.size(); ++i) {
        samples[i] = static_cast<float>(pcm_samples[i]) / 32768.0F;
    }
    return samples;
}

}  // namespace engine::audio

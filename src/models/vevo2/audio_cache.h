#pragma once

#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace engine::models::vevo2 {

inline uint64_t fnv1a_update(uint64_t hash, const void * data, size_t bytes) {
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    const auto * ptr = static_cast<const unsigned char *>(data);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= static_cast<uint64_t>(ptr[index]);
        hash *= kFnvPrime;
    }
    return hash;
}

inline uint64_t audio_buffer_key(const runtime::AudioBuffer & audio) {
    uint64_t hash = 1469598103934665603ull;
    hash = fnv1a_update(hash, &audio.sample_rate, sizeof(audio.sample_rate));
    hash = fnv1a_update(hash, &audio.channels, sizeof(audio.channels));
    const size_t samples = audio.samples.size();
    hash = fnv1a_update(hash, &samples, sizeof(samples));
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = fnv1a_update(hash, &bits, sizeof(bits));
    }
    return hash;
}

inline bool matches_audio_cache_key(
    uint64_t stored_key,
    uint64_t requested_key,
    int sample_rate,
    int channels,
    size_t samples,
    const runtime::AudioBuffer & audio) {
    return stored_key == requested_key && sample_rate == audio.sample_rate && channels == audio.channels &&
           samples == audio.samples.size();
}

}  // namespace engine::models::vevo2

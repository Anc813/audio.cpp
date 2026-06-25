#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::miotts {

struct MioTTSConfig {
    std::string model_type;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t eos_token_id = 0;
    int64_t pad_token_id = 0;
    int64_t speech_token_start_id = 0;
    int64_t speech_token_count = 12800;
    int64_t max_tokens = 700;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
    bool do_sample = true;
    int top_k = 50;
    float top_p = 1.0F;
    float temperature = 0.8F;
    float repetition_penalty = 1.0F;
    float presence_penalty = 0.0F;
    float frequency_penalty = 0.0F;
};

struct MioTTSAssetPaths {
    std::filesystem::path model_root;
    std::filesystem::path config_path;
    std::filesystem::path generation_config_path;
    std::filesystem::path model_weights_path;
    std::filesystem::path tokenizer_config_path;
    std::filesystem::path tokenizer_vocab_path;
    std::filesystem::path tokenizer_merges_path;
    std::filesystem::path tokenizer_json_path;
};

struct MioTTSAssets {
    MioTTSAssetPaths paths;
    MioTTSConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
};

MioTTSAssetPaths resolve_miotts_assets(const std::filesystem::path & model_path);
std::shared_ptr<const MioTTSAssets> load_miotts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::miotts

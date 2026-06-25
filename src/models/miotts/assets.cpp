#include "engine/models/miotts/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>

namespace engine::models::miotts {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("MioTTS model path does not exist: " + model_path.string());
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"config", "config.json", true},
        {"generation_config", "generation_config.json", true},
        {"weights", "model.safetensors", true},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"vocab", "vocab.json", true},
        {"merges", "merges.txt", true},
        {"tokenizer_json", "tokenizer.json", false},
    });
    return resources;
}

int64_t require_added_token_id(
    const engine::io::json::Value & tokenizer_config,
    const std::string & content) {
    const auto & tokens = tokenizer_config.require("added_tokens_decoder").as_object();
    for (const auto & [key, value] : tokens) {
        if (value.require("content").as_string() == content) {
            return std::stoll(key);
        }
    }
    throw std::runtime_error("MioTTS tokenizer_config is missing added token: " + content);
}

MioTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto generation = resources.parse_json("generation_config");
    const auto tokenizer_config = resources.parse_json("tokenizer_config");

    MioTTSConfig config;
    config.model_type = root.require("model_type").as_string();
    config.vocab_size = root.require("vocab_size").as_i64();
    config.hidden_size = root.require("hidden_size").as_i64();
    config.intermediate_size = root.require("intermediate_size").as_i64();
    config.num_hidden_layers = root.require("num_hidden_layers").as_i64();
    config.num_attention_heads = root.require("num_attention_heads").as_i64();
    config.num_key_value_heads = root.require("num_key_value_heads").as_i64();
    config.head_dim = json::optional_i64(root, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);
    config.eos_token_id = json::optional_i64(generation, "eos_token_id", json::optional_i64(root, "eos_token_id", config.eos_token_id));
    config.pad_token_id = json::optional_i64(generation, "pad_token_id", json::optional_i64(root, "pad_token_id", config.pad_token_id));
    config.max_tokens = 700;
    config.do_sample = json::optional_bool(generation, "do_sample", config.do_sample);
    config.top_k = static_cast<int>(json::optional_i64(generation, "top_k", config.top_k));
    config.top_p = json::optional_f32(generation, "top_p", config.top_p);
    config.temperature = json::optional_f32(generation, "temperature", config.temperature);
    config.repetition_penalty = json::optional_f32(generation, "repetition_penalty", config.repetition_penalty);
    config.speech_token_start_id = require_added_token_id(tokenizer_config, "<|s_0|>");
    if (config.model_type != "qwen3") {
        throw std::runtime_error("MioTTS expects a qwen3 causal LM config");
    }
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("MioTTS currently expects tied input/output embeddings");
    }
    if (config.speech_token_start_id <= 0 || config.speech_token_start_id + config.speech_token_count > config.vocab_size) {
        throw std::runtime_error("MioTTS speech token range is outside the vocabulary");
    }
    return config;
}

void fill_paths(MioTTSAssetPaths & paths, assets::ResourceBundle & resources) {
    const auto * tokenizer_json = resources.find_file("tokenizer_json");
    paths.model_root = resources.model_root();
    paths.config_path = resources.require_file("config");
    paths.generation_config_path = resources.require_file("generation_config");
    paths.model_weights_path = resources.require_file("weights");
    paths.tokenizer_config_path = resources.require_file("tokenizer_config");
    paths.tokenizer_vocab_path = resources.require_file("vocab");
    paths.tokenizer_merges_path = resources.require_file("merges");
    paths.tokenizer_json_path = tokenizer_json != nullptr ? *tokenizer_json : std::filesystem::path{};
}

}  // namespace

MioTTSAssetPaths resolve_miotts_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    MioTTSAssetPaths paths;
    fill_paths(paths, resources);
    return paths;
}

std::shared_ptr<const MioTTSAssets> load_miotts_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    MioTTSAssets assets;
    fill_paths(assets.paths, resources);
    assets.config = parse_config(resources);
    assets.model_weights = resources.open_tensor_source("weights");
    return std::make_shared<MioTTSAssets>(std::move(assets));
}

}  // namespace engine::models::miotts

#include "engine/models/qwen3_asr/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace json = engine::io::json;
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Qwen3 ASR model path does not exist: " + model_path.string());
}

Qwen3ASRAudioEncoderConfig parse_audio_encoder_config(const engine::io::json::Value & value) {
    Qwen3ASRAudioEncoderConfig config;
    config.num_mel_bins = value.require("num_mel_bins").as_i64();
    config.encoder_layers = value.require("encoder_layers").as_i64();
    config.encoder_attention_heads = value.require("encoder_attention_heads").as_i64();
    config.encoder_ffn_dim = value.require("encoder_ffn_dim").as_i64();
    config.d_model = value.require("d_model").as_i64();
    config.max_source_positions = value.require("max_source_positions").as_i64();
    config.n_window = value.require("n_window").as_i64();
    config.n_window_infer = value.require("n_window_infer").as_i64();
    config.conv_chunksize = value.require("conv_chunksize").as_i64();
    config.downsample_hidden_size = value.require("downsample_hidden_size").as_i64();
    config.output_dim = value.require("output_dim").as_i64();
    config.activation_function = value.require("activation_function").as_string();
    if (config.activation_function != "gelu") {
        throw std::runtime_error("Qwen3 ASR currently supports gelu audio activation");
    }
    return config;
}

Qwen3ASRTextDecoderConfig parse_text_decoder_config(
    const engine::io::json::Value & thinker_config,
    const engine::io::json::Value & text_config) {
    Qwen3ASRTextDecoderConfig config;
    config.vocab_size = text_config.require("vocab_size").as_i64();
    config.output_size = json::optional_i64(thinker_config, "classify_num", config.vocab_size);
    config.hidden_size = text_config.require("hidden_size").as_i64();
    config.intermediate_size = text_config.require("intermediate_size").as_i64();
    config.num_hidden_layers = text_config.require("num_hidden_layers").as_i64();
    config.num_attention_heads = text_config.require("num_attention_heads").as_i64();
    config.num_key_value_heads = text_config.require("num_key_value_heads").as_i64();
    config.head_dim = json::optional_i64(text_config, "head_dim", config.hidden_size / config.num_attention_heads);
    config.max_position_embeddings = text_config.require("max_position_embeddings").as_i64();
    config.audio_token_id = thinker_config.require("audio_token_id").as_i64();
    config.audio_start_token_id = thinker_config.require("audio_start_token_id").as_i64();
    config.audio_end_token_id = thinker_config.require("audio_end_token_id").as_i64();
    config.pad_token_id = json::optional_i64(text_config, "pad_token_id", config.pad_token_id);
    config.rms_norm_eps = json::optional_f32(text_config, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(text_config, "rope_theta", config.rope_theta);
    const auto * rope_scaling = text_config.find("rope_scaling");
    if (rope_scaling != nullptr && rope_scaling->is_object()) {
        config.mrope_section = json::optional_i64_array_or_scalar(*rope_scaling, "mrope_section", config.mrope_section);
    }
    return config;
}

Qwen3ASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto & thinker_config = root.require("thinker_config");
    const auto & audio_config = thinker_config.require("audio_config");
    const auto & text_config = thinker_config.require("text_config");

    Qwen3ASRConfig config;
    config.model_type = root.require("model_type").as_string();
    config.thinker_model_type = thinker_config.require("model_type").as_string();
    config.model_size = json::optional_string(root, "model_size", config.model_type);
    config.classify_num = json::optional_i64(root.require("thinker_config"), "classify_num", 0);
    config.timestamp_token_id = json::optional_i64(root, "timestamp_token_id", 0);
    config.timestamp_segment_time_ms = json::optional_i64(root, "timestamp_segment_time", 0);
    config.audio_encoder = parse_audio_encoder_config(audio_config);
    config.text_decoder = parse_text_decoder_config(thinker_config, text_config);

    const auto generation = resources.parse_json("generation_config");
    config.max_new_tokens = json::optional_i64(generation, "max_new_tokens", config.max_new_tokens);
    config.text_decoder.pad_token_id = json::optional_i64(generation, "pad_token_id", config.text_decoder.pad_token_id);
    config.text_decoder.eos_token_ids = json::require_i64_array_or_scalar(generation, "eos_token_id");

    const auto preprocessor = resources.parse_json("preprocessor_config");
    config.frontend.sample_rate = static_cast<int>(json::optional_i64(preprocessor, "sampling_rate", config.frontend.sample_rate));
    config.frontend.feature_size = preprocessor.require("feature_size").as_i64();
    config.frontend.hop_length = preprocessor.require("hop_length").as_i64();
    config.frontend.n_fft = preprocessor.require("n_fft").as_i64();
    config.sample_rate = config.frontend.sample_rate;
    if (config.frontend.feature_size != config.audio_encoder.num_mel_bins) {
        throw std::runtime_error("Qwen3 ASR frontend feature size does not match audio encoder config");
    }

    config.supported_languages.clear();
    config.supported_languages.push_back("Auto");
    for (const auto & language : root.require("support_languages").as_array()) {
        config.supported_languages.push_back(language.as_string());
    }
    return config;
}

assets::ResourceBundle make_resource_bundle(const std::filesystem::path & model_path) {
    assets::ResourceBundle resources(resolve_model_root(model_path));
    resources.add_model_files({
        {"config", "config.json", true},
        {"generation_config", "generation_config.json", true},
        {"preprocessor_config", "preprocessor_config.json", true},
        {"chat_template", "chat_template.json", false},
        {"weights", "model.safetensors", true},
        {"tokenizer_config", "tokenizer_config.json", true},
        {"vocab", "vocab.json", true},
        {"merges", "merges.txt", true},
    });
    return resources;
}

}  // namespace

Qwen3ASRAssetPaths resolve_qwen3_asr_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    const auto * chat_template = resources.find_file("chat_template");
    Qwen3ASRAssetPaths paths;
    paths.model_root = resources.model_root();
    paths.config_path = resources.require_file("config");
    paths.generation_config_path = resources.require_file("generation_config");
    paths.preprocessor_config_path = resources.require_file("preprocessor_config");
    paths.chat_template_path = chat_template != nullptr ? *chat_template : std::filesystem::path{};
    paths.model_weights_path = resources.require_file("weights");
    paths.tokenizer_config_path = resources.require_file("tokenizer_config");
    paths.tokenizer_vocab_path = resources.require_file("vocab");
    paths.tokenizer_merges_path = resources.require_file("merges");
    return paths;
}

std::shared_ptr<const Qwen3ASRAssets> load_qwen3_asr_assets(const std::filesystem::path & model_path) {
    auto resources = make_resource_bundle(model_path);
    Qwen3ASRAssets assets;
    const auto * chat_template = resources.find_file("chat_template");
    assets.paths.model_root = resources.model_root();
    assets.paths.config_path = resources.require_file("config");
    assets.paths.generation_config_path = resources.require_file("generation_config");
    assets.paths.preprocessor_config_path = resources.require_file("preprocessor_config");
    assets.paths.chat_template_path = chat_template != nullptr ? *chat_template : std::filesystem::path{};
    assets.paths.model_weights_path = resources.require_file("weights");
    assets.paths.tokenizer_config_path = resources.require_file("tokenizer_config");
    assets.paths.tokenizer_vocab_path = resources.require_file("vocab");
    assets.paths.tokenizer_merges_path = resources.require_file("merges");
    assets.config = parse_config(resources);
    assets.model_weights = resources.open_tensor_source("weights");
    return std::make_shared<Qwen3ASRAssets>(std::move(assets));
}

}  // namespace engine::models::qwen3_asr

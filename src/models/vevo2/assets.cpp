#include "engine/models/vevo2/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::vevo2 {
namespace json = engine::io::json;
namespace {

std::filesystem::path require_existing_file(const std::filesystem::path & path, const char * role) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("Vevo2 missing ") + role + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path require_existing_directory(const std::filesystem::path & path, const char * role) {
    if (!engine::io::is_existing_directory(path)) {
        throw std::runtime_error(std::string("Vevo2 missing ") + role + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::string option_or_empty(const runtime::ModelLoadRequest & request, const char * key) {
    const auto it = request.options.find(key);
    return it == request.options.end() ? std::string{} : it->second;
}

std::string strip_jsonc_comments(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    bool escaped = false;
    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (in_string) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (ch == '/' && index + 1 < text.size() && text[index + 1] == '/') {
            while (index < text.size() && text[index] != '\n') {
                ++index;
            }
            if (index < text.size()) {
                out.push_back('\n');
            }
            continue;
        }
        out.push_back(ch);
    }
    return out;
}

std::string strip_jsonc_trailing_commas(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_string = false;
    bool escaped = false;
    for (size_t index = 0; index < text.size(); ++index) {
        const char ch = text[index];
        if (in_string) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            out.push_back(ch);
            continue;
        }
        if (ch == ',') {
            size_t lookahead = index + 1;
            while (lookahead < text.size() && std::isspace(static_cast<unsigned char>(text[lookahead])) != 0) {
                ++lookahead;
            }
            if (lookahead < text.size() && (text[lookahead] == '}' || text[lookahead] == ']')) {
                continue;
            }
        }
        out.push_back(ch);
    }
    return out;
}

engine::io::json::Value parse_jsonc_file(const std::filesystem::path & path) {
    const auto without_comments = strip_jsonc_comments(engine::io::read_text_file(path));
    return engine::io::json::parse(strip_jsonc_trailing_commas(without_comments));
}

engine::modules::WhisperEmbeddingConfig parse_whisper_config(const std::filesystem::path & whisper_config_path) {
    const auto root = engine::io::json::parse_file(whisper_config_path);
    const auto & audio = root.require("audio_encoder");
    engine::modules::WhisperEmbeddingConfig config;
    config.n_mels = audio.require("n_mels").as_i64();
    config.n_audio_ctx = audio.require("n_audio_ctx").as_i64();
    config.n_audio_state = audio.require("n_audio_state").as_i64();
    config.n_audio_head = audio.require("n_audio_head").as_i64();
    config.n_audio_layer = audio.require("n_audio_layer").as_i64();
    if (config.n_mels <= 0 || config.n_audio_ctx <= 0 || config.n_audio_state <= 0 ||
        config.n_audio_head <= 0 || config.n_audio_layer <= 0) {
        throw std::runtime_error("Vevo2 Whisper config contains non-positive dimensions");
    }
    return config;
}

Vevo2ARConfig parse_ar_config(
    const std::filesystem::path & ar_config_path,
    const std::filesystem::path & generation_config_path) {
    const auto root = engine::io::json::parse_file(ar_config_path);
    Vevo2ARConfig config;
    config.model_type = root.require("model_type").as_string();
    config.vocab_size = root.require("vocab_size").as_i64();
    config.hidden_size = root.require("hidden_size").as_i64();
    config.intermediate_size = root.require("intermediate_size").as_i64();
    config.num_hidden_layers = root.require("num_hidden_layers").as_i64();
    config.num_attention_heads = root.require("num_attention_heads").as_i64();
    config.num_key_value_heads = root.require("num_key_value_heads").as_i64();
    config.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    config.max_window_layers = json::optional_i64(root, "max_window_layers", config.num_hidden_layers);
    config.bos_token_id = json::optional_i64(root, "bos_token_id", config.bos_token_id);
    config.eos_token_id = json::optional_i64(root, "eos_token_id", config.eos_token_id);
    config.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", config.rms_norm_eps);
    config.rope_theta = json::optional_f32(root, "rope_theta", config.rope_theta);
    config.tie_word_embeddings = json::optional_bool(root, "tie_word_embeddings", config.tie_word_embeddings);

    const auto generation = engine::io::json::parse_file(generation_config_path);
    config.pad_token_id = json::optional_i64(generation, "pad_token_id", config.pad_token_id);
    config.generation_eos_token_ids =
        json::optional_i64_array_or_scalar(generation, "eos_token_id", {config.eos_token_id});
    config.generation_top_k = json::optional_i64(generation, "top_k", config.generation_top_k);
    config.generation_top_p = json::optional_f32(generation, "top_p", config.generation_top_p);
    config.generation_temperature = json::optional_f32(generation, "temperature", config.generation_temperature);
    config.generation_repetition_penalty =
        json::optional_f32(generation, "repetition_penalty", config.generation_repetition_penalty);
    config.generation_do_sample = json::optional_bool(generation, "do_sample", config.generation_do_sample);

    if (config.model_type != "qwen2") {
        throw std::runtime_error("Vevo2 AR currently expects qwen2 model_type");
    }
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_hidden_layers <= 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.vocab_size <= 0) {
        throw std::runtime_error("Vevo2 AR config contains non-positive dimensions");
    }
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Vevo2 AR hidden_size must be divisible by num_attention_heads");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Vevo2 AR attention heads must be divisible by key-value heads");
    }
    if (config.generation_eos_token_ids.empty()) {
        throw std::runtime_error("Vevo2 AR generation config requires at least one eos token id");
    }
    return config;
}

Vevo2CocoTokenizerConfig parse_coco_config(const engine::io::json::Value & value, const std::string & expected_type) {
    Vevo2CocoTokenizerConfig config;
    config.coco_type = value.require("coco_type").as_string();
    config.downsample_rate = value.require("downsample_rate").as_i64();
    config.codebook_size = value.require("codebook_size").as_i64();
    config.hidden_size = value.require("hidden_size").as_i64();
    config.codebook_dim = value.require("codebook_dim").as_i64();
    const auto & encoder = value.require("encoder");
    config.vocos_dim = encoder.require("vocos_dim").as_i64();
    config.vocos_intermediate_dim = encoder.require("vocos_intermediate_dim").as_i64();
    config.vocos_num_layers = encoder.require("vocos_num_layers").as_i64();
    config.use_normed_whisper = json::optional_bool(value, "use_normed_whisper", config.use_normed_whisper);
    config.whisper_dim = value.require("whisper_dim").as_i64();
    config.chromagram_dim = value.require("chromagram_dim").as_i64();
    if (config.coco_type != expected_type) {
        throw std::runtime_error("Vevo2 Coco tokenizer type mismatch: expected " + expected_type);
    }
    if (config.downsample_rate <= 0 || config.codebook_size <= 0 || config.hidden_size <= 0 ||
        config.codebook_dim <= 0 || config.vocos_dim <= 0 || config.vocos_intermediate_dim <= 0 ||
        config.vocos_num_layers <= 0 || config.whisper_dim <= 0 || config.chromagram_dim <= 0) {
        throw std::runtime_error("Vevo2 Coco tokenizer config contains non-positive dimensions");
    }
    return config;
}

Vevo2AcousticPreprocessConfig parse_acoustic_preprocess_config(const engine::io::json::Value & value) {
    Vevo2AcousticPreprocessConfig config;
    config.sample_rate = value.require("sample_rate").as_i64();
    config.hop_size = value.require("hop_size").as_i64();
    config.n_fft = value.require("n_fft").as_i64();
    config.num_mels = value.require("num_mels").as_i64();
    config.win_size = value.require("win_size").as_i64();
    config.fmin = value.require("fmin").as_f32();
    config.fmax = value.require("fmax").as_f32();
    config.mel_var = value.require("mel_var").as_f32();
    config.mel_mean = value.require("mel_mean").as_f32();
    if (config.sample_rate <= 0 || config.hop_size <= 0 || config.n_fft <= 0 || config.num_mels <= 0 ||
        config.win_size <= 0) {
        throw std::runtime_error("Vevo2 acoustic preprocess config contains non-positive dimensions");
    }
    return config;
}

std::pair<Vevo2CocoTokenizerConfig, Vevo2CocoTokenizerConfig> parse_ar_coco_configs(
    const std::filesystem::path & ar_amphion_config_path) {
    const auto root = parse_jsonc_file(ar_amphion_config_path);
    const auto & model = root.require("model");
    return {
        parse_coco_config(model.require("coco_style"), "style"),
        parse_coco_config(model.require("coco_content_style"), "content_style"),
    };
}

Vevo2FMConfig parse_fm_config(const std::filesystem::path & fm_config_path, bool expect_text_condition) {
    const auto root = parse_jsonc_file(fm_config_path);
    Vevo2FMConfig config;
    if (root.require("model_type").as_string() != "FlowMatchingTransformer") {
        throw std::runtime_error("Vevo2 FM config model_type mismatch");
    }
    config.preprocess = parse_acoustic_preprocess_config(root.require("preprocess"));
    const auto & model = root.require("model");
    const auto & fmt = model.require("flow_matching_transformer");
    config.mel_dim = fmt.require("mel_dim").as_i64();
    config.hidden_size = fmt.require("hidden_size").as_i64();
    config.num_layers = fmt.require("num_layers").as_i64();
    config.num_heads = fmt.require("num_heads").as_i64();
    config.cfg_scale = fmt.require("cfg_scale").as_f32();
    config.use_cond_code = json::optional_bool(fmt, "use_cond_code", config.use_cond_code);
    config.cond_codebook_size = fmt.require("cond_codebook_size").as_i64();
    config.cond_scale_factor = fmt.require("cond_scale_factor").as_i64();
    config.sigma = fmt.require("sigma").as_f32();
    config.time_scheduler = fmt.require("time_scheduler").as_string();
    config.cond_sample_rate = model.require("cond_sample_rate").as_i64();
    config.coco_content_style = parse_coco_config(model.require("coco"), "content_style");
    config.use_text_as_condition = json::optional_bool(fmt, "use_text_as_condition", false);
    config.text_cond_codebook_size = json::optional_i64(fmt, "text_cond_codebook_size", 0);
    if (config.use_text_as_condition != expect_text_condition) {
        throw std::runtime_error("Vevo2 FM text-condition config mismatch");
    }
    if (config.mel_dim <= 0 || config.hidden_size <= 0 || config.num_layers <= 0 || config.num_heads <= 0 ||
        config.cond_codebook_size <= 0 || config.cond_scale_factor <= 0 || config.cond_sample_rate <= 0) {
        throw std::runtime_error("Vevo2 FM config contains non-positive dimensions");
    }
    if (config.time_scheduler != "cos") {
        throw std::runtime_error("Vevo2 FM currently expects cos time scheduler");
    }
    if (config.mel_dim != config.preprocess.num_mels) {
        throw std::runtime_error("Vevo2 FM mel_dim does not match preprocess num_mels");
    }
    return config;
}

Vevo2VocoderConfig parse_vocoder_config(const std::filesystem::path & vocoder_config_path) {
    const auto root = parse_jsonc_file(vocoder_config_path);
    Vevo2VocoderConfig config;
    if (root.require("model_type").as_string() != "Vocoder") {
        throw std::runtime_error("Vevo2 vocoder config model_type mismatch");
    }
    config.preprocess = parse_acoustic_preprocess_config(root.require("preprocess"));
    const auto & vocos = root.require("model").require("vocos");
    config.input_channels = vocos.require("input_channels").as_i64();
    config.dim = vocos.require("dim").as_i64();
    config.intermediate_dim = vocos.require("intermediate_dim").as_i64();
    config.num_layers = vocos.require("num_layers").as_i64();
    config.n_fft = vocos.require("n_fft").as_i64();
    config.hop_size = vocos.require("hop_size").as_i64();
    config.padding = vocos.require("padding").as_string();
    if (config.input_channels <= 0 || config.dim <= 0 || config.intermediate_dim <= 0 ||
        config.num_layers <= 0 || config.n_fft <= 0 || config.hop_size <= 0) {
        throw std::runtime_error("Vevo2 vocoder config contains non-positive dimensions");
    }
    if (config.input_channels != config.preprocess.num_mels) {
        throw std::runtime_error("Vevo2 vocoder input channels do not match preprocess num_mels");
    }
    return config;
}

runtime::ModelMetadata make_metadata() {
    runtime::ModelMetadata metadata;
    metadata.family = "vevo2";
    metadata.variant = "RMSnow/Vevo2";
    metadata.description = "Vevo2 singing voice conversion draft loaded from local extracted assets.";
    metadata.config_candidates = {
        "contentstyle_modeling/posttrained/config.json",
        "contentstyle_modeling/posttrained/amphion_config.json",
        "contentstyle_modeling/posttrained/generation_config.json",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/config.json",
        "vocoder/config.json",
        "../whisper-medium/config.json",
    };
    metadata.weight_candidates = {
        "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors",
        "tokenizer/prosody_fvq512_6.25hz/model.safetensors",
        "contentstyle_modeling/posttrained/model.safetensors",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa/model.safetensors",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.safetensors",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/model.safetensors",
        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/whisper_stats.safetensors",
        "vocoder/model.safetensors",
        "vocoder/model_1.safetensors",
        "vocoder/model_2.safetensors",
        "../whisper-medium/model.safetensors",
    };
    return metadata;
}

runtime::CapabilitySet make_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::SpeechToSpeech, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::Svc, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"en", "zh"};
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    return capabilities;
}

runtime::ModelCliInterface make_cli_interface() {
    runtime::ModelCliInterface cli;
    cli.request_options = {
        {
            "--task-route",
            "route",
            "VeVo2 route: zero_shot_tts, text_to_singing, svs, style_preserved_vc, "
            "style_preserved_svc, style_converted_vc, style_converted_svc, editing, "
            "singing_style_conversion, humming_to_singing, or instrument_to_singing",
        },
        {"--source-audio", "wav", "Source speech or singing audio for conversion/editing routes"},
        {"--target-voice", "wav", "Target timbre reference audio"},
        {"--prosody-ref", "wav", "Prosody or melody reference audio"},
        {"--style-ref", "wav", "Style reference audio"},
        {"--target-text", "text", "Text or lyrics to vocalize"},
        {"--style-ref-text", "text", "Transcript for the style reference when text conditioning is used"},
        {"--use-prosody-code", "true|false", "Enable explicit prosody-code conditioning"},
        {"--predict-target-prosody", "true|false", "Predict target prosody during AR generation"},
        {"--use-pitch-shift", "true|false", "Pitch-align source/prosody/style references to the target voice"},
        {"--source-shift-steps", "n", "Manual pitch-shift semitone steps for source audio"},
        {"--prosody-shift-steps", "n", "Manual pitch-shift semitone steps for prosody reference audio"},
        {"--style-shift-steps", "n", "Manual pitch-shift semitone steps for style reference audio"},
        {"--target-duration-seconds", "seconds", "Target output duration hint for flow matching"},
        {"--reference-duration-seconds", "seconds", "Trim target voice reference duration before conditioning"},
        {"--temperature", "value", "AR sampling temperature"},
        {"--top-k", "n", "AR top-k sampling limit"},
        {"--top-p", "value", "AR nucleus sampling threshold"},
        {"--repetition-penalty", "value", "AR repetition penalty"},
        {"--max-tokens", "n", "Maximum generated AR content/style tokens"},
        {"--num-inference-steps", "n", "Flow-matching denoising steps"},
        {"--seed", "n", "Request seed; omitted means random"},
    };
    cli.session_options = {
        {"--session-option vevo2.weight_type", "type", "Weight storage type for VeVo2 GGML graphs"},
        {
            "--session-option vevo2.whisper_weight_type",
            "type",
            "Weight storage type for the reusable Whisper feature extractor",
        },
        {
            "--session-option vevo2.fm_weight_type",
            "type",
            "Weight storage type for VeVo2 flow-matching graphs",
        },
        {
            "--session-option vevo2.vocoder_weight_type",
            "type",
            "Weight storage type for VeVo2 vocoder graphs",
        },
    };
    cli.load_options = {
        {
            "--load-option vevo2.whisper_model_path",
            "dir",
            "Local Whisper model directory used by VeVo2 feature extraction",
        },
    };
    return cli;
}

std::vector<runtime::NamedAsset> discovered_configs(const Vevo2AssetPaths & paths) {
    return {
        {"ar_config", paths.ar_config},
        {"ar_amphion_config", paths.ar_amphion_config},
        {"ar_generation_config", paths.ar_generation_config},
        {"ar_tokenizer_config", paths.ar_tokenizer_config},
        {"ar_tokenizer_json", paths.ar_tokenizer_json},
        {"fm_config", paths.fm_config},
        {"fm_text_config", paths.fm_text_config},
        {"vocoder_config", paths.vocoder_config},
        {"whisper_config", paths.whisper_config},
    };
}

std::vector<runtime::NamedAsset> discovered_weights(const Vevo2AssetPaths & paths) {
    return {
        {"content_style_tokenizer", paths.content_style_tokenizer_weights},
        {"prosody_tokenizer", paths.prosody_tokenizer_weights},
        {"ar", paths.ar_weights},
        {"fm", paths.fm_weights},
        {"fm_whisper_stats", paths.fm_whisper_stats},
        {"fm_text", paths.fm_text_weights},
        {"fm_text_whisper_stats", paths.fm_text_whisper_stats},
        {"vocoder_0", paths.vocoder_weights_0},
        {"vocoder_1", paths.vocoder_weights_1},
        {"vocoder_2", paths.vocoder_weights_2},
        {"whisper", paths.whisper_weights},
    };
}

}  // namespace

std::filesystem::path resolve_vevo2_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Vevo2 model path does not exist: " + model_path.string());
}

std::filesystem::path resolve_vevo2_whisper_root(
    const runtime::ModelLoadRequest & request,
    const std::filesystem::path & model_root) {
    const auto explicit_vevo2 = option_or_empty(request, "vevo2.whisper_model_path");
    if (!explicit_vevo2.empty()) {
        return require_existing_directory(std::filesystem::path(explicit_vevo2), "Whisper model root");
    }
    const auto explicit_common = option_or_empty(request, "whisper_model_path");
    if (!explicit_common.empty()) {
        return require_existing_directory(std::filesystem::path(explicit_common), "Whisper model root");
    }
    return require_existing_directory(model_root.parent_path() / "whisper-medium", "Whisper model root");
}

Vevo2AssetPaths resolve_vevo2_asset_paths(const runtime::ModelLoadRequest & request) {
    Vevo2AssetPaths paths;
    paths.model_root = resolve_vevo2_model_root(request.model_path);
    paths.whisper_root = resolve_vevo2_whisper_root(request, paths.model_root);

    paths.content_style_tokenizer_weights = require_existing_file(
        paths.model_root / "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors",
        "content-style tokenizer weights");
    paths.prosody_tokenizer_weights = require_existing_file(
        paths.model_root / "tokenizer/prosody_fvq512_6.25hz/model.safetensors",
        "prosody tokenizer weights");

    const auto ar_root = paths.model_root / "contentstyle_modeling/posttrained";
    paths.ar_config = require_existing_file(ar_root / "config.json", "AR config");
    paths.ar_amphion_config = require_existing_file(ar_root / "amphion_config.json", "AR Amphion config");
    paths.ar_generation_config = require_existing_file(ar_root / "generation_config.json", "AR generation config");
    paths.ar_weights = require_existing_file(ar_root / "model.safetensors", "AR weights");
    paths.ar_tokenizer_config = require_existing_file(ar_root / "tokenizer_config.json", "AR tokenizer config");
    paths.ar_tokenizer_json = require_existing_file(ar_root / "tokenizer.json", "AR tokenizer JSON");
    paths.ar_vocab = require_existing_file(ar_root / "vocab.json", "AR tokenizer vocab");
    paths.ar_merges = require_existing_file(ar_root / "merges.txt", "AR tokenizer merges");
    paths.ar_added_tokens = require_existing_file(ar_root / "added_tokens.json", "AR added tokens");
    paths.ar_special_tokens = require_existing_file(ar_root / "special_tokens_map.json", "AR special tokens");

    const auto fm_root = paths.model_root / "acoustic_modeling/fm_emilia101k_singnet7k_repa";
    paths.fm_config = require_existing_file(fm_root / "config.json", "FM config");
    paths.fm_weights = require_existing_file(fm_root / "model.safetensors", "FM weights");
    paths.fm_whisper_stats = require_existing_file(fm_root / "whisper_stats.safetensors", "FM Whisper stats");

    const auto fm_text_root = paths.model_root / "acoustic_modeling/fm_emilia101k_singnet7k_repa_text";
    paths.fm_text_config = require_existing_file(fm_text_root / "config.json", "FM text config");
    paths.fm_text_weights = require_existing_file(fm_text_root / "model.safetensors", "FM text weights");
    paths.fm_text_whisper_stats = require_existing_file(fm_text_root / "whisper_stats.safetensors", "FM text Whisper stats");

    const auto vocoder_root = paths.model_root / "vocoder";
    paths.vocoder_config = require_existing_file(vocoder_root / "config.json", "vocoder config");
    paths.vocoder_weights_0 = require_existing_file(vocoder_root / "model.safetensors", "vocoder weights 0");
    paths.vocoder_weights_1 = require_existing_file(vocoder_root / "model_1.safetensors", "vocoder weights 1");
    paths.vocoder_weights_2 = require_existing_file(vocoder_root / "model_2.safetensors", "vocoder weights 2");

    paths.whisper_config = require_existing_file(paths.whisper_root / "config.json", "Whisper config");
    paths.whisper_weights = require_existing_file(paths.whisper_root / "model.safetensors", "Whisper weights");
    return paths;
}

std::shared_ptr<const Vevo2Assets> load_vevo2_assets(const runtime::ModelLoadRequest & request) {
    Vevo2Assets assets;
    assets.paths = resolve_vevo2_asset_paths(request);
    assets.ar_config = parse_ar_config(assets.paths.ar_config, assets.paths.ar_generation_config);
    const auto coco_configs = parse_ar_coco_configs(assets.paths.ar_amphion_config);
    assets.prosody_tokenizer_config = coco_configs.first;
    assets.content_style_tokenizer_config = coco_configs.second;
    assets.fm_config = parse_fm_config(assets.paths.fm_config, false);
    assets.fm_text_config = parse_fm_config(assets.paths.fm_text_config, true);
    assets.vocoder_config = parse_vocoder_config(assets.paths.vocoder_config);
    assets.whisper_config = parse_whisper_config(assets.paths.whisper_config);
    assets.metadata = make_metadata();
    assets.capabilities = make_capabilities();
    assets.discovered_configs = discovered_configs(assets.paths);
    assets.discovered_weights = discovered_weights(assets.paths);
    return std::make_shared<Vevo2Assets>(std::move(assets));
}

runtime::ModelInspection inspect_vevo2_model(const runtime::ModelLoadRequest & request) {
    const auto assets = load_vevo2_assets(request);
    runtime::ModelInspection inspection;
    inspection.metadata = assets->metadata;
    inspection.capabilities = assets->capabilities;
    inspection.model_root = assets->paths.model_root;
    inspection.discovered_configs = assets->discovered_configs;
    inspection.discovered_weights = assets->discovered_weights;
    inspection.cli = make_cli_interface();
    return inspection;
}

}  // namespace engine::models::vevo2

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::seed_vc {
namespace {

std::filesystem::path require_file(const std::filesystem::path & path, const char * role) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("Seed-VC missing ") + role + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path require_directory(const std::filesystem::path & path, const char * role) {
    if (!engine::io::is_existing_directory(path)) {
        throw std::runtime_error(std::string("Seed-VC missing ") + role + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

std::filesystem::path optional_file(const std::filesystem::path & path) {
    return engine::io::is_existing_file(path) ? std::filesystem::weakly_canonical(path) : std::filesystem::path{};
}

std::filesystem::path optional_directory(const std::filesystem::path & path) {
    return engine::io::is_existing_directory(path) ? std::filesystem::weakly_canonical(path) : std::filesystem::path{};
}

runtime::ModelMetadata make_metadata() {
    runtime::ModelMetadata metadata;
    metadata.family = "seed_vc";
    metadata.variant = "v2-vc-v1-svc";
    metadata.description = "SeedVC-MLX asset bundle for V2 voice conversion and V1 voice conversion/SVC variants.";
    metadata.config_candidates = {
        "seed_vc_manifest.json",
        "v2/vc_wrapper.json",
        "astral/bsq32.json",
        "astral/bsq2048.json",
        "v1/svc.json",
        "v1/whisper_bigvgan.json",
        "v1/xlsr_hift.json",
        "hift/config.json",
        "bigvgan/v2_22khz_80band_256x/config.json",
        "bigvgan/v2_44khz_128band_512x/config.json",
        "whisper-small/config.json",
        "hubert-large-ll60k/config.json",
    };
    metadata.weight_candidates = {
        "v2/ar.safetensors",
        "v2/cfm.safetensors",
        "v1/svc.safetensors",
        "v1/whisper_bigvgan.safetensors",
        "v1/xlsr_hift.safetensors",
        "astral/bsq32.safetensors",
        "astral/bsq2048.safetensors",
        "campplus/model.safetensors",
        "rmvpe/model.safetensors",
        "hift/model.safetensors",
        "bigvgan/v2_22khz_80band_256x/model.safetensors",
        "bigvgan/v2_44khz_128band_512x/model.safetensors",
        "whisper-small/model.safetensors",
        "hubert-large-ll60k/model.safetensors",
        "wav2vec2-xls-r-300m/model.safetensors",
    };
    return metadata;
}

runtime::CapabilitySet make_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::Svc, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = true;
    return capabilities;
}

SeedVcLengthRegulatorConfig parse_length_regulator(
    const engine::io::json::Value & value,
    const char * codebook_size_key) {
    SeedVcLengthRegulatorConfig config;
    config.channels = value.require("channels").as_i64();
    config.codebook_size = value.require(codebook_size_key).as_i64();
    config.is_discrete = value.require("is_discrete").as_bool();
    config.f0_condition = value.require("f0_condition").as_bool();
    if (const auto * sampling_ratios = value.find("sampling_ratios")) {
        config.sampling_ratios = engine::io::json::number_array_as<int64_t>(*sampling_ratios);
    }
    return config;
}

SeedVcAstralConfig parse_astral_config(
    const engine::io::json::Value & root,
    bool skip_ssl_default) {
    const auto & encoder = root.require("encoder");
    const auto & quantizer = root.require("quantizer");
    SeedVcAstralConfig config;
    config.ssl_output_layer = root.require("ssl_output_layer").as_i64();
    config.encoder_dim = encoder.require("dim").as_i64();
    config.encoder_blocks = encoder.require("num_blocks").as_i64();
    config.encoder_intermediate_dim = encoder.require("intermediate_dim").as_i64();
    config.encoder_dilation = encoder.require("dilation").as_i64();
    config.encoder_input_dim = encoder.require("input_dim").as_i64();
    config.quantizer_codebook_size = quantizer.require("codebook_size").as_i64();
    config.quantizer_dim = quantizer.require("dim").as_i64();
    config.skip_ssl = engine::io::json::optional_bool(root, "skip_ssl", skip_ssl_default);
    return config;
}

SeedVcBigVganConfig parse_bigvgan_config(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    SeedVcBigVganConfig config;
    config.sampling_rate = root.require("sampling_rate").as_i64();
    config.num_mels = root.require("num_mels").as_i64();
    config.n_fft = root.require("n_fft").as_i64();
    config.hop_size = root.require("hop_size").as_i64();
    config.win_size = root.require("win_size").as_i64();
    config.upsample_initial_channel = root.require("upsample_initial_channel").as_i64();
    config.snake_logscale = root.require("snake_logscale").as_bool();
    config.upsample_rates = engine::io::json::number_array_as<int64_t>(root.require("upsample_rates"));
    config.upsample_kernel_sizes =
        engine::io::json::number_array_as<int64_t>(root.require("upsample_kernel_sizes"));
    config.resblock_kernel_sizes =
        engine::io::json::number_array_as<int64_t>(root.require("resblock_kernel_sizes"));
    return config;
}

std::vector<std::vector<int64_t>> parse_int_matrix(const engine::io::json::Value & value) {
    std::vector<std::vector<int64_t>> out;
    const auto & rows = value.as_array();
    out.reserve(rows.size());
    for (const auto & row : rows) {
        out.push_back(engine::io::json::number_array_as<int64_t>(row));
    }
    return out;
}

SeedVcHiftConfig parse_hift_config(const std::filesystem::path & path) {
    SeedVcHiftConfig config;
    const auto root = engine::io::json::parse_file(path);
    const auto & hift = root.require("hift");
    config.in_channels = hift.require("in_channels").as_i64();
    config.base_channels = hift.require("base_channels").as_i64();
    config.nb_harmonics = hift.require("nb_harmonics").as_i64();
    config.sampling_rate = hift.require("sampling_rate").as_i64();
    config.nsf_alpha = hift.require("nsf_alpha").as_f32();
    config.nsf_sigma = hift.require("nsf_sigma").as_f32();
    config.nsf_voiced_threshold = hift.require("nsf_voiced_threshold").as_f32();
    config.upsample_rates = engine::io::json::number_array_as<int64_t>(hift.require("upsample_rates"));
    config.upsample_kernel_sizes =
        engine::io::json::number_array_as<int64_t>(hift.require("upsample_kernel_sizes"));
    const auto & istft = hift.require("istft_params");
    config.istft_n_fft = istft.require("n_fft").as_i64();
    config.istft_hop = istft.require("hop_len").as_i64();
    config.resblock_kernel_sizes =
        engine::io::json::number_array_as<int64_t>(hift.require("resblock_kernel_sizes"));
    config.resblock_dilation_sizes = parse_int_matrix(hift.require("resblock_dilation_sizes"));
    config.source_resblock_kernel_sizes =
        engine::io::json::number_array_as<int64_t>(hift.require("source_resblock_kernel_sizes"));
    config.source_resblock_dilation_sizes = parse_int_matrix(hift.require("source_resblock_dilation_sizes"));
    config.lrelu_slope = hift.require("lrelu_slope").as_f32();
    config.audio_limit = hift.require("audio_limit").as_f32();

    const auto & f0_predictor = root.require("f0_predictor");
    config.f0_num_class = f0_predictor.require("num_class").as_i64();
    config.f0_in_channels = f0_predictor.require("in_channels").as_i64();
    config.f0_cond_channels = f0_predictor.require("cond_channels").as_i64();
    return config;
}

SeedVcConfig load_config(const SeedVcAssetPaths & paths) {
    SeedVcConfig config;
    const auto v2 = engine::io::json::parse_file(paths.v2_wrapper_json);
    const auto & v2_mel = v2.require("mel_fn");
    config.v2_mel.sample_rate = v2_mel.require("sampling_rate").as_i64();
    config.v2_mel.n_fft = v2_mel.require("n_fft").as_i64();
    config.v2_mel.win_size = v2_mel.require("win_size").as_i64();
    config.v2_mel.hop_size = v2_mel.require("hop_size").as_i64();
    config.v2_mel.num_mels = v2_mel.require("num_mels").as_i64();
    config.v2_mel.fmin = v2_mel.require("fmin").as_f32();
    if (const auto * fmax = v2_mel.find("fmax"); fmax != nullptr && !fmax->is_null()) {
        config.v2_mel.fmax = fmax->as_f32();
    }

    const auto & v2_cfm = v2.require("cfm").require("estimator");
    config.v2_cfm.block_size = v2_cfm.require("block_size").as_i64();
    config.v2_cfm.depth = v2_cfm.require("depth").as_i64();
    config.v2_cfm.num_heads = v2_cfm.require("num_heads").as_i64();
    config.v2_cfm.hidden_dim = v2_cfm.require("hidden_dim").as_i64();
    config.v2_cfm.in_channels = v2_cfm.require("in_channels").as_i64();
    config.v2_cfm.content_dim = v2_cfm.require("content_dim").as_i64();
    config.v2_cfm.style_encoder_dim = v2_cfm.require("style_encoder_dim").as_i64();
    config.v2_cfm.time_as_token = v2_cfm.require("time_as_token").as_bool();
    config.v2_cfm.style_as_token = v2_cfm.require("style_as_token").as_bool();
    config.v2_cfm.uvit_skip_connection = v2_cfm.require("uvit_skip_connection").as_bool();
    config.v2_cfm_length_regulator = parse_length_regulator(v2.require("cfm_length_regulator"), "codebook_size");

    const auto & v2_ar = v2.require("ar").require("model").require("config");
    config.v2_ar.dim = v2_ar.require("dim").as_i64();
    config.v2_ar.head_dim = v2_ar.require("head_dim").as_i64();
    config.v2_ar.n_local_heads = v2_ar.require("n_local_heads").as_i64();
    config.v2_ar.intermediate_size = v2_ar.require("intermediate_size").as_i64();
    config.v2_ar.n_head = v2_ar.require("n_head").as_i64();
    config.v2_ar.n_layer = v2_ar.require("n_layer").as_i64();
    config.v2_ar.vocab_size = v2_ar.require("vocab_size").as_i64();
    config.v2_ar.rope_base = v2_ar.require("rope_base").as_f32();
    config.v2_ar_length_regulator = parse_length_regulator(v2.require("ar_length_regulator"), "codebook_size");

    const auto & style_encoder = v2.require("style_encoder");
    config.v2_style_feat_dim = style_encoder.require("feat_dim").as_i64();
    config.v2_style_embedding_size = style_encoder.require("embedding_size").as_i64();
    config.v2_astral_narrow = parse_astral_config(engine::io::json::parse_file(paths.astral_bsq32_json), true);
    config.v2_astral_wide = parse_astral_config(engine::io::json::parse_file(paths.astral_bsq2048_json), false);

    auto parse_v1_variant = [](const std::filesystem::path & path,
                               SeedVcMelConfig & mel,
                               SeedVcV1DitConfig & dit,
                               SeedVcLengthRegulatorConfig & length_regulator,
                               SeedVcV1WavenetConfig & wavenet,
                               int64_t & style_dim) {
        const auto v1 = engine::io::json::parse_file(path);
        const auto & preprocess = v1.require("preprocess_params");
        const auto & spect = preprocess.require("spect_params");
        mel.sample_rate = preprocess.require("sr").as_i64();
        mel.n_fft = spect.require("n_fft").as_i64();
        mel.win_size = spect.require("win_length").as_i64();
        mel.hop_size = spect.require("hop_length").as_i64();
        mel.num_mels = spect.require("n_mels").as_i64();
        mel.fmin = spect.require("fmin").as_f32();
        if (const auto * fmax = spect.find("fmax"); fmax != nullptr && !fmax->is_null()) {
            if (fmax->is_number()) {
                mel.fmax = fmax->as_f32();
            } else if (fmax->as_string() != "None") {
                throw std::runtime_error("Seed-VC V1 fmax config must be numeric or None");
            }
        }

        const auto & model_params = v1.require("model_params");
        style_dim = model_params.require("style_encoder").require("dim").as_i64();
        length_regulator =
            parse_length_regulator(model_params.require("length_regulator"), "content_codebook_size");
        const auto & v1_dit = model_params.require("DiT");
        dit.hidden_dim = v1_dit.require("hidden_dim").as_i64();
        dit.num_heads = v1_dit.require("num_heads").as_i64();
        dit.depth = v1_dit.require("depth").as_i64();
        dit.block_size = v1_dit.require("block_size").as_i64();
        dit.in_channels = v1_dit.require("in_channels").as_i64();
        dit.content_dim = v1_dit.require("content_dim").as_i64();
        dit.content_codebook_size = v1_dit.require("content_codebook_size").as_i64();
        dit.n_f0_bins = v1_dit.require("n_f0_bins").as_i64();
        dit.style_condition = v1_dit.require("style_condition").as_bool();
        dit.f0_condition = v1_dit.require("f0_condition").as_bool();
        dit.time_as_token = v1_dit.require("time_as_token").as_bool();
        dit.style_as_token = v1_dit.require("style_as_token").as_bool();
        dit.uvit_skip_connection = v1_dit.require("uvit_skip_connection").as_bool();
        dit.long_skip_connection = v1_dit.require("long_skip_connection").as_bool();
        dit.final_layer_type = v1_dit.require("final_layer_type").as_string();
        if (const auto * wavenet_json = model_params.find("wavenet")) {
            wavenet.hidden_dim = wavenet_json->require("hidden_dim").as_i64();
            wavenet.num_layers = wavenet_json->require("num_layers").as_i64();
            wavenet.kernel_size = wavenet_json->require("kernel_size").as_i64();
            wavenet.dilation_rate = wavenet_json->require("dilation_rate").as_i64();
            wavenet.style_condition = wavenet_json->require("style_condition").as_bool();
        }
    };

    parse_v1_variant(
        paths.v1_svc_json,
        config.v1_mel,
        config.v1_dit,
        config.v1_length_regulator,
        config.v1_wavenet,
        config.v1_style_dim);
    if (!paths.v1_whisper_bigvgan_json.empty()) {
        parse_v1_variant(
            paths.v1_whisper_bigvgan_json,
            config.v1_whisper_bigvgan_mel,
            config.v1_whisper_bigvgan_dit,
            config.v1_whisper_bigvgan_length_regulator,
            config.v1_whisper_bigvgan_wavenet,
            config.v1_whisper_bigvgan_style_dim);
    }
    if (!paths.v1_xlsr_hift_json.empty()) {
        parse_v1_variant(
            paths.v1_xlsr_hift_json,
            config.v1_xlsr_hift_mel,
            config.v1_xlsr_hift_dit,
            config.v1_xlsr_hift_length_regulator,
            config.v1_xlsr_hift_wavenet,
            config.v1_xlsr_hift_style_dim);
    }

    config.bigvgan_22k = parse_bigvgan_config(paths.bigvgan_22k_config);
    config.bigvgan_44k = parse_bigvgan_config(paths.bigvgan_44k_config);
    if (!paths.hift_json.empty()) {
        config.hift = parse_hift_config(paths.hift_json);
    }
    return config;
}

}  // namespace

std::filesystem::path resolve_seed_vc_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("Seed-VC model path does not exist: " + model_path.string());
}

SeedVcAssetPaths resolve_seed_vc_asset_paths(const std::filesystem::path & model_path) {
    const auto root = resolve_seed_vc_model_root(model_path);
    SeedVcAssetPaths paths;
    paths.model_root = root;
    paths.v2_ar_weights = require_file(root / "v2/ar.safetensors", "V2 AR weights");
    paths.v2_cfm_weights = require_file(root / "v2/cfm.safetensors", "V2 CFM weights");
    paths.v1_svc_weights = require_file(root / "v1/svc.safetensors", "V1 SVC weights");
    paths.v1_whisper_bigvgan_weights = optional_file(root / "v1/whisper_bigvgan.safetensors");
    paths.v1_xlsr_hift_weights = optional_file(root / "v1/xlsr_hift.safetensors");
    paths.astral_bsq32_weights = require_file(root / "astral/bsq32.safetensors", "ASTRAL BSQ32 weights");
    paths.astral_bsq2048_weights = require_file(root / "astral/bsq2048.safetensors", "ASTRAL BSQ2048 weights");
    paths.campplus_weights = require_file(root / "campplus/model.safetensors", "CAMPPlus weights");
    paths.rmvpe_weights = require_file(root / "rmvpe/model.safetensors", "RMVPE weights");
    paths.hift_weights = optional_file(root / "hift/model.safetensors");
    paths.bigvgan_22k_weights = require_file(root / "bigvgan/v2_22khz_80band_256x/model.safetensors", "BigVGAN 22 kHz weights");
    paths.bigvgan_44k_weights = require_file(root / "bigvgan/v2_44khz_128band_512x/model.safetensors", "BigVGAN 44 kHz weights");
    paths.manifest = require_file(root / "seed_vc_manifest.json", "manifest");
    paths.v1_svc_json = require_file(root / "v1/svc.json", "V1 SVC config");
    paths.v1_whisper_bigvgan_json = optional_file(root / "v1/whisper_bigvgan.json");
    paths.v1_xlsr_hift_json = optional_file(root / "v1/xlsr_hift.json");
    paths.hift_json = optional_file(root / "hift/config.json");
    paths.bigvgan_22k_config = require_file(root / "bigvgan/v2_22khz_80band_256x/config.json", "BigVGAN 22 kHz config");
    paths.bigvgan_44k_config = require_file(root / "bigvgan/v2_44khz_128band_512x/config.json", "BigVGAN 44 kHz config");
    paths.v2_wrapper_json = require_file(root / "v2/vc_wrapper.json", "V2 wrapper config");
    paths.astral_bsq32_json = require_file(root / "astral/bsq32.json", "ASTRAL BSQ32 config");
    paths.astral_bsq2048_json = require_file(root / "astral/bsq2048.json", "ASTRAL BSQ2048 config");
    paths.whisper_small_root = require_directory(root / "whisper-small", "Whisper-small directory");
    paths.hubert_large_root = require_directory(root / "hubert-large-ll60k", "HuBERT-large directory");
    paths.wav2vec2_xlsr_root = optional_directory(root / "wav2vec2-xls-r-300m");
    paths.whisper_small_config = require_file(paths.whisper_small_root / "config.json", "Whisper-small config");
    paths.whisper_small_weights = require_file(paths.whisper_small_root / "model.safetensors", "Whisper-small weights");
    paths.hubert_large_config = require_file(paths.hubert_large_root / "config.json", "HuBERT-large config");
    paths.hubert_large_weights = require_file(paths.hubert_large_root / "model.safetensors", "HuBERT-large weights");
    if (!paths.wav2vec2_xlsr_root.empty()) {
        paths.wav2vec2_xlsr_config = require_file(paths.wav2vec2_xlsr_root / "config.json", "Wav2Vec2 XLS-R config");
        paths.wav2vec2_xlsr_weights =
            require_file(paths.wav2vec2_xlsr_root / "model.safetensors", "Wav2Vec2 XLS-R weights");
    }
    return paths;
}

std::shared_ptr<const SeedVcAssets> load_seed_vc_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SeedVcAssets>();
    assets->paths = resolve_seed_vc_asset_paths(model_path);
    assets->config = load_config(assets->paths);
    assets->metadata = make_metadata();
    assets->capabilities = make_capabilities();
    assets->discovered_configs = runtime::discover_named_assets(
        assets->paths.model_root,
        assets->metadata.config_candidates);
    assets->discovered_weights = runtime::discover_named_assets(
        assets->paths.model_root,
        assets->metadata.weight_candidates);
    return assets;
}

runtime::ModelInspection inspect_seed_vc_model(const std::filesystem::path & model_path) {
    const auto assets = load_seed_vc_assets(model_path);
    runtime::ModelInspection inspection;
    inspection.metadata = assets->metadata;
    inspection.capabilities = assets->capabilities;
    inspection.model_root = assets->paths.model_root;
    inspection.discovered_configs = assets->discovered_configs;
    inspection.discovered_weights = assets->discovered_weights;
    return inspection;
}

}  // namespace engine::models::seed_vc

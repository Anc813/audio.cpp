#include "engine/models/miocodec/assets.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/yaml.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::models::miocodec {
namespace {

std::filesystem::path resolve_model_root(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_directory(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path.parent_path());
    }
    throw std::runtime_error("MioCodec model path does not exist: " + model_path.string());
}

std::vector<int> require_int_list(
    const engine::io::yaml::FlattenedDocument & document,
    const std::string & key) {
    const auto it = document.lists.find(key);
    if (it == document.lists.end() || it->second.empty()) {
        throw std::runtime_error("Missing yaml list key: " + key);
    }
    std::vector<int> values;
    values.reserve(it->second.size());
    for (const auto & value : it->second) {
        values.push_back(std::stoi(value));
    }
    return values;
}

bool require_bool(
    const engine::io::yaml::FlattenedDocument & document,
    const std::string & key) {
    const auto it = document.scalars.find(key);
    if (it == document.scalars.end()) {
        throw std::runtime_error("Missing yaml scalar key: " + key);
    }
    return engine::io::yaml::parse_bool_scalar(it->second, key);
}

int product(const std::vector<int> & values) {
    if (values.empty()) {
        throw std::runtime_error("MioCodec quantizer levels must not be empty");
    }
    return std::accumulate(values.begin(), values.end(), 1, [](int lhs, int rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("MioCodec quantizer levels must be positive");
        }
        return lhs * rhs;
    });
}

MioCodecConfig parse_config(const MioCodecAssetPaths & paths) {
    const auto parsed = engine::io::yaml::parse_flattened_document(engine::io::read_text_file(paths.config_path));
    MioCodecConfig config;
    config.local_ssl_layers = require_int_list(parsed, "model.init_args.config.local_ssl_layers");
    config.global_ssl_layers = require_int_list(parsed, "model.init_args.config.global_ssl_layers");
    config.normalize_ssl_features = require_bool(parsed, "model.init_args.config.normalize_ssl_features");
    config.downsample_factor = engine::io::yaml::require_int(parsed, "model.init_args.config.downsample_factor");
    config.use_conv_downsample = require_bool(parsed, "model.init_args.config.use_conv_downsample");
    config.sample_rate = engine::io::yaml::require_int(parsed, "model.init_args.config.sample_rate");
    config.n_fft = engine::io::yaml::require_int(parsed, "model.init_args.config.n_fft");
    config.hop_length = engine::io::yaml::require_int(parsed, "model.init_args.config.hop_length");
    config.use_wave_decoder = require_bool(parsed, "model.init_args.config.use_wave_decoder");
    config.wave_upsample_factor = engine::io::yaml::require_int(parsed, "model.init_args.config.wave_upsample_factor");
    config.wave_decoder_dim = engine::io::yaml::require_int(parsed, "model.init_args.config.wave_decoder_dim");
    config.wave_resnet_num_blocks = engine::io::yaml::require_int(parsed, "model.init_args.config.wave_resnet_num_blocks");
    config.wave_resnet_num_groups = engine::io::yaml::require_int(parsed, "model.init_args.config.wave_resnet_num_groups");
    config.wave_upsampler_factors = require_int_list(parsed, "model.init_args.config.wave_upsampler_factors");
    config.wave_upsampler_kernel_sizes = require_int_list(parsed, "model.init_args.config.wave_upsampler_kernel_sizes");

    config.local_encoder_dim = engine::io::yaml::require_int(parsed, "model.init_args.local_encoder.init_args.dim");
    config.local_encoder_layers = engine::io::yaml::require_int(parsed, "model.init_args.local_encoder.init_args.n_layers");
    config.local_encoder_heads = engine::io::yaml::require_int(parsed, "model.init_args.local_encoder.init_args.n_heads");
    config.local_encoder_window_size = engine::io::yaml::require_int(parsed, "model.init_args.local_encoder.init_args.window_size");
    config.local_encoder_max_seq_len = engine::io::yaml::require_int(parsed, "model.init_args.local_encoder.init_args.max_seq_len");
    config.quantizer_input_dim = engine::io::yaml::require_int(parsed, "model.init_args.local_quantizer.init_args.input_dim");
    config.quantizer_output_dim = engine::io::yaml::require_int(parsed, "model.init_args.local_quantizer.init_args.output_dim");
    config.quantizer_levels = require_int_list(parsed, "model.init_args.local_quantizer.init_args.levels");
    config.codebook_size = product(config.quantizer_levels);

    config.global_encoder_input_channels = engine::io::yaml::require_int(parsed, "model.init_args.global_encoder.init_args.input_channels");
    config.global_encoder_output_channels = engine::io::yaml::require_int(parsed, "model.init_args.global_encoder.init_args.output_channels");
    config.global_encoder_layers = engine::io::yaml::require_int(parsed, "model.init_args.global_encoder.init_args.num_layers");
    config.global_encoder_dim = engine::io::yaml::require_int(parsed, "model.init_args.global_encoder.init_args.dim");
    config.global_encoder_intermediate_dim = engine::io::yaml::require_int(parsed, "model.init_args.global_encoder.init_args.intermediate_dim");

    config.wave_prenet_dim = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.dim");
    config.wave_prenet_output_dim = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.output_dim");
    config.wave_prenet_layers = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.n_layers");
    config.wave_prenet_heads = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.n_heads");
    config.wave_prenet_window_size = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.window_size");
    config.wave_prenet_max_seq_len = engine::io::yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.max_seq_len");

    config.wave_decoder_layers = engine::io::yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.n_layers");
    config.wave_decoder_heads = engine::io::yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.n_heads");
    config.wave_decoder_window_size = engine::io::yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.window_size");
    config.wave_decoder_max_seq_len = engine::io::yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.max_seq_len");
    config.wave_decoder_condition_dim = engine::io::yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.adanorm_condition_dim");
    return config;
}

}  // namespace

MioCodecAssetPaths resolve_miocodec_assets(const std::filesystem::path & model_path) {
    const auto root = resolve_model_root(model_path);
    MioCodecAssetPaths paths;
    paths.model_root = root;
    paths.config_path = engine::io::require_file(root / "config.yaml", "MioCodec config");
    paths.model_weights_path = engine::io::require_file(root / "model.safetensors", "MioCodec model weights");
    paths.wavlm_weights_path = engine::io::require_file(
        root / "wavlm-base-plus-mlx" / "weights.safetensors",
        "MioCodec WavLM weights");
    return paths;
}

std::shared_ptr<const MioCodecAssets> load_miocodec_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MioCodecAssets>();
    assets->paths = resolve_miocodec_assets(model_path);
    assets->config = parse_config(assets->paths);
    return assets;
}

}  // namespace engine::models::miocodec

#include "engine/models/miocodec/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/miocodec/session.h"

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

bool has_miocodec_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.yaml")
        && engine::io::is_existing_file(root / "model.safetensors")
        && engine::io::is_existing_file(root / "wavlm-base-plus-mlx" / "weights.safetensors");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(resolve_model_root(request.model_path), {"config.yaml"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    return runtime::discover_named_assets(
        resolve_model_root(request.model_path),
        {"model.safetensors", "wavlm-base-plus-mlx/weights.safetensors"});
}

runtime::CapabilitySet make_capabilities() {
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::VoiceConversion, {runtime::RunMode::Offline}},
        {runtime::VoiceTaskKind::SpeechToSpeech, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.supports_style_condition = false;
    capabilities.supports_timestamps = false;
    return capabilities;
}

runtime::ModelMetadata make_metadata(const MioCodecAssets & assets) {
    runtime::ModelMetadata metadata;
    metadata.family = "miocodec";
    metadata.variant = std::to_string(assets.config.sample_rate) + "hz-" + std::to_string(assets.config.codebook_size);
    metadata.description = "MioCodec loaded from local extracted assets.";
    metadata.config_candidates = {"config.yaml"};
    metadata.weight_candidates = {"model.safetensors", "wavlm-base-plus-mlx/weights.safetensors"};
    return metadata;
}

class MioCodecLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "miocodec";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            return has_miocodec_assets(resolve_model_root(request.model_path));
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_miocodec_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata = make_metadata(*assets);
        inspection.capabilities = make_capabilities();
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_miocodec_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

MioCodecLoadedModel::MioCodecLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MioCodecAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MioCodecLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MioCodecLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MioCodecLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioCodec only supports offline sessions");
    }
    if (task.task != runtime::VoiceTaskKind::VoiceConversion
        && task.task != runtime::VoiceTaskKind::SpeechToSpeech) {
        throw std::runtime_error("MioCodec supports VoiceConversion or SpeechToSpeech tasks");
    }
    return std::make_unique<MioCodecSession>(task, options, assets_);
}

std::unique_ptr<MioCodecLoadedModel> load_miocodec_model(const std::filesystem::path & model_path) {
    auto assets = load_miocodec_assets(model_path);
    return std::make_unique<MioCodecLoadedModel>(make_metadata(*assets), make_capabilities(), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_miocodec_loader() {
    return std::make_shared<MioCodecLoader>();
}

}  // namespace engine::models::miocodec

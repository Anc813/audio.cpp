#include "engine/models/miotts/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/miotts/session.h"

#include <stdexcept>
#include <utility>

namespace engine::models::miotts {
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

bool has_miotts_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "config.json")
        && engine::io::is_existing_file(root / "generation_config.json")
        && engine::io::is_existing_file(root / "model.safetensors")
        && engine::io::is_existing_file(root / "tokenizer_config.json")
        && engine::io::is_existing_file(root / "vocab.json")
        && engine::io::is_existing_file(root / "merges.txt");
}

std::vector<runtime::NamedAsset> discover_config_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"config.json", "generation_config.json", "tokenizer_config.json"});
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    const auto root = resolve_model_root(request.model_path);
    return runtime::discover_named_assets(root, {"model.safetensors"});
}

class MioTTSLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "miotts";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        try {
            const auto root = resolve_model_root(request.model_path);
            return has_miotts_assets(root)
                && (!request.family_hint.has_value() || *request.family_hint == family());
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto assets = load_miotts_assets(resolve_model_root(request.model_path));
        runtime::ModelInspection inspection;
        inspection.model_root = assets->paths.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = assets->config.model_type;
        inspection.metadata.description = "MioTTS loaded from local Qwen3 assets with MioCodec decoding.";
        inspection.metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
        inspection.metadata.weight_candidates = {"model.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_speaker_reference = true;
        inspection.capabilities.supports_timestamps = false;
        inspection.discovered_configs = discover_config_assets(request);
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_miotts_model(resolve_model_root(request.model_path));
    }
};

}  // namespace

MioTTSLoadedModel::MioTTSLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const MioTTSAssets> assets)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      assets_(std::move(assets)) {}

const runtime::ModelMetadata & MioTTSLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & MioTTSLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> MioTTSLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("MioTTS only supports the Tts task");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioTTS currently supports offline sessions");
    }
    return std::make_unique<MioTTSSession>(task, options, assets_);
}

std::unique_ptr<MioTTSLoadedModel> load_miotts_model(const std::filesystem::path & model_path) {
    auto assets = load_miotts_assets(model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "miotts";
    metadata.variant = assets->config.model_type;
    metadata.description = "MioTTS loaded from local Qwen3 assets with MioCodec decoding.";
    metadata.config_candidates = {"config.json", "generation_config.json", "tokenizer_config.json"};
    metadata.weight_candidates = {"model.safetensors"};

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Tts, {runtime::RunMode::Offline}},
    };
    capabilities.supports_speaker_reference = true;
    capabilities.supports_timestamps = false;

    return std::make_unique<MioTTSLoadedModel>(std::move(metadata), std::move(capabilities), std::move(assets));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_miotts_loader() {
    return std::make_shared<MioTTSLoader>();
}

}  // namespace engine::models::miotts

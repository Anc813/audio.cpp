#include "engine/models/citrinet_asr/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::citrinet_asr {
namespace {

std::filesystem::path resolve_weight_path(const std::filesystem::path & model_path) {
    if (engine::io::is_existing_file(model_path)) {
        return std::filesystem::weakly_canonical(model_path);
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Citrinet ASR model path does not exist: " + model_path.string());
    }
    const auto candidate = model_path / "citrinet_256.safetensors";
    if (!engine::io::is_existing_file(candidate)) {
        throw std::runtime_error("Citrinet ASR weights not found: " + candidate.string());
    }
    return std::filesystem::weakly_canonical(candidate);
}

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    if (engine::io::is_existing_file(request.model_path)) {
        return {{"default", std::filesystem::weakly_canonical(request.model_path)}};
    }
    return runtime::discover_named_assets(
        std::filesystem::weakly_canonical(request.model_path),
        {"citrinet_256.safetensors"});
}

engine::assets::TensorStorageType citrinet_weight_type_from_options(const runtime::SessionOptions & options) {
    const auto it = options.options.find("citrinet_asr.weight_type");
    if (it == options.options.end()) {
        return engine::assets::TensorStorageType::Native;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("citrinet_asr.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

class CitrinetASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "citrinet_asr";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) resolve_citrinet_assets(resolve_weight_path(request.model_path));
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Citrinet ASR does not expose selectable config assets");
        }
        const auto weight_path = resolve_weight_path(request.model_path);
        const auto assets = resolve_citrinet_assets(weight_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets.model_root;
        inspection.metadata.family = family();
        inspection.metadata.variant = weight_path.stem().string();
        inspection.metadata.description = "Citrinet ASR loaded from safetensors weights.";
        inspection.metadata.weight_candidates = {"citrinet_256.safetensors"};
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        inspection.capabilities.supports_timestamps = false;
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_citrinet_asr_model(request);
    }
};

}  // namespace

CitrinetASRSession::CitrinetASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const CitrinetWeights> weights)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      weight_storage_type_(citrinet_weight_type_from_options(RuntimeSessionBase::options())),
      runtime_(std::move(weights), execution_context(), weight_storage_type_) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Citrinet ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Citrinet ASR only supports offline mode");
    }
}

std::string CitrinetASRSession::family() const {
    return "citrinet_asr";
}

runtime::VoiceTaskKind CitrinetASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode CitrinetASRSession::run_mode() const {
    return task_.mode;
}

void CitrinetASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Citrinet ASR prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult CitrinetASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Citrinet ASR run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Citrinet ASR run() requires audio_input");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    const auto transcription = runtime_.transcribe_audio(*request.audio_input);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{transcription.text, request.text_input.has_value() ? request.text_input->language : ""};
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

CitrinetASRLoadedModel::CitrinetASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const CitrinetWeights> weights)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      weights_(std::move(weights)) {}

const runtime::ModelMetadata & CitrinetASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & CitrinetASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> CitrinetASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<CitrinetASRSession>(task, options, weights_);
}

std::unique_ptr<CitrinetASRLoadedModel> load_citrinet_asr_model(const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::runtime_error("Citrinet ASR does not expose selectable config assets");
    }
    const auto weights = discover_weight_assets(request);
    const auto * selected_weight = runtime::select_named_asset(weights, request.weight_id, "weight");
    const auto weight_path = selected_weight != nullptr ? selected_weight->path : resolve_weight_path(request.model_path);
    const auto assets = resolve_citrinet_assets(weight_path);
    runtime::ModelMetadata metadata;
    metadata.family = "citrinet_asr";
    metadata.variant = weight_path.stem().string();
    metadata.description = "Citrinet ASR loaded from safetensors weights.";
    metadata.weight_candidates = {"citrinet_256.safetensors"};
    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.supports_timestamps = false;
    return std::make_unique<CitrinetASRLoadedModel>(
        std::move(metadata),
        std::move(capabilities),
        load_citrinet_weights_cached(assets.checkpoint_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_citrinet_asr_loader() {
    return std::make_shared<CitrinetASRLoader>();
}

}  // namespace engine::models::citrinet_asr

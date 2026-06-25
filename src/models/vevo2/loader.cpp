#include "engine/models/vevo2/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/session.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::vevo2 {
namespace {

bool has_vevo2_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors") &&
        engine::io::is_existing_file(root / "tokenizer/prosody_fvq512_6.25hz/model.safetensors") &&
        engine::io::is_existing_file(root / "contentstyle_modeling/posttrained/model.safetensors") &&
        engine::io::is_existing_file(root / "acoustic_modeling/fm_emilia101k_singnet7k_repa/model.safetensors") &&
        engine::io::is_existing_file(root / "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.safetensors") &&
        engine::io::is_existing_file(root / "vocoder/model.safetensors");
}

class Vevo2LoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit Vevo2LoadedModel(std::shared_ptr<const Vevo2Assets> assets)
        : assets_(std::move(assets)) {
        if (assets_ == nullptr) {
            throw std::runtime_error("Vevo2 loaded model requires assets");
        }
    }

    const runtime::ModelMetadata & metadata() const noexcept override {
        return assets_->metadata;
    }

    const runtime::CapabilitySet & capabilities() const noexcept override {
        return assets_->capabilities;
    }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        if (task.task != runtime::VoiceTaskKind::Tts &&
            task.task != runtime::VoiceTaskKind::VoiceConversion &&
            task.task != runtime::VoiceTaskKind::SpeechToSpeech &&
            task.task != runtime::VoiceTaskKind::Svc) {
            throw std::runtime_error("Vevo2 supports tts, vc, s2s, and svc tasks");
        }
        if (task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Vevo2 currently supports offline sessions");
        }
        return std::make_unique<Vevo2Session>(task, options, assets_);
    }

private:
    std::shared_ptr<const Vevo2Assets> assets_;
};

class Vevo2Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "vevo2";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        if (engine::io::is_existing_directory(request.model_path)) {
            return has_vevo2_assets(std::filesystem::weakly_canonical(request.model_path));
        }
        if (engine::io::is_existing_file(request.model_path)) {
            return has_vevo2_assets(std::filesystem::weakly_canonical(request.model_path.parent_path()));
        }
        return false;
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        return inspect_vevo2_model(request);
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_vevo2_model(request);
    }
};

}  // namespace

std::unique_ptr<runtime::ILoadedVoiceModel> load_vevo2_model(const runtime::ModelLoadRequest & request) {
    return std::make_unique<Vevo2LoadedModel>(load_vevo2_assets(request));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vevo2_loader() {
    return std::make_shared<Vevo2Loader>();
}

}  // namespace engine::models::vevo2

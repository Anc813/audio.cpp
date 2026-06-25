#include "engine/models/seed_vc/loader.h"

#include "engine/framework/io/filesystem.h"
#include "engine/models/seed_vc/assets.h"
#include "engine/models/seed_vc/session.h"

#include <filesystem>
#include <memory>

namespace engine::models::seed_vc {
namespace {

bool has_seed_vc_assets(const std::filesystem::path & root) {
    return engine::io::is_existing_file(root / "seed_vc_manifest.json") &&
        engine::io::is_existing_file(root / "v2/ar.safetensors") &&
        engine::io::is_existing_file(root / "v2/cfm.safetensors") &&
        engine::io::is_existing_file(root / "v1/svc.safetensors") &&
        engine::io::is_existing_file(root / "astral/bsq32.safetensors") &&
        engine::io::is_existing_file(root / "astral/bsq2048.safetensors") &&
        engine::io::is_existing_file(root / "campplus/model.safetensors") &&
        engine::io::is_existing_file(root / "rmvpe/model.safetensors") &&
        engine::io::is_existing_file(root / "bigvgan/v2_22khz_80band_256x/model.safetensors") &&
        engine::io::is_existing_file(root / "bigvgan/v2_44khz_128band_512x/model.safetensors") &&
        engine::io::is_existing_file(root / "whisper-small/model.safetensors") &&
        engine::io::is_existing_file(root / "hubert-large-ll60k/model.safetensors");
}

class SeedVcLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "seed_vc";
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            const auto root = resolve_seed_vc_model_root(request.model_path);
            return has_seed_vc_assets(root);
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        return inspect_seed_vc_model(request.model_path);
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_seed_vc_model(request.model_path);
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_seed_vc_loader() {
    return std::make_shared<SeedVcLoader>();
}

}  // namespace engine::models::seed_vc

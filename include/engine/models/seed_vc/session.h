#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/seed_vc/assets.h"

#include <memory>
#include <optional>
#include <string>

namespace engine::models::seed_vc {

struct SeedVcComponentSources;

class SeedVcSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    SeedVcSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const SeedVcAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const SeedVcAssets> assets_;
    std::shared_ptr<SeedVcComponentSources> sources_;
    std::optional<engine::assets::TensorStorageType> weight_storage_type_;
};

std::unique_ptr<runtime::ILoadedVoiceModel> load_seed_vc_model(const std::filesystem::path & model_path);

}  // namespace engine::models::seed_vc

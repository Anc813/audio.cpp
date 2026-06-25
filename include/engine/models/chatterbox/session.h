#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/models/chatterbox/assets.h"
#include "engine/models/chatterbox/conditionals.h"
#include "engine/models/chatterbox/tts.h"

#include <filesystem>
#include <memory>
#include <optional>

namespace engine::models::chatterbox {

class ChatterboxSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ChatterboxSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ChatterboxAssetPaths> assets);
    ~ChatterboxSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult run_voice_cloning(const runtime::TaskRequest & request);

    runtime::TaskSpec task_;
    std::shared_ptr<const ChatterboxAssetPaths> assets_;
    engine::assets::TensorStorageType t3_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType component_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    std::unique_ptr<ChatterboxTtsComponent> component_;
    std::optional<ChatterboxVoiceCloneConfig> voice_clone_config_;
    std::optional<ChatterboxConditionalsOutputs> cached_conditionals_;
    double cached_prompt_prep_ms_ = 0.0;
};

}  // namespace engine::models::chatterbox

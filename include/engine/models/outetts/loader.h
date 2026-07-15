#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/models/outetts/assets.h"

#include <memory>

namespace engine::models::outetts {

class OuteTTSLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit OuteTTSLoadedModel(std::shared_ptr<const OuteTTSAssets> assets);
    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    std::shared_ptr<const OuteTTSAssets> assets_;
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_outetts_loader();

}  // namespace engine::models::outetts

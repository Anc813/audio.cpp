#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::vevo2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_vevo2_loader();

}  // namespace engine::models::vevo2

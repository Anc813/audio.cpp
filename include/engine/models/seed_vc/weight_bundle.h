#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcWeightBundle {
    std::filesystem::path source_path;
    std::string name;
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, engine::core::TensorValue> tensors;
    int64_t loaded_tensor_count = 0;
    int64_t skipped_generated_constant_count = 0;
    int64_t parameter_count = 0;
};

std::shared_ptr<const SeedVcWeightBundle> load_seed_vc_weight_bundle(
    const std::filesystem::path & checkpoint_path,
    const std::string & name,
    engine::core::BackendConfig backend,
    const std::vector<std::string> & generated_constants,
    const std::vector<std::string> & generated_constant_suffixes = {},
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::Native);

}  // namespace engine::models::seed_vc

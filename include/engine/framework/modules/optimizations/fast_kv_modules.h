#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

class FastKVSetRowsModule {
public:
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & cache,
        const core::TensorValue & row,
        const core::TensorValue & row_index) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

}  // namespace engine::modules

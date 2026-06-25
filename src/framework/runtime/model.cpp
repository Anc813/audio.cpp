#include "engine/framework/runtime/model.h"

#include "engine/framework/io/filesystem.h"

#include <unordered_map>
#include <stdexcept>

namespace engine::runtime {

namespace {

std::string normalized_candidate_key(const std::string & relative_candidate) {
    return std::filesystem::path(relative_candidate).generic_string();
}

std::string candidate_stem_or_filename(const std::string & relative_candidate) {
    const auto path = std::filesystem::path(relative_candidate);
    if (!path.stem().empty()) {
        return path.stem().string();
    }
    return path.filename().string();
}

}

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates) {
    std::unordered_map<std::string, int> stem_counts;
    for (const auto & candidate : relative_candidates) {
        ++stem_counts[candidate_stem_or_filename(candidate)];
    }

    std::vector<NamedAsset> assets;
    assets.reserve(relative_candidates.size());
    for (const auto & candidate : relative_candidates) {
        const auto path = root / candidate;
        if (engine::io::is_existing_file(path)) {
            const auto base_id = candidate_stem_or_filename(candidate);
            const auto id = stem_counts[base_id] == 1 ? base_id : normalized_candidate_key(candidate);
            assets.push_back({id, std::filesystem::weakly_canonical(path)});
        }
    }
    return assets;
}

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept {
    for (const auto & asset : assets) {
        if (asset.id == id) {
            return &asset;
        }
    }
    return nullptr;
}

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role) {
    const auto * asset = find_named_asset(assets, id);
    if (asset == nullptr) {
        throw std::runtime_error("unknown " + role + " id: " + id);
    }
    return *asset;
}

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) {
    if (id.has_value()) {
        return &require_named_asset(assets, *id, role);
    }
    if (assets.empty()) {
        return nullptr;
    }
    if (assets.size() > 1) {
        throw std::runtime_error("multiple " + role + " candidates found; specify --" + role);
    }
    return &assets.front();
}

}  // namespace engine::runtime

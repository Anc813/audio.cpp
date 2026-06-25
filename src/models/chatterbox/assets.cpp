#include "engine/models/chatterbox/assets.h"

#include "engine/framework/io/filesystem.h"

#include <stdexcept>

namespace engine::models::chatterbox {

namespace {

std::filesystem::path require_existing_file(const std::filesystem::path & path, const char * what) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error(std::string("missing ") + what + ": " + path.string());
    }
    return std::filesystem::weakly_canonical(path);
}

}  // namespace

ChatterboxAssetPaths resolve_chatterbox_assets(const std::filesystem::path & model_root) {
    const auto root = std::filesystem::weakly_canonical(model_root);
    if (!engine::io::is_existing_directory(root)) {
        throw std::runtime_error("chatterbox model root does not exist: " + model_root.string());
    }
    ChatterboxAssetPaths assets;
    assets.model_root = root;
    assets.voice_encoder_weights = require_existing_file(root / "ve.safetensors", "voice encoder weights");
    assets.s3tokenizer_weights = require_existing_file(root / "s3gen.safetensors", "S3TokenizerV2 weights");
    assets.t3_english_weights = require_existing_file(root / "t3_cfg.safetensors", "T3 english weights");
    assets.t3_multilingual_v2_weights = require_existing_file(root / "t3_mtl23ls_v2.safetensors", "T3 multilingual v2 weights");
    assets.t3_multilingual_v3_weights = require_existing_file(root / "t3_mtl23ls_v3.safetensors", "T3 multilingual v3 weights");
    assets.s3gen_weights = require_existing_file(root / "s3gen.safetensors", "S3Gen weights");
    assets.english_tokenizer = require_existing_file(root / "tokenizer.json", "english tokenizer");
    assets.multilingual_tokenizer = require_existing_file(root / "grapheme_mtl_merged_expanded_v1.json", "multilingual tokenizer");
    assets.cangjie_mapping = require_existing_file(root / "Cangjie5_TC.json", "Cangjie mapping");
    assets.builtin_conditionals = require_existing_file(root / "conds.pt", "builtin conditionals");
    return assets;
}

}  // namespace engine::models::chatterbox

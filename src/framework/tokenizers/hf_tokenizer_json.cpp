#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>

namespace engine::tokenizers {
namespace {

std::string replace_all(std::string text, const std::string & needle, const std::string & replacement) {
    if (needle.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
    return text;
}

std::string decode_with_optional_skip(
    const std::vector<std::string> & id_to_token,
    const std::vector<int32_t> & ids,
    const std::string & metaspace_replacement,
    bool trim_leading_space,
    const int32_t * skip_token_id) {
    std::string text;
    for (const int32_t id : ids) {
        if (id < 0 || id >= static_cast<int32_t>(id_to_token.size())) {
            continue;
        }
        if (skip_token_id != nullptr && id == *skip_token_id) {
            continue;
        }
        text += id_to_token[static_cast<size_t>(id)];
    }
    if (!metaspace_replacement.empty()) {
        text = replace_all(text, metaspace_replacement, " ");
    }
    if (trim_leading_space) {
        while (!text.empty() && text.front() == ' ') {
            text.erase(text.begin());
        }
    }
    return text;
}

}  // namespace

HuggingFaceTokenizerJson::HuggingFaceTokenizerJson(
    std::vector<std::string> id_to_token,
    std::string metaspace_replacement,
    bool trim_leading_space)
    : id_to_token_(std::move(id_to_token)),
      metaspace_replacement_(std::move(metaspace_replacement)),
      trim_leading_space_(trim_leading_space) {}

const std::vector<std::string> & HuggingFaceTokenizerJson::id_to_token() const noexcept {
    return id_to_token_;
}

std::string HuggingFaceTokenizerJson::decode_ids(const std::vector<int32_t> & ids) const {
    return decode_with_optional_skip(id_to_token_, ids, metaspace_replacement_, trim_leading_space_, nullptr);
}

std::string HuggingFaceTokenizerJson::decode_ids(const std::vector<int32_t> & ids, int32_t skip_token_id) const {
    return decode_with_optional_skip(id_to_token_, ids, metaspace_replacement_, trim_leading_space_, &skip_token_id);
}

std::shared_ptr<HuggingFaceTokenizerJson> load_huggingface_tokenizer_json(
    const std::filesystem::path & path) {
    const engine::io::json::Value root = engine::io::json::parse_file(path);
    const engine::io::json::Value & model = root.require("model");
    const engine::io::json::Value & vocab = model.require("vocab");

    size_t max_id = 0;
    for (const auto & [_, value] : vocab.as_object()) {
        max_id = std::max(max_id, static_cast<size_t>(value.as_i64()));
    }

    std::vector<std::string> id_to_token(max_id + 1);
    for (const auto & [token, value] : vocab.as_object()) {
        const size_t id = static_cast<size_t>(value.as_i64());
        if (id >= id_to_token.size()) {
            throw std::runtime_error("token id is out of bounds while loading huggingface tokenizer");
        }
        id_to_token[id] = token;
    }

    std::string metaspace_replacement;
    bool trim_leading_space = false;
    if (const auto * decoder = root.find("decoder"); decoder != nullptr
        && decoder->is_object()
        && decoder->find("type") != nullptr
        && decoder->require("type").as_string() == "Metaspace") {
        metaspace_replacement = decoder->require("replacement").as_string();
        trim_leading_space = true;
    }

    return std::make_shared<HuggingFaceTokenizerJson>(
        std::move(id_to_token),
        std::move(metaspace_replacement),
        trim_leading_space);
}

}  // namespace engine::tokenizers

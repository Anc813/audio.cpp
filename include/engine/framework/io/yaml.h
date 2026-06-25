#pragma once

#include "engine/framework/io/config.h"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::io::yaml {

struct FlattenedDocument {
    ConfigMap scalars;
    std::unordered_map<std::string, std::vector<std::string>> lists;
};

ConfigMap parse_scalar_mapping(std::string_view text);
FlattenedDocument parse_flattened_document(std::string_view text);
bool parse_bool_scalar(const std::string & value, const std::string & key);
int require_int(const FlattenedDocument & document, const std::string & key);
float require_float(const FlattenedDocument & document, const std::string & key);
std::optional<int> optional_int(const FlattenedDocument & document, const std::string & key);

}  // namespace engine::io::yaml

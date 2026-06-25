#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::io {

using ConfigMap = std::unordered_map<std::string, std::string>;

ConfigMap parse_flat_json_object(const std::string & text);
ConfigMap parse_key_value_text(const std::string & text, char separator = '=');

ConfigMap load_config_map(const std::filesystem::path & path);
std::vector<std::string> discover_config_keys(const ConfigMap & config);

}  // namespace engine::io

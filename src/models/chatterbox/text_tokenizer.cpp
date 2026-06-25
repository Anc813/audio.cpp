#include "engine/models/chatterbox/text_tokenizer.h"

#include "engine/framework/io/filesystem.h"
#include "unicode.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::chatterbox {
struct MergePairHash {
    size_t operator()(const std::pair<std::string, std::string> & value) const noexcept {
        return std::hash<std::string>{}(value.first) ^ (std::hash<std::string>{}(value.second) << 1U);
    }
};

struct ChatterboxEnglishTokenizerModel {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, MergePairHash> merge_ranks;
    std::vector<std::string> special_tokens;
    int32_t unk_id = -1;
    int32_t start_id = -1;
    int32_t stop_id = -1;
};

namespace {

struct JsonValue {
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    Type type = Type::Null;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::unordered_map<std::string, JsonValue> object_value;
};

void skip_ws(std::string_view text, size_t & pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
}

void append_utf8_codepoint(std::string & out, uint32_t codepoint) {
    if (codepoint <= 0x7FU) {
        out.push_back(static_cast<char>(codepoint));
        return;
    }
    if (codepoint <= 0x7FFU) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    if (codepoint <= 0xFFFFU) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    if (codepoint <= 0x10FFFFU) {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        return;
    }
    throw std::runtime_error("invalid unicode codepoint in tokenizer json");
}

uint32_t parse_hex4(std::string_view text, size_t & pos) {
    if (pos + 4 > text.size()) {
        throw std::runtime_error("truncated json unicode escape");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value <<= 4U;
        const char ch = text[pos++];
        if (ch >= '0' && ch <= '9') {
            value |= static_cast<uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            value |= static_cast<uint32_t>(10 + (ch - 'a'));
        } else if (ch >= 'A' && ch <= 'F') {
            value |= static_cast<uint32_t>(10 + (ch - 'A'));
        } else {
            throw std::runtime_error("invalid hex digit in tokenizer json unicode escape");
        }
    }
    return value;
}

std::string parse_json_string(std::string_view text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '"') {
        throw std::runtime_error("expected json string");
    }
    ++pos;
    std::string out;
    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '\\') {
            if (pos >= text.size()) {
                throw std::runtime_error("invalid json escape");
            }
            const char esc = text[pos++];
            switch (esc) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                uint32_t codepoint = parse_hex4(text, pos);
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                    if (pos + 2 > text.size() || text[pos] != '\\' || text[pos + 1] != 'u') {
                        throw std::runtime_error("missing low surrogate in tokenizer json");
                    }
                    pos += 2;
                    const uint32_t low = parse_hex4(text, pos);
                    if (low < 0xDC00U || low > 0xDFFFU) {
                        throw std::runtime_error("invalid low surrogate in tokenizer json");
                    }
                    codepoint = 0x10000U + (((codepoint - 0xD800U) << 10U) | (low - 0xDC00U));
                }
                append_utf8_codepoint(out, codepoint);
                break;
            }
            default:
                throw std::runtime_error("unsupported json escape");
            }
            continue;
        }
        if (ch == '"') {
            return out;
        }
        out.push_back(ch);
    }
    throw std::runtime_error("unterminated json string");
}

double parse_json_number(std::string_view text, size_t & pos) {
    skip_ws(text, pos);
    const size_t begin = pos;
    while (pos < text.size()) {
        const char ch = text[pos];
        if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
            ++pos;
            continue;
        }
        break;
    }
    if (begin == pos) {
        throw std::runtime_error("expected json number");
    }
    return std::stod(std::string(text.substr(begin, pos - begin)));
}

JsonValue parse_json_value(std::string_view text, size_t & pos);

JsonValue parse_json_array(std::string_view text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '[') {
        throw std::runtime_error("expected json array");
    }
    ++pos;
    JsonValue value;
    value.type = JsonValue::Type::Array;
    while (true) {
        skip_ws(text, pos);
        if (pos >= text.size()) {
            throw std::runtime_error("unterminated json array");
        }
        if (text[pos] == ']') {
            ++pos;
            return value;
        }
        value.array_value.push_back(parse_json_value(text, pos));
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return value;
        }
        throw std::runtime_error("invalid json array");
    }
}

JsonValue parse_json_object(std::string_view text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size() || text[pos] != '{') {
        throw std::runtime_error("expected json object");
    }
    ++pos;
    JsonValue value;
    value.type = JsonValue::Type::Object;
    while (true) {
        skip_ws(text, pos);
        if (pos >= text.size()) {
            throw std::runtime_error("unterminated json object");
        }
        if (text[pos] == '}') {
            ++pos;
            return value;
        }
        const auto key = parse_json_string(text, pos);
        skip_ws(text, pos);
        if (pos >= text.size() || text[pos] != ':') {
            throw std::runtime_error("expected ':' in json object");
        }
        ++pos;
        value.object_value.emplace(key, parse_json_value(text, pos));
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == '}') {
            ++pos;
            return value;
        }
        throw std::runtime_error("invalid json object");
    }
}

JsonValue parse_json_value(std::string_view text, size_t & pos) {
    skip_ws(text, pos);
    if (pos >= text.size()) {
        throw std::runtime_error("unexpected end of json");
    }
    const char ch = text[pos];
    if (ch == '{') {
        return parse_json_object(text, pos);
    }
    if (ch == '[') {
        return parse_json_array(text, pos);
    }
    if (ch == '"') {
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.string_value = parse_json_string(text, pos);
        return value;
    }
    if (text.substr(pos, 4) == "null") {
        pos += 4;
        return JsonValue{};
    }
    if (text.substr(pos, 4) == "true") {
        pos += 4;
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.bool_value = true;
        return value;
    }
    if (text.substr(pos, 5) == "false") {
        pos += 5;
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.bool_value = false;
        return value;
    }
    JsonValue value;
    value.type = JsonValue::Type::Number;
    value.number_value = parse_json_number(text, pos);
    return value;
}

const JsonValue & require_object_field(const JsonValue & value, const char * field) {
    if (value.type != JsonValue::Type::Object) {
        throw std::runtime_error("expected json object");
    }
    const auto it = value.object_value.find(field);
    if (it == value.object_value.end()) {
        throw std::runtime_error(std::string("missing tokenizer json field: ") + field);
    }
    return it->second;
}

std::string require_string_field(const JsonValue & value, const char * field) {
    const auto & out = require_object_field(value, field);
    if (out.type != JsonValue::Type::String) {
        throw std::runtime_error(std::string("expected string tokenizer json field: ") + field);
    }
    return out.string_value;
}

const std::vector<JsonValue> & require_array_field(const JsonValue & value, const char * field) {
    const auto & out = require_object_field(value, field);
    if (out.type != JsonValue::Type::Array) {
        throw std::runtime_error(std::string("expected array tokenizer json field: ") + field);
    }
    return out.array_value;
}

const std::unordered_map<std::string, JsonValue> & require_map_field(const JsonValue & value, const char * field) {
    const auto & out = require_object_field(value, field);
    if (out.type != JsonValue::Type::Object) {
        throw std::runtime_error(std::string("expected object tokenizer json field: ") + field);
    }
    return out.object_value;
}

int32_t as_i32(const JsonValue & value) {
    if (value.type != JsonValue::Type::Number) {
        throw std::runtime_error("expected numeric tokenizer json value");
    }
    const double rounded = std::round(value.number_value);
    if (std::fabs(value.number_value - rounded) > 1.0e-6) {
        throw std::runtime_error("expected integer tokenizer json value");
    }
    return static_cast<int32_t>(rounded);
}

std::vector<std::string> split_utf8_codepoints(const std::string & text) {
    std::vector<std::string> parts;
    for (size_t pos = 0; pos < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[pos]);
        size_t width = 1;
        if ((lead & 0x80U) == 0) {
            width = 1;
        } else if ((lead & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((lead & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((lead & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error("invalid utf-8 byte sequence in text tokenizer");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error("truncated utf-8 byte sequence in text tokenizer");
        }
        parts.emplace_back(text.substr(pos, width));
        pos += width;
    }
    return parts;
}

std::pair<uint32_t, size_t> read_utf8_codepoint_at(const std::string & text, size_t pos) {
    if (pos >= text.size()) {
        throw std::runtime_error("invalid utf-8 offset in chatterbox tokenizer");
    }
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    if ((lead & 0x80U) == 0) {
        return {lead, 1};
    }
    if ((lead & 0xE0U) == 0xC0U) {
        if (pos + 2 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {((lead & 0x1FU) << 6U) | (static_cast<unsigned char>(text[pos + 1]) & 0x3FU), 2};
    }
    if ((lead & 0xF0U) == 0xE0U) {
        if (pos + 3 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {
            ((lead & 0x0FU) << 12U) |
                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 6U) |
                (static_cast<unsigned char>(text[pos + 2]) & 0x3FU),
            3};
    }
    if ((lead & 0xF8U) == 0xF0U) {
        if (pos + 4 > text.size()) {
            throw std::runtime_error("truncated utf-8 text in chatterbox tokenizer");
        }
        return {
            ((lead & 0x07U) << 18U) |
                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 12U) |
                ((static_cast<unsigned char>(text[pos + 2]) & 0x3FU) << 6U) |
                (static_cast<unsigned char>(text[pos + 3]) & 0x3FU),
            4};
    }
    throw std::runtime_error("invalid utf-8 text in chatterbox tokenizer");
}

std::vector<int32_t> bpe_encode_segment(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & segment) {
    auto symbols = split_utf8_codepoints(segment);
    if (symbols.empty()) {
        return {};
    }
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = tokenizer.merge_ranks.find({symbols[i], symbols[i + 1]});
            if (it == tokenizer.merge_ranks.end()) {
                continue;
            }
            if (it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
            }
        }
        if (best_index >= symbols.size()) {
            break;
        }
        std::vector<std::string> merged;
        merged.reserve(symbols.size() - 1);
        for (size_t i = 0; i < symbols.size();) {
            if (i == best_index) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols = std::move(merged);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const auto & symbol : symbols) {
        const auto it = tokenizer.vocab.find(symbol);
        ids.push_back(it != tokenizer.vocab.end() ? it->second : tokenizer.unk_id);
    }
    return ids;
}

bool starts_with_at(const std::string & text, size_t pos, const std::string & token) {
    return pos + token.size() <= text.size() && text.compare(pos, token.size(), token) == 0;
}

std::string replace_spaces_with_marker(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char ch : text) {
        if (ch == ' ') {
            out += "[SPACE]";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string lower_and_normalize_nfd(const std::string & text) {
    auto codepoints = unicode_cpts_from_utf8(text);
    for (uint32_t & codepoint : codepoints) {
        codepoint = unicode_tolower(codepoint);
    }
    codepoints = unicode_cpts_normalize_nfd(codepoints);
    std::string out;
    for (const uint32_t codepoint : codepoints) {
        out += unicode_cpt_to_utf8(codepoint);
    }
    return out;
}

std::string trim_spaces(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) { return !is_space(static_cast<unsigned char>(ch)); }).base(), value.end());
    return value;
}

std::string decompose_korean_hangul(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    const std::string input(text);
    for (size_t pos = 0; pos < input.size();) {
        const auto [codepoint, width] = read_utf8_codepoint_at(input, pos);
        if (codepoint >= 0xAC00U && codepoint <= 0xD7AFU) {
            const uint32_t base = codepoint - 0xAC00U;
            append_utf8_codepoint(out, 0x1100U + base / (21U * 28U));
            append_utf8_codepoint(out, 0x1161U + (base % (21U * 28U)) / 28U);
            const uint32_t final = base % 28U;
            if (final > 0) {
                append_utf8_codepoint(out, 0x11A7U + final);
            }
        } else {
            out.append(input, pos, width);
        }
        pos += width;
    }
    return trim_spaces(std::move(out));
}

const std::unordered_set<std::string> & supported_chatterbox_languages() {
    static const std::vector<std::string> codes = supported_chatterbox_language_codes();
    static const std::unordered_set<std::string> languages(codes.begin(), codes.end());
    return languages;
}

std::vector<int32_t> encode_marked_text(
    const ChatterboxEnglishTokenizerModel & tokenizer,
    const std::string & marked) {
    std::vector<int32_t> ids;
    std::string segment;
    for (size_t pos = 0; pos < marked.size();) {
        bool matched_special = false;
        for (const auto & special : tokenizer.special_tokens) {
            if (!starts_with_at(marked, pos, special)) {
                continue;
            }
            if (!segment.empty()) {
                const auto segment_ids = bpe_encode_segment(tokenizer, segment);
                ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
                segment.clear();
            }
            const auto it = tokenizer.vocab.find(special);
            ids.push_back(it != tokenizer.vocab.end() ? it->second : tokenizer.unk_id);
            pos += special.size();
            matched_special = true;
            break;
        }
        if (matched_special) {
            continue;
        }
        const auto [_, width] = read_utf8_codepoint_at(marked, pos);
        segment.append(marked, pos, width);
        pos += width;
    }
    if (!segment.empty()) {
        const auto segment_ids = bpe_encode_segment(tokenizer, segment);
        ids.insert(ids.end(), segment_ids.begin(), segment_ids.end());
    }
    return ids;
}

}  // namespace

std::shared_ptr<const ChatterboxEnglishTokenizerModel> load_chatterbox_english_tokenizer(
    const std::filesystem::path & tokenizer_path) {
    const auto json_text = engine::io::read_text_file(tokenizer_path);
    size_t pos = 0;
    const auto root = parse_json_value(json_text, pos);
    const auto & model = require_object_field(root, "model");
    if (require_string_field(model, "type") != "BPE") {
        throw std::runtime_error("Chatterbox English tokenizer expects BPE model");
    }
    const auto & vocab_map = require_map_field(model, "vocab");
    const auto & merges = require_array_field(model, "merges");
    auto tokenizer = std::make_shared<ChatterboxEnglishTokenizerModel>();
    tokenizer->vocab.reserve(vocab_map.size());
    for (const auto & [piece, id_value] : vocab_map) {
        tokenizer->vocab.emplace(piece, as_i32(id_value));
    }
    tokenizer->unk_id = tokenizer->vocab.at("[UNK]");
    tokenizer->start_id = tokenizer->vocab.at("[START]");
    tokenizer->stop_id = tokenizer->vocab.at("[STOP]");

    int32_t rank = 0;
    for (const auto & item : merges) {
        if (item.type != JsonValue::Type::String) {
            throw std::runtime_error("expected tokenizer merge string");
        }
        const auto sep = item.string_value.find(' ');
        if (sep == std::string::npos) {
            throw std::runtime_error("invalid tokenizer merge pair");
        }
        tokenizer->merge_ranks.emplace(
            std::make_pair(item.string_value.substr(0, sep), item.string_value.substr(sep + 1)),
            rank++);
    }

    const auto & added_tokens = require_array_field(root, "added_tokens");
    for (const auto & item : added_tokens) {
        if (item.type != JsonValue::Type::Object) {
            continue;
        }
        const auto content_it = item.object_value.find("content");
        if (content_it == item.object_value.end() || content_it->second.type != JsonValue::Type::String) {
            continue;
        }
        tokenizer->special_tokens.push_back(content_it->second.string_value);
    }
    std::sort(
        tokenizer->special_tokens.begin(),
        tokenizer->special_tokens.end(),
        [](const std::string & a, const std::string & b) {
            if (a.size() != b.size()) {
                return a.size() > b.size();
            }
            return a < b;
        });
    tokenizer->special_tokens.erase(
        std::unique(tokenizer->special_tokens.begin(), tokenizer->special_tokens.end()),
        tokenizer->special_tokens.end());
    return tokenizer;
}

std::string normalize_chatterbox_tts_text(const std::string & text) {
    if (text.empty()) {
        return "You need to add some text for me to talk.";
    }

    std::string normalized = text;
    if (!normalized.empty() && std::islower(static_cast<unsigned char>(normalized.front())) != 0) {
        normalized.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(normalized.front())));
    }

    std::string compact;
    compact.reserve(normalized.size());
    bool previous_space = false;
    for (char ch : normalized) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!previous_space) {
                compact.push_back(' ');
                previous_space = true;
            }
            continue;
        }
        compact.push_back(ch);
        previous_space = false;
    }
    normalized = trim_spaces(std::move(compact));

    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"...", ", "},
        {"…", ", "},
        {":", ","},
        {" - ", ", "},
        {";", ", "},
        {"—", "-"},
        {"–", "-"},
        {" ,", ","},
        {"“", "\""},
        {"”", "\""},
        {"‘", "'"},
        {"’", "'"},
    };
    for (const auto & [old_text, new_text] : replacements) {
        size_t search = 0;
        while ((search = normalized.find(old_text, search)) != std::string::npos) {
            normalized.replace(search, old_text.size(), new_text);
            search += new_text.size();
        }
    }

    normalized = trim_spaces(std::move(normalized));
    const std::vector<std::string> sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"};
    const bool has_sentence_ender = std::any_of(
        sentence_enders.begin(),
        sentence_enders.end(),
        [&](const std::string & ending) {
            return normalized.size() >= ending.size() &&
                normalized.compare(normalized.size() - ending.size(), ending.size(), ending) == 0;
        });
    if (normalized.empty() || !has_sentence_ender) {
        normalized.push_back('.');
    }
    return normalized;
}

std::vector<std::string> supported_chatterbox_language_codes() {
    return {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "hi", "it",
        "ko", "ms", "nl", "no", "pl", "pt", "sv", "sw", "tr",
    };
}

std::string normalize_chatterbox_language_code(const std::string & language) {
    std::string normalized = lower_ascii(trim_spaces(language.empty() ? "en" : language));
    const auto dash = normalized.find('-');
    if (dash != std::string::npos) {
        normalized = normalized.substr(0, dash);
    }
    if (supported_chatterbox_languages().count(normalized) == 0) {
        throw std::runtime_error("unsupported Chatterbox language: " + language);
    }
    return normalized;
}

bool chatterbox_language_uses_multilingual_t3(const std::string & language) {
    return normalize_chatterbox_language_code(language) != "en";
}

std::vector<int32_t> encode_chatterbox_english_text(
    const ChatterboxEnglishTokenizerModel & tokenizer_base,
    const std::string & text) {
    const auto & tokenizer = tokenizer_base;
    return encode_marked_text(tokenizer, replace_spaces_with_marker(text));
}

std::vector<int32_t> encode_chatterbox_multilingual_text(
    const ChatterboxEnglishTokenizerModel & tokenizer_base,
    const std::string & text,
    const std::string & language) {
    const auto & tokenizer = tokenizer_base;
    const std::string normalized_language = normalize_chatterbox_language_code(language);
    std::string prepared = lower_and_normalize_nfd(text);
    if (normalized_language == "ko") {
        prepared = decompose_korean_hangul(prepared);
    }
    return encode_marked_text(tokenizer, "[" + normalized_language + "]" + replace_spaces_with_marker(prepared));
}

int32_t chatterbox_english_start_token_id(const ChatterboxEnglishTokenizerModel & tokenizer_base) {
    return tokenizer_base.start_id;
}

int32_t chatterbox_english_stop_token_id(const ChatterboxEnglishTokenizerModel & tokenizer_base) {
    return tokenizer_base.stop_id;
}

}  // namespace engine::models::chatterbox

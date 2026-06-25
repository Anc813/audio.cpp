#include "engine/framework/text/chunking.h"

#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace engine::text {
namespace {

struct Utf8Span {
    size_t start = 0;
    size_t end = 0;
    std::string_view text;
};

struct WordRange {
    size_t span_start = 0;
    size_t span_end = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
    bool sentence_break = false;
    bool clause_break = false;
};

bool is_ascii_space(std::string_view token) noexcept {
    return token.size() == 1 && std::isspace(static_cast<unsigned char>(token.front())) != 0;
}

bool is_sentence_break(std::string_view token) {
    return token == "." || token == "!" || token == "?" ||
           token == u8"。" || token == u8"！" || token == u8"？";
}

bool is_clause_break(std::string_view token) {
    return token == "," || token == ";" || token == ":" ||
           token == u8"，" || token == u8"、" || token == u8"；" || token == u8"：";
}

std::vector<Utf8Span> split_utf8_spans(std::string_view text, std::string_view label) {
    std::vector<Utf8Span> spans;
    spans.reserve(utf8_codepoint_count(text, label));
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        if (ch <= 0x7FU) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error(std::string(label) + " contains invalid UTF-8");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error(std::string(label) + " contains truncated UTF-8");
        }
        for (size_t i = 1; i < width; ++i) {
            if (!is_utf8_continuation(static_cast<unsigned char>(text[pos + i]))) {
                throw std::runtime_error(std::string(label) + " contains invalid UTF-8 continuation byte");
            }
        }
        spans.push_back({pos, pos + width, text.substr(pos, width)});
        pos += width;
    }
    return spans;
}

}  // namespace

std::vector<std::string> split_text_chunks(
    std::string_view text,
    int64_t codepoint_budget) {
    if (codepoint_budget <= 0) {
        throw std::runtime_error("text chunk budget must be positive");
    }
    const std::string trimmed = engine::io::trim_ascii_whitespace(std::string(text));
    if (trimmed.empty()) {
        return {};
    }
    const auto spans = split_utf8_spans(trimmed, "text chunk");
    if (static_cast<int64_t>(spans.size()) <= codepoint_budget) {
        return {trimmed};
    }

    std::vector<WordRange> words;
    size_t span_pos = 0;
    while (span_pos < spans.size()) {
        while (span_pos < spans.size() && is_ascii_space(spans[span_pos].text)) {
            ++span_pos;
        }
        if (span_pos >= spans.size()) {
            break;
        }
        const size_t word_start = span_pos;
        size_t word_end = span_pos + 1;
        while (word_end < spans.size() && !is_ascii_space(spans[word_end].text)) {
            ++word_end;
        }
        const auto last = spans[word_end - 1].text;
        words.push_back({
            word_start,
            word_end,
            spans[word_start].start,
            spans[word_end - 1].end,
            is_sentence_break(last),
            is_clause_break(last),
        });
        span_pos = word_end;
    }
    if (words.empty()) {
        return {};
    }

    std::vector<std::string> chunks;
    size_t word_start = 0;
    while (word_start < words.size()) {
        size_t hard_end = word_start;
        while (hard_end < words.size()) {
            const size_t chunk_span_start = words[word_start].span_start;
            const size_t candidate_span_end = words[hard_end].span_end;
            const auto chunk_codepoints = static_cast<int64_t>(candidate_span_end - chunk_span_start);
            if (chunk_codepoints > codepoint_budget) {
                break;
            }
            ++hard_end;
        }

        if (hard_end == word_start) {
            hard_end = word_start + 1;
        }

        size_t chunk_end = hard_end;
        if (hard_end > word_start + 1) {
            for (size_t i = hard_end; i > word_start + 1; --i) {
                if (words[i - 1].sentence_break) {
                    chunk_end = i;
                    break;
                }
            }
            if (chunk_end == hard_end) {
                for (size_t i = hard_end; i > word_start + 1; --i) {
                    if (words[i - 1].clause_break) {
                        chunk_end = i;
                        break;
                    }
                }
            }
        }

        const size_t byte_start = words[word_start].byte_start;
        const size_t byte_end = words[chunk_end - 1].byte_end;
        auto chunk = engine::io::trim_ascii_whitespace(trimmed.substr(byte_start, byte_end - byte_start));
        if (!chunk.empty()) {
            chunks.push_back(std::move(chunk));
        }
        word_start = chunk_end;
    }
    return chunks;
}

std::optional<int64_t> parse_text_chunk_size_override(
    const std::unordered_map<std::string, std::string> & options) {
    const auto match = runtime::find_option_match(options, {"text_chunk_size", "chunk_size"});
    if (!match.has_value()) {
        return std::nullopt;
    }
    const int64_t value = std::stoll(match->value);
    if (value <= 0) {
        throw std::runtime_error(std::string(match->key) + " must be positive");
    }
    return value;
}

}  // namespace engine::text

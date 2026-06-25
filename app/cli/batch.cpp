#include "batch.h"

#include "args.h"
#include "request.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace minitts::cli {
namespace {

bool is_wav_path(const std::filesystem::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".wav";
}

minitts::app::AppBatchRequest build_request_sequence_from_json(
    const std::filesystem::path & sequence_path) {
    const auto root = engine::io::json::parse_file(sequence_path);
    const auto * requests_value = root.is_array() ? &root : root.find("requests");
    if (requests_value == nullptr) {
        throw std::runtime_error("request sequence json requires an array or a requests array");
    }
    minitts::app::AppBatchRequest batch;
    const auto base_dir = sequence_path.parent_path();
    int index = 0;
    for (const auto & item : requests_value->as_array()) {
        std::ostringstream fallback;
        fallback << "request_" << index;
        batch.requests.push_back(minitts::app::AppRequest{
            json_optional_string(item, "id").value_or(fallback.str()),
            build_request_from_json(item, base_dir),
        });
        ++index;
    }
    if (batch.requests.empty()) {
        throw std::runtime_error("request sequence must contain at least one request");
    }
    return batch;
}

minitts::app::AppBatchRequest build_text_file_batch(
    const std::filesystem::path & path,
    const engine::runtime::TaskRequest & base_request,
    const std::string & language) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open --batch-text-file: " + path.string());
    }
    minitts::app::AppBatchRequest batch;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        auto request = base_request;
        request.text_input = engine::runtime::Transcript{line, language};
        std::ostringstream id;
        id << "line_" << line_number;
        batch.requests.push_back(minitts::app::AppRequest{id.str(), std::move(request)});
    }
    if (batch.requests.empty()) {
        throw std::runtime_error("--batch-text-file contains no non-empty requests");
    }
    return batch;
}

minitts::app::AppBatchRequest build_audio_dir_batch(
    const std::filesystem::path & dir,
    const engine::runtime::TaskRequest & base_request,
    const std::string & audio_role) {
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error("--batch-audio-dir must be an existing directory: " + dir.string());
    }
    std::vector<std::filesystem::path> audio_paths;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_wav_path(entry.path())) {
            audio_paths.push_back(entry.path());
        }
    }
    std::sort(audio_paths.begin(), audio_paths.end());
    minitts::app::AppBatchRequest batch;
    batch.requests.reserve(audio_paths.size());
    for (const auto & path : audio_paths) {
        auto request = base_request;
        if (audio_role == "audio") {
            request.audio_input = read_audio_buffer(path);
        } else if (audio_role == "voice_ref") {
            if (!request.voice.has_value()) {
                request.voice = engine::runtime::VoiceCondition{};
            }
            if (!request.voice->speaker.has_value()) {
                request.voice->speaker = engine::runtime::VoiceReference{};
            }
            request.voice->speaker->audio = read_audio_buffer(path);
        } else if (audio_role == "source_audio" ||
                   audio_role == "target_voice" ||
                   audio_role == "prosody_ref" ||
                   audio_role == "style_ref") {
            set_option(request.options, audio_role, path.string());
        } else {
            throw std::runtime_error(
                "--batch-audio-role must be audio, voice_ref, source_audio, target_voice, prosody_ref, or style_ref");
        }
        batch.requests.push_back(minitts::app::AppRequest{path.stem().string(), std::move(request)});
    }
    if (batch.requests.empty()) {
        throw std::runtime_error("--batch-audio-dir contains no .wav files: " + dir.string());
    }
    return batch;
}

}  // namespace

bool has_batch_input(int argc, char ** argv) {
    int count = 0;
    count += optional_path_arg(argc, argv, "--request-sequence").has_value() ? 1 : 0;
    count += optional_path_arg(argc, argv, "--batch-text-file").has_value() ? 1 : 0;
    count += optional_path_arg(argc, argv, "--batch-audio-dir").has_value() ? 1 : 0;
    return count > 0;
}

minitts::app::AppBatchRequest build_batch_request_from_cli(
    int argc,
    char ** argv,
    const engine::runtime::TaskRequest & base_request) {
    const auto request_sequence_path = optional_path_arg(argc, argv, "--request-sequence");
    const auto batch_text_file = optional_path_arg(argc, argv, "--batch-text-file");
    const auto batch_audio_dir = optional_path_arg(argc, argv, "--batch-audio-dir");
    const int count =
        (request_sequence_path.has_value() ? 1 : 0) +
        (batch_text_file.has_value() ? 1 : 0) +
        (batch_audio_dir.has_value() ? 1 : 0);
    if (count == 0) {
        return minitts::app::AppBatchRequest{};
    }
    if (count > 1) {
        throw std::runtime_error("choose only one of --request-sequence, --batch-text-file, or --batch-audio-dir");
    }
    if (request_sequence_path.has_value()) {
        return build_request_sequence_from_json(*request_sequence_path);
    }
    if (batch_text_file.has_value()) {
        return build_text_file_batch(
            *batch_text_file,
            base_request,
            find_arg(argc, argv, "--language").value_or(""));
    }
    return build_audio_dir_batch(
        *batch_audio_dir,
        base_request,
        find_arg(argc, argv, "--batch-audio-role").value_or("audio"));
}

}  // namespace minitts::cli

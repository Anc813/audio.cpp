#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/voxcpm2/assets.h"
#include "engine/models/voxcpm2/audiovae.h"
#include "engine/models/voxcpm2/generator.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace engine::models::voxcpm2 {

class VoxCPM2Session final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession {
public:
  VoxCPM2Session(runtime::TaskSpec task, runtime::SessionOptions options,
                 std::shared_ptr<const VoxCPM2Assets> assets);

  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;
  runtime::TaskResult run_streaming(const runtime::TaskRequest &request);

private:
  struct EncodedPromptCacheEntry {
    std::string prompt_text;
    std::optional<runtime::AudioBuffer> prompt_audio;
    std::optional<runtime::AudioBuffer> reference_audio;
    VoxCPM2EncodedPrompt encoded;
  };

  VoxCPM2GenerationOptions
  generation_options_from_request(const runtime::TaskRequest &request) const;
  void validate_request(const runtime::TaskRequest &request) const;
  const VoxCPM2EncodedPrompt *encoded_prompt_for_request(
      const std::optional<runtime::AudioBuffer> &prompt_audio,
      const std::string &prompt_text,
      const std::optional<runtime::AudioBuffer> &reference_audio);

  runtime::TaskSpec task_;
  std::shared_ptr<const VoxCPM2Assets> assets_;
  VoxCPM2FeatureGeneratorConfig generator_config_;
  VoxCPM2AudioVAEDecoderConfig decoder_config_;
  std::unique_ptr<VoxCPM2FeatureGeneratorRuntime> generator_;
  std::unique_ptr<VoxCPM2AudioVAEDecoderRuntime> decoder_;
  std::optional<EncodedPromptCacheEntry> encoded_prompt_cache_;
};

} // namespace engine::models::voxcpm2

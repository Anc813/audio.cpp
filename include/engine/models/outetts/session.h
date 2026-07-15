#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/outetts/assets.h"
#include "engine/models/outetts/dac.h"
#include "engine/models/outetts/llama.h"
#include "engine/models/outetts/tokenizer.h"

#include <memory>
#include <optional>

namespace engine::models::outetts {

class OuteTTSSession final : public runtime::RuntimeSessionBase,
                             public runtime::IOfflineVoiceTaskSession {
public:
  OuteTTSSession(runtime::TaskSpec task, runtime::SessionOptions options,
                 std::shared_ptr<const OuteTTSAssets> assets);
  std::string family() const override;
  runtime::VoiceTaskKind task_kind() const override;
  runtime::RunMode run_mode() const override;
  void prepare(const runtime::SessionPreparationRequest &request) override;
  runtime::TaskResult run(const runtime::TaskRequest &request) override;

private:
  OuteTTSLlamaRuntime &llama(bool voice_cloning);

  runtime::TaskSpec task_;
  std::shared_ptr<const OuteTTSAssets> assets_;
  OuteTTSTokenizer tokenizer_;
  std::unique_ptr<OuteTTSLlamaRuntime> llama_;
  std::unique_ptr<OuteTTSLlamaRuntime> clone_llama_;
  OuteTTSDacDecoder dac_;
  std::optional<OuteTTSVoiceProfile> voice_profile_;
};

} // namespace engine::models::outetts

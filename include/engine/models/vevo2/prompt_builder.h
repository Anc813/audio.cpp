#pragma once

#include "engine/models/vevo2/types.h"

namespace engine::models::vevo2 {

class Vevo2PromptBuilder {
public:
    Vevo2PromptParts build(
        const Vevo2Request & request,
        const Vevo2TokenSequence & prosody_tokens,
        const Vevo2TokenSequence & style_content_tokens) const;

private:
    std::string build_text_prompt(const Vevo2Request & request) const;
    std::string build_prosody_prompt(
        const Vevo2Request & request,
        const Vevo2TokenSequence & prosody_tokens) const;
    std::string build_content_style_prompt(const Vevo2TokenSequence & style_content_tokens) const;
};

}  // namespace engine::models::vevo2

#pragma once

#include "engine/models/chatterbox/components.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine::models::chatterbox::components {

using CampplusEncoderOutputs = engine::models::chatterbox::SpeakerEncoderOutputs;

struct S3TokenizerLogMelOutputs {
    std::vector<float> log_mel;
    int64_t n_mels = 0;
    int64_t frames = 0;
};

struct S3PromptMelOutputs {
    std::vector<float> mel;
    int64_t n_mels = 0;
    int64_t frames = 0;
};

struct CampplusFbankOutputs {
    std::vector<float> features;
    int64_t frames = 0;
    int64_t dims = 0;
};

uint64_t choose_seed(uint64_t seed);

S3TokenizerLogMelOutputs compute_s3tokenizer_log_mel(const runtime::AudioBuffer & audio);
S3PromptMelOutputs compute_s3_prompt_mel(const runtime::AudioBuffer & audio);
CampplusFbankOutputs compute_campplus_fbank(const runtime::AudioBuffer & audio);

struct CampplusEncoderWeights {
    struct BatchNorm1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        engine::core::TensorValue scale_tensor;
        engine::core::TensorValue shift_tensor;
        bool affine = true;
    };

    struct BatchNorm2dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        engine::core::TensorValue scale_tensor;
        engine::core::TensorValue shift_tensor;
    };

    struct Conv1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool use_bias = false;
    };

    struct Conv2dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        bool use_bias = false;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel_h = 0;
        int64_t kernel_w = 0;
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        int64_t padding_h = 0;
        int64_t padding_w = 0;
    };

    struct BasicResBlockWeights {
        Conv2dWeights conv1;
        BatchNorm2dWeights bn1;
        Conv2dWeights conv1_folded;
        Conv2dWeights conv2;
        BatchNorm2dWeights bn2;
        Conv2dWeights conv2_folded;
        bool use_shortcut = false;
        Conv2dWeights shortcut_conv;
        BatchNorm2dWeights shortcut_bn;
        Conv2dWeights shortcut_conv_folded;
    };

    struct CAMLayerWeights {
        Conv1dWeights linear_local;
        Conv1dWeights linear1;
        Conv1dWeights linear2;
    };

    struct CAMDenseTDNNLayerWeights {
        BatchNorm1dWeights nonlinear1_bn;
        Conv1dWeights linear1;
        BatchNorm1dWeights nonlinear2_bn;
        CAMLayerWeights cam_layer;
    };

    struct CAMDenseTDNNBlockWeights {
        std::vector<CAMDenseTDNNLayerWeights> layers;
    };

    struct TransitLayerWeights {
        BatchNorm1dWeights nonlinear_bn;
        Conv1dWeights linear;
    };

    Conv2dWeights head_conv1;
    BatchNorm2dWeights head_bn1;
    Conv2dWeights head_conv1_folded;
    std::vector<BasicResBlockWeights> head_layer1;
    std::vector<BasicResBlockWeights> head_layer2;
    Conv2dWeights head_conv2;
    BatchNorm2dWeights head_bn2;
    Conv2dWeights head_conv2_folded;

    Conv1dWeights tdnn_linear;
    BatchNorm1dWeights tdnn_bn;
    Conv1dWeights tdnn_linear_folded;
    std::vector<CAMDenseTDNNBlockWeights> blocks;
    std::vector<TransitLayerWeights> transits;
    BatchNorm1dWeights out_nonlinear_bn;
    Conv1dWeights dense_linear;
    BatchNorm1dWeights dense_bn;
    Conv1dWeights dense_linear_folded;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct S3TokenizerV2Weights {
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
    };

    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };

    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct BlockWeights {
        LayerNormWeights attn_ln;
        LinearWeights attn_qkv_packed;
        LinearWeights attn_out;
        engine::core::TensorValue fsmn_weight_tensor;
        LayerNormWeights mlp_ln;
        LinearWeights mlp_fc1;
        LinearWeights mlp_fc2;
    };

    Conv1dWeights conv1;
    Conv1dWeights conv2;
    std::vector<BlockWeights> blocks;
    LinearWeights quantizer_project_down;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

CampplusEncoderOutputs compute_campplus_embedding_from_audio(
    const CampplusEncoderWeights & weights,
    const runtime::AudioBuffer & audio,
    engine::core::BackendConfig backend = {});

std::vector<float> read_f32_tensor(
    const engine::assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape);

CampplusEncoderWeights::Conv1dWeights load_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type = engine::assets::TensorStorageType::Native);

}  // namespace engine::models::chatterbox::components

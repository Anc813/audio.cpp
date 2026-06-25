#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::common {

class ConstantTensorCache {
public:
    ConstantTensorCache(
        ggml_backend_t backend,
        int threads,
        std::string name,
        size_t context_bytes = 128ull * 1024ull * 1024ull)
        : backend_(backend),
          threads_(std::max(1, threads)),
          name_(std::move(name)) {
        if (backend_ == nullptr) {
            throw std::runtime_error(name_ + " constant cache backend is not initialized");
        }
        ggml_init_params params{context_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize " + name_ + " constant tensor cache");
        }
    }

    ~ConstantTensorCache() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    ConstantTensorCache(const ConstantTensorCache &) = delete;
    ConstantTensorCache & operator=(const ConstantTensorCache &) = delete;

    void begin_graph() {
        cursor_ = 0;
    }

    void finish_graph() const {
        if (cursor_ != entries_.size()) {
            throw std::runtime_error(name_ + " constant tensor cache graph used a different tensor sequence");
        }
    }

    core::TensorValue make_f32(
        const core::TensorShape & shape,
        const std::vector<float> & values) {
        const size_t size = values.size() * sizeof(float);
        if (static_cast<size_t>(shape.num_elements()) * ggml_type_size(GGML_TYPE_F32) != size) {
            throw std::runtime_error(name_ + " constant tensor cache byte size mismatch");
        }
        return make_tensor(shape, GGML_TYPE_F32, values.data(), size);
    }

    core::TensorValue make_tensor(
        const core::TensorShape & shape,
        ggml_type type,
        const void * data,
        size_t size) {
        const auto ggml_dims = core::to_ggml_dims(shape);
        ggml_tensor * tensor = nullptr;
        switch (shape.rank) {
            case 1:
                tensor = tensor_1d(type, ggml_dims[0], data, size);
                break;
            case 2:
                tensor = tensor_2d(type, ggml_dims[0], ggml_dims[1], data, size);
                break;
            case 3:
                tensor = tensor_3d(type, ggml_dims[0], ggml_dims[1], ggml_dims[2], data, size);
                break;
            case 4:
                tensor = tensor_4d(type, ggml_dims[0], ggml_dims[1], ggml_dims[2], ggml_dims[3], data, size);
                break;
            default:
                throw std::runtime_error(name_ + " constant tensor cache unsupported tensor rank");
        }
        return core::wrap_tensor(tensor, shape, type);
    }

    ggml_tensor * tensor_1d(ggml_type type, int64_t ne0, const void * data, size_t size) {
        return get_or_create(type, {ne0}, data, size, [&]() {
            return ggml_new_tensor_1d(ctx_.get(), type, ne0);
        });
    }

    ggml_tensor * tensor_2d(ggml_type type, int64_t ne0, int64_t ne1, const void * data, size_t size) {
        return get_or_create(type, {ne0, ne1}, data, size, [&]() {
            return ggml_new_tensor_2d(ctx_.get(), type, ne0, ne1);
        });
    }

    ggml_tensor * tensor_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, const void * data, size_t size) {
        return get_or_create(type, {ne0, ne1, ne2}, data, size, [&]() {
            return ggml_new_tensor_3d(ctx_.get(), type, ne0, ne1, ne2);
        });
    }

    ggml_tensor * tensor_4d(
        ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3,
        const void * data,
        size_t size) {
        return get_or_create(type, {ne0, ne1, ne2, ne3}, data, size, [&]() {
            return ggml_new_tensor_4d(ctx_.get(), type, ne0, ne1, ne2, ne3);
        });
    }

    double ensure_uploaded() {
        if (buffer_ != nullptr) {
            return 0.0;
        }
        if (entries_.empty()) {
            return 0.0;
        }
        core::set_backend_threads(backend_, threads_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate " + name_ + " constant tensor cache");
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const auto upload_start = std::chrono::steady_clock::now();
        for (const auto & entry : entries_) {
            ggml_backend_tensor_set(entry.tensor, entry.bytes.data(), 0, entry.bytes.size());
        }
        const auto upload_end = std::chrono::steady_clock::now();
        for (auto & entry : entries_) {
            entry.bytes.clear();
            entry.bytes.shrink_to_fit();
        }
        return std::chrono::duration<double, std::milli>(upload_end - upload_start).count();
    }

private:
    struct GgmlContextDeleter {
        void operator()(ggml_context * ctx) const noexcept {
            if (ctx != nullptr) {
                ggml_free(ctx);
            }
        }
    };

    struct Entry {
        ggml_type type = GGML_TYPE_F32;
        std::vector<int64_t> dims;
        ggml_tensor * tensor = nullptr;
        size_t size = 0;
        std::vector<uint8_t> bytes;
    };

    template <typename MakeTensor>
    ggml_tensor * get_or_create(
        ggml_type type,
        std::vector<int64_t> dims,
        const void * data,
        size_t size,
        MakeTensor make_tensor) {
        if (buffer_ != nullptr) {
            if (cursor_ >= entries_.size()) {
                throw std::runtime_error(name_ + " constant tensor cache saw a new tensor after upload");
            }
            const auto & entry = entries_[cursor_++];
            if (entry.type != type || entry.dims != dims || entry.size != size) {
                throw std::runtime_error(name_ + " constant tensor cache tensor sequence mismatch");
            }
            return entry.tensor;
        }

        ggml_tensor * tensor = make_tensor();
        if (ggml_nbytes(tensor) != size) {
            throw std::runtime_error(name_ + " constant tensor cache byte size mismatch");
        }
        std::vector<uint8_t> bytes(size);
        std::memcpy(bytes.data(), data, size);
        Entry entry;
        entry.type = type;
        entry.dims = std::move(dims);
        entry.tensor = tensor;
        entry.size = size;
        entry.bytes = std::move(bytes);
        entries_.push_back(std::move(entry));
        ++cursor_;
        return tensor;
    }

    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::string name_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<Entry> entries_;
    size_t cursor_ = 0;
};

}  // namespace engine::models::common

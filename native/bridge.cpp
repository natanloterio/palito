#include "bridge.h"
#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

struct li_handle {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_sampler * sampler = nullptr;
    llama_batch_ext * batch = nullptr;
    uint64_t type_mask = 0;
    int32_t next_position = 0;
    bool logits_ready = false;
    ~li_handle() {
        if (batch) llama_batch_ext_free(batch);
        if (sampler) llama_sampler_free(sampler);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

static thread_local char error_text[2048] = {};
static std::once_flag initialized;

template<typename F> static int guarded(F && action) noexcept {
    try {
        action();
        return 0;
    } catch (const std::exception & error) {
        std::snprintf(error_text, sizeof(error_text), "%s", error.what());
    } catch (...) {
        std::snprintf(error_text, sizeof(error_text), "unknown native exception");
    }
    return -1;
}

static void check(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}

static void log_message(ggml_log_level level, const char * message, void *) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(message, stderr);
}

const char * li_error(void) { return error_text; }
void li_close(li_handle * handle) { delete handle; }

int li_open(const char * path, uint32_t context, uint32_t batch, int32_t threads, li_handle ** out) {
    return guarded([&] {
        *out = nullptr;
        check(context > 0 && context <= INT32_MAX && batch > 0 && batch <= context && threads > 0,
              "invalid context, batch, or thread count");
        std::call_once(initialized, [] {
            llama_log_set(log_message, nullptr);
            ggml_backend_load_all();
            llama_backend_init();
        });
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(path, {true, nullptr}), gguf_free);
        check(metadata != nullptr, "cannot read GGUF metadata");
        check(gguf_get_version(metadata.get()) == 3, "v0.1 requires GGUF version 3");
        const auto key = gguf_find_key(metadata.get(), "general.architecture");
        check(key >= 0 && gguf_get_kv_type(metadata.get(), key) == GGUF_TYPE_STRING,
              "missing string general.architecture");
        check(std::strcmp(gguf_get_val_str(metadata.get(), key), "llama") == 0,
              "v0.1 supports only dense llama-architecture GGUF models");
        const auto experts = gguf_find_key(metadata.get(), "llama.expert_count");
        if (experts >= 0) {
            check(gguf_get_kv_type(metadata.get(), experts) == GGUF_TYPE_UINT32 &&
                  gguf_get_val_u32(metadata.get(), experts) == 0, "MoE is not supported in v0.1");
        }
        const auto split = gguf_find_key(metadata.get(), "split.count");
        if (split >= 0) {
            check(gguf_get_kv_type(metadata.get(), split) == GGUF_TYPE_UINT16 &&
                  gguf_get_val_u16(metadata.get(), split) == 1, "split GGUF models are not supported");
        }
        const auto train_context = gguf_find_key(metadata.get(), "llama.context_length");
        check(train_context >= 0 && gguf_get_kv_type(metadata.get(), train_context) == GGUF_TYPE_UINT32,
              "missing uint32 llama.context_length");
        check(context <= gguf_get_val_u32(metadata.get(), train_context), "context exceeds trained context");
        auto handle = std::make_unique<li_handle>();
        check(gguf_get_n_tensors(metadata.get()) > 0, "GGUF contains no weight tensors");
        for (int64_t i = 0; i < gguf_get_n_tensors(metadata.get()); ++i) {
            const auto type = gguf_get_tensor_type(metadata.get(), i);
            if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 &&
                type != GGML_TYPE_Q5_0 && type != GGML_TYPE_Q8_0 &&
                type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K) {
                throw std::runtime_error(std::string("unsupported tensor type ") + ggml_type_name(type) +
                                         " in " + gguf_get_tensor_name(metadata.get(), i) +
                                         "; no implicit conversion is allowed");
            }
            handle->type_mask |= uint64_t(1) << type;
        }
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        handle->model = llama_model_load_from_file(path, model_params);
        check(handle->model != nullptr, "llama.cpp model loading failed");
        check(llama_model_has_decoder(handle->model) && !llama_model_has_encoder(handle->model) &&
              !llama_model_is_recurrent(handle->model) && !llama_model_is_diffusion(handle->model),
              "model is not a supported autoregressive dense decoder");
        auto params = llama_context_default_params();
        params.n_ctx = context;
        params.n_batch = batch;
        params.n_ubatch = batch;
        params.n_seq_max = 1;
        params.n_threads = threads;
        params.n_threads_batch = threads;
        params.offload_kqv = false;
        params.op_offload = false;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.no_perf = true;
        handle->context = llama_init_from_model(handle->model, params);
        check(handle->context != nullptr, "llama.cpp context allocation failed");
        handle->sampler = llama_sampler_init_greedy();
        handle->batch = llama_batch_ext_init(handle->context);
        check(handle->sampler && handle->batch, "sampler or batch allocation failed");
        *out = handle.release();
    });
}

int li_get_info(li_handle * handle, li_info * out) {
    return guarded([&] {
        *out = {llama_model_size(handle->model), llama_model_n_params(handle->model), handle->type_mask,
                llama_n_ctx(handle->context), llama_n_batch(handle->context),
                uint32_t(llama_vocab_n_tokens(llama_model_get_vocab(handle->model)))};
    });
}

int li_tokenize(li_handle * handle, const char * text, int32_t length, int32_t * out, int32_t capacity, int32_t * count) {
    return guarded([&] {
        const auto result = llama_tokenize(llama_model_get_vocab(handle->model), text, length, out, capacity, true, true);
        check(result != INT32_MIN, "tokenization overflow");
        *count = result < 0 ? -result : result;
        check(capacity == 0 || result >= 0, "tokenization buffer is too small");
    });
}

int li_detokenize(li_handle * handle, const int32_t * tokens, int32_t count, char * out, int32_t capacity, int32_t * length) {
    return guarded([&] {
        const auto result = llama_detokenize(llama_model_get_vocab(handle->model), tokens, count, out, capacity, true, false);
        check(result != INT32_MIN, "detokenization overflow");
        *length = result < 0 ? -result : result;
        check(capacity == 0 || result >= 0, "detokenization buffer is too small");
    });
}

int li_reset(li_handle * handle) {
    return guarded([&] {
        llama_synchronize(handle->context);
        llama_memory_clear(llama_get_memory(handle->context), true);
        llama_sampler_reset(handle->sampler);
        handle->next_position = 0;
        handle->logits_ready = false;
    });
}

int li_eval(li_handle * handle, const int32_t * tokens, int32_t count, int32_t position) {
    return guarded([&] {
        check(count > 0 && uint32_t(count) <= llama_n_batch(handle->context), "invalid evaluation batch size");
        check(position == handle->next_position, "non-contiguous sequence position");
        check(position >= 0 && uint64_t(position) + count <= llama_n_ctx(handle->context), "evaluation exceeds context");
        handle->logits_ready = false;
        llama_batch_ext_clear(handle->batch);
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(handle->model));
        for (int32_t i = 0; i < count; ++i) {
            check(tokens[i] >= 0 && tokens[i] < vocab, "token outside vocabulary");
            const auto index = llama_batch_ext_add_token(handle->batch, 0, tokens[i]);
            check(index >= 0, "cannot add token to batch");
            const llama_pos pos = position + i;
            check(llama_batch_ext_set_pos(handle->batch, index, &pos), "cannot set token position");
        }
        check(llama_batch_ext_set_output_logits(handle->batch, count - 1, true), "cannot request logits");
        const int result = llama_process(handle->context, LLAMA_PROCESS_TYPE_DECODE, handle->batch);
        if (result != 0) {
            std::snprintf(error_text, sizeof(error_text), "llama_process returned %d; reset required", result);
            throw std::runtime_error(error_text);
        }
        llama_synchronize(handle->context);
        handle->next_position += count;
        handle->logits_ready = true;
    });
}

int li_sample_greedy(li_handle * handle, int32_t * token) {
    return guarded([&] {
        check(handle->logits_ready, "no valid logits; evaluate tokens before sampling");
        *token = llama_sampler_sample(handle->sampler, handle->context, -1);
        check(*token >= 0, "greedy sampler returned an invalid token");
    });
}

int li_is_eog(li_handle * handle, int32_t token) {
    return llama_vocab_is_eog(llama_model_get_vocab(handle->model), token);
}

int li_copy_logits(li_handle * handle, float * out, int32_t count) {
    return guarded([&] {
        check(handle->logits_ready, "no valid logits; evaluate first");
        check(out != nullptr && count == llama_vocab_n_tokens(llama_model_get_vocab(handle->model)),
              "invalid logits output buffer");
        const float * logits = llama_get_logits_ith(handle->context, -1);
        check(logits != nullptr, "missing reference logits");
        std::memcpy(out, logits, sizeof(float) * count);
    });
}

// Standalone external numerical oracle. Never linked into the Rust runtime.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Trace {
    FILE * file = nullptr;
    bool failed = false;
    size_t records = 0;
};

static bool trace_tensor(ggml_tensor * tensor, bool ask, void * data) {
    auto & trace = *static_cast<Trace *>(data);
    const char * name = ggml_get_name(tensor);
    const char * stages[] = {"attn_norm-", "Qcur-", "Kcur-", "Vcur-", "ffn_inp-",
        "ffn_norm-", "ffn_moe_logits-", "ffn_moe_topk-", "ffn_moe_weights_norm-",
        "ffn_moe_out-", "l_out-"};
    bool selected = false;
    for (const char * stage : stages) selected |= std::strncmp(name, stage, std::strlen(stage)) == 0;
    if (ask) return selected;
    if (!selected) return true;
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_I32) {
        trace.failed = true;
        return false;
    }
    std::vector<unsigned char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    bool heads = name[0] == 'Q' || name[0] == 'K' || name[0] == 'V';
    bool projection = heads && tensor->op == GGML_OP_MUL_MAT;
    std::string projected_name = projection
        ? std::string(1, name[0]) + "proj" + std::strchr(name, '-') : name;
    bool three_dimensional = (heads && !projection) || tensor->ne[2] > 1;
    int64_t rows = three_dimensional ? tensor->ne[1] : 1;
    int axis = three_dimensional ? 2 : 1;
    size_t offset = (tensor->ne[axis] - 1) * tensor->nb[axis];
    std::fprintf(trace.file, "{\"name\":\"%s\",\"shape\":[%lld,%lld,%lld,%lld],\"values\":[",
        projected_name.c_str(), (long long)tensor->ne[0], (long long)tensor->ne[1],
        (long long)tensor->ne[2], (long long)tensor->ne[3]);
    bool first = true;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t i = 0; i < tensor->ne[0]; ++i) {
            size_t index = offset + row * tensor->nb[1] + i * tensor->nb[0];
            if (index + sizeof(float) > bytes.size()) { trace.failed = true; return false; }
            if (!first) std::fputc(',', trace.file);
            first = false;
            if (tensor->type == GGML_TYPE_F32) {
                float value;
                std::memcpy(&value, bytes.data() + index, sizeof(value));
                if (!std::isfinite(value)) { trace.failed = true; return false; }
                std::fprintf(trace.file, "%.9g", value);
            } else {
                int32_t value;
                std::memcpy(&value, bytes.data() + index, sizeof(value));
                std::fprintf(trace.file, "%d", value);
            }
        }
    }
    std::fputs("]}\n", trace.file);
    trace.failed |= std::ferror(trace.file) != 0;
    ++trace.records;
    return !trace.failed;
}

static void check(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        check(argc >= 4 && argc <= 6,
              "usage: logits-oracle MODEL INPUT_IDS_TEXT GPU_LAYERS(-1=all,0=CPU) [NEW_TRACE_JSONL] [--f32-reference]");
        const char * trace_path = nullptr;
        bool require_f32 = false;
        for (int i = 4; i < argc; ++i) {
            if (std::strcmp(argv[i], "--f32-reference") == 0) {
                check(!require_f32, "duplicate --f32-reference");
                require_f32 = true;
            } else {
                check(trace_path == nullptr, "multiple trace paths");
                trace_path = argv[i];
            }
        }
        auto close_trace = [](FILE * file) { std::fclose(file); };
        std::unique_ptr<FILE, decltype(close_trace)> trace_file(
            trace_path ? std::fopen(trace_path, "wx") : nullptr, close_trace);
        check(!trace_path || trace_file != nullptr, "cannot create new trace");
        Trace trace{trace_file.get()};
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = std::stoi(argv[3]);
        const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        check(cpu != nullptr, "CPU backend unavailable");
        const auto marker = reinterpret_cast<const char * (*)()>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(cpu), "local_f32_oracle_id"));
        const bool f32_reference = mp.n_gpu_layers == 0 && marker
            && std::strcmp(marker(), "local-f32-activations-v1") == 0;
        check(!require_f32 || f32_reference,
              "--f32-reference requires the isolated patched CPU backend and GPU_LAYERS=0");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(model != nullptr, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_seq_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = mp.n_gpu_layers == 0
            ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = cp.op_offload = mp.n_gpu_layers != 0;
        if (trace.file) {
            cp.cb_eval = trace_tensor;
            cp.cb_eval_user_data = &trace;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), llama_free);
        check(context != nullptr, "context allocation failed");
        std::ifstream input(argv[2]);
        check(input.is_open(), "cannot open token input");
        std::vector<llama_token> ids;
        int id;
        const auto vocab = llama_model_get_vocab(model.get());
        while (input >> id) {
            check(id >= 0 && id < llama_vocab_n_tokens(vocab), "invalid token ID");
            ids.push_back(id);
        }
        check(input.eof() && !ids.empty() && ids.size() <= 1024, "invalid input tokens");
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(context.get()), llama_batch_ext_free);
        check(batch != nullptr, "batch allocation failed");
        for (size_t start = 0; start < ids.size(); start += 128) {
            llama_batch_ext_clear(batch.get());
            const auto count = std::min<size_t>(128, ids.size() - start);
            for (size_t i = 0; i < count; ++i) {
                const int index = llama_batch_ext_add_token(batch.get(), 0, ids[start + i]);
                check(index >= 0, "cannot add token");
                const llama_pos pos = start + i;
                check(llama_batch_ext_set_pos(batch.get(), index, &pos), "cannot set position");
            }
            check(llama_batch_ext_set_output_logits(batch.get(), count - 1, true),
                  "cannot request logits");
            check(llama_process(context.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                  "llama_process failed");
        }
        llama_synchronize(context.get());
        check(!trace.failed && (!trace.file || trace.records > 0), "tensor trace failed or empty");
        if (trace.file) check(std::fflush(trace.file) == 0, "cannot flush tensor trace");
        const float * logits = llama_get_logits_ith(context.get(), -1);
        check(logits != nullptr, "missing logits");
        const int n = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n; ++i) check(std::isfinite(logits[i]), "nonfinite logits");
        std::printf("{\"arithmetic\":\"%s\",\"logits\":[",
                    f32_reference ? "local-f32-activations-v1" : "backend-default");
        for (int i = 0; i < n; ++i) std::printf(i ? ",%.9g" : "%.9g", logits[i]);
        std::puts("]}");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "external logits oracle: %s\n", error.what());
        return 1;
    }
}

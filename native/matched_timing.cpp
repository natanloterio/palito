// Standalone timing counterpart to examples/cuda_matched_timing.rs.
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

static void check(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}

static std::vector<llama_token> ids_from(const std::string & line) {
    std::istringstream input(line);
    std::vector<llama_token> ids;
    llama_token id;
    while (input >> id) ids.push_back(id);
    check(input.eof(), "invalid token ID");
    return ids;
}

int main(int argc, char ** argv) {
    try {
        check(argc == 5, "usage: matched-timing MODEL FIXTURE WARMUPS RUNS");
        int warmups = std::stoi(argv[3]), runs = std::stoi(argv[4]);
        check(warmups >= 0 && warmups <= 20 && runs > 0 && runs <= 100, "invalid timing counts");
        std::ifstream input(argv[2]);
        std::string prompt_line, continuation_line, extra;
        check(bool(std::getline(input, prompt_line)) && bool(std::getline(input, continuation_line))
            && !std::getline(input, extra), "fixture needs exactly two lines");
        auto prompt = ids_from(prompt_line);
        const auto continuation = ids_from(continuation_line);
        check(!prompt.empty() && prompt.size() <= 512 && continuation.size() <= 31,
            "fixture exceeds timing shape limits");

        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
        }, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        check(model != nullptr, "model load failed");
        const int vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (const auto * ids : {static_cast<const std::vector<llama_token> *>(&prompt), &continuation})
            for (auto id : *ids) check(id >= 0 && id < vocab_size, "token out of vocabulary");
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_batch = cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = cp.op_offload = true;
        auto sp = llama_sampler_chain_default_params();
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
            llama_sampler_chain_init(sp), llama_sampler_free);
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        llama_sampler_seq_config sampler_config{0, sampler.get()};
        cp.samplers = &sampler_config;
        cp.n_samplers = 1;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), llama_free);
        check(context != nullptr && llama_n_ctx(context.get()) == 1024, "matched context unavailable");
        const auto step = [&](llama_token * tokens, int count) {
            check(llama_decode(context.get(), llama_batch_get_one(tokens, count)) == 0, "llama_decode failed");
            llama_synchronize(context.get());
            return llama_sampler_sample(sampler.get(), context.get(), -1);
        };
        using Clock = std::chrono::steady_clock;
        auto ns = [](Clock::duration d) {
            return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
        };
        for (bool greedy : {true, false}) {
            for (int i = 0; i < (greedy ? 1 : warmups + runs); ++i) {
                std::vector<llama_token> generated;
                generated.reserve(1 + continuation.size());
                const auto start = Clock::now();
                llama_memory_clear(llama_get_memory(context.get()), false);
                llama_sampler_reset(sampler.get());
                llama_synchronize(context.get());
                const auto prefill_start = Clock::now();
                generated.push_back(step(prompt.data(), static_cast<int>(prompt.size())));
                const auto prefill_end = Clock::now();
                for (auto forced : continuation) {
                    auto id = greedy ? generated.back() : forced;
                    generated.push_back(step(&id, 1));
                }
                const auto end = Clock::now();
                if (!greedy && i < warmups) continue;
                std::printf("{\"greedy\":%s,\"sample\":%d,\"prefill_ns\":%lld,\"decode_ns\":%lld,"
                            "\"request_ns\":%lld,\"output_ids\":[", greedy ? "true" : "false",
                            greedy ? 0 : i - warmups, ns(prefill_end - prefill_start),
                            ns(end - prefill_end), ns(end - start));
                for (size_t j = 0; j < generated.size(); ++j)
                    std::printf(j ? ",%d" : "%d", generated[j]);
                std::puts("]}");
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "external matched timing: %s\n", error.what());
        return 1;
    }
}

#include "bridge.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
static uint64_t elapsed(clock_type::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - start).count();
}
static void checked(int result) {
    if (result != 0) throw std::runtime_error(li_error());
}
static int positive(const char * value) {
    size_t end = 0;
    int result = std::stoi(value, &end);
    if (result <= 0 || value[end] != '\0') throw std::runtime_error("expected a positive integer");
    return result;
}

int main(int argc, char ** argv) {
    try {
        if (argc != 11) throw std::runtime_error(
            "usage: li-reference MODEL INPUT_FILE CONTEXT BATCH CHUNK THREADS TOKENS WARMUPS REPETITIONS {v1|v2-token-ids|v3-logits|v4-tokenize}");
        const int context = positive(argv[3]), batch = positive(argv[4]), chunk = positive(argv[5]);
        const int threads = positive(argv[6]), count = positive(argv[7]), warmups = positive(argv[8]);
        const int reps = positive(argv[9]);
        const std::string protocol = argv[10];
        if (protocol != "v1" && protocol != "v2-token-ids" && protocol != "v3-logits" && protocol != "v4-tokenize")
            throw std::runtime_error("unsupported benchmark protocol");
        if (chunk > batch || batch > context) throw std::runtime_error("require chunk <= batch <= context");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open prompt file");
        const std::string prompt{std::istreambuf_iterator<char>(input), {}};
        const auto load_start = clock_type::now();
        li_handle * raw = nullptr;
        checked(li_open(argv[1], context, batch, threads, &raw));
        std::unique_ptr<li_handle, decltype(&li_close)> handle(raw, li_close);
        const auto load_ns = elapsed(load_start);
        int32_t size = 0;
        std::vector<int32_t> tokens;
        if (protocol == "v1" || protocol == "v4-tokenize") {
            checked(li_tokenize(raw, prompt.data(), prompt.size(), nullptr, 0, &size));
            tokens.resize(size);
            checked(li_tokenize(raw, prompt.data(), prompt.size(), tokens.data(), size, &size));
        } else {
            std::istringstream ids(prompt);
            int32_t token;
            while (ids >> token) {
                if (token < 0 || tokens.size() >= static_cast<size_t>(context))
                    throw std::runtime_error("invalid token ID or input exceeds context");
                tokens.push_back(token);
            }
            if (!ids.eof()) throw std::runtime_error("invalid token ID file");
            size = static_cast<int32_t>(tokens.size());
        }
        if (protocol == "v4-tokenize") {
            int32_t length = 0;
            checked(li_detokenize(raw, tokens.data(), size, nullptr, 0, &length));
            std::vector<char> decoded(length);
            checked(li_detokenize(raw, tokens.data(), size, decoded.data(), length, &length));
            std::cout << "{\"prompt_token_ids\":[";
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << tokens[i];
            }
            std::cout << "],\"decoded_bytes\":[";
            for (int32_t i = 0; i < length; ++i) {
                if (i) std::cout << ',';
                std::cout << static_cast<unsigned>(static_cast<unsigned char>(decoded[i]));
            }
            std::cout << "]}\n";
            return 0;
        }
        if (size == 0 || uint64_t(size) + count - 1 > uint64_t(context))
            throw std::runtime_error("prompt plus evaluated generation exceeds context");
        if (protocol == "v3-logits") {
            for (int32_t pos = 0; pos < size; pos += chunk)
                checked(li_eval(raw, tokens.data() + pos, std::min(chunk, size - pos), pos));
            li_info info{};
            checked(li_get_info(raw, &info));
            std::vector<float> logits(info.vocab_tokens);
            checked(li_copy_logits(raw, logits.data(), static_cast<int32_t>(logits.size())));
            std::cout << "{\"logits\":[" << std::setprecision(9);
            for (size_t i = 0; i < logits.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << logits[i];
            }
            std::cout << "]}\n";
            return 0;
        }
        std::cout << "{\"schema_version\":1,\"runner\":\"cpp-reference\",\"load_ns\":" << load_ns
                  << ",\"prompt_tokens\":" << size << ",\"prompt_token_ids\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tokens[i];
        }
        std::cout << "],\"samples\":[";
        for (int rep = -warmups; rep < reps; ++rep) {
            checked(li_reset(raw));
            const auto start = clock_type::now();
            for (int32_t pos = 0; pos < size; pos += chunk)
                checked(li_eval(raw, tokens.data() + pos, std::min(chunk, size - pos), pos));
            const auto prefill_ns = elapsed(start);
            int32_t token = 0;
            checked(li_sample_greedy(raw, &token));
            const auto ttft_ns = elapsed(start);
            std::vector<int32_t> generated{token};
            const auto decode_start = clock_type::now();
            for (int i = 1; i < count; ++i) {
                checked(li_eval(raw, &token, 1, size + i - 1));
                checked(li_sample_greedy(raw, &token));
                generated.push_back(token);
            }
            const auto decode_ns = count > 1 ? elapsed(decode_start) : 0;
            if (rep < 0) continue;
            if (rep > 0) std::cout << ',';
            std::cout << "{\"prefill_ns\":" << prefill_ns << ",\"ttft_ns\":" << ttft_ns
                      << ",\"decode_ns\":" << decode_ns << ",\"decode_steps\":" << count - 1
                      << ",\"generated_tokens\":[";
            for (size_t i = 0; i < generated.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << generated[i];
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "li-reference: " << error.what() << '\n';
        return 1;
    }
}

// External-only graph tests: original packed weights and unquantized activations.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void test(ggml_type type, int layout, int threads) {
    constexpr int width = 256, rows = 5, tokens = 7, experts = 3, used = 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({8 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("context allocation failed");
    const bool indexed = layout >= 2;
    const int matrices = indexed ? experts : (layout == 1 ? 2 : 1);
    auto * w = ggml_new_tensor_3d(ctx.get(), type, width, rows, matrices);
    std::vector<float> original(width * rows * matrices), weights(original.size());
    for (size_t i = 0; i < original.size(); ++i)
        original[i] = std::sin(i * 0.137) * 0.7 + std::cos(i * 0.013) * 0.11;
    ggml_quantize_chunk(type, original.data(), w->data, 0, rows * matrices, width, nullptr);
    if (type == GGML_TYPE_F32) {
        std::memcpy(weights.data(), w->data, weights.size() * sizeof(float));
    } else {
        ggml_get_type_traits(type)->to_float(w->data, weights.data(), weights.size());
    }
    const int slots = layout == 3 ? used : 1;
    const int batches = layout == 1 ? 4 : 1;
    auto * storage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width * 2,
                                       indexed ? slots : tokens, indexed ? tokens : batches);
    auto * x = ggml_view_3d(ctx.get(), storage, width, storage->ne[1], storage->ne[2],
                            storage->nb[1], storage->nb[2], sizeof(float));
    for (int64_t i = 0; i < ggml_nelements(storage); ++i)
        static_cast<float *>(storage->data)[i] = std::sin(i * 0.071) + std::cos(i * 0.017) * 0.17;
    ggml_tensor * ids = nullptr;
    ggml_tensor * y;
    if (indexed) {
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
        for (int t = 0; t < tokens; ++t)
            for (int s = 0; s < used; ++s)
                static_cast<int32_t *>(ids->data)[t * used + s] = (t + s * 2) % experts;
        y = ggml_mul_mat_id(ctx.get(), w, x, ids);
    } else {
        y = ggml_mul_mat(ctx.get(), w, x);
    }
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, threads) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("graph compute failed");
    for (int b = 0; b < (indexed ? tokens : batches); ++b) {
        for (int s = 0; s < (indexed ? used : tokens); ++s) {
            const int matrix = indexed ? static_cast<int32_t *>(ids->data)[b * used + s]
                                       : b / (batches / matrices);
            const auto * input = reinterpret_cast<const float *>(
                static_cast<const char *>(x->data) + b * x->nb[2]
                + (indexed ? s % slots : s) * x->nb[1]);
            for (int r = 0; r < rows; ++r) {
                double sum = 0;
                for (int k = 0; k < width; ++k)
                    sum += double(weights[(matrix * rows + r) * width + k]) * input[k];
                float actual;
                std::memcpy(&actual, static_cast<const char *>(y->data)
                    + r * y->nb[0] + s * y->nb[1] + b * y->nb[2], sizeof(actual));
                const float expected = float(sum);
                const float tolerance = type == GGML_TYPE_F32 ? 0.00002f : 0.0f;
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                    std::fprintf(stderr, "%s layout=%d threads=%d: actual=%.9g expected=%.9g\n",
                                 ggml_type_name(type), layout, threads, actual, expected);
                    throw std::runtime_error("F32 activation matmul mismatch");
                }
            }
        }
    }
}

int main() {
    try {
        int cases = 0;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q5_0,
                          GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K})
            for (int layout = 0; layout < 4; ++layout)
                for (int threads : {1, 3}) { test(type, layout, threads); ++cases; }
        std::printf("{\"passed\":%d,\"formats\":6,\"layouts\":4,\"thread_counts\":[1,3]}\n", cases);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "F32 oracle graph tests: %s\n", error.what());
        return 1;
    }
}

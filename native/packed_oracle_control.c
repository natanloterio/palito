// Negative control: identical external graph/configuration, original packed products.
#include "ggml-cpu-impl.h"

const char * local_f32_oracle_id(void) {
    return "local-packed-control-v1";
}

bool local_f32_matmul(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    (void) params;
    (void) dst;
    return false;
}

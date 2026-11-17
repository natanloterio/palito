// Correctness-only extension to an isolated pinned GGML CPU build, never Rust.
#include "ggml-cpu-impl.h"
#include <stdint.h>
#include <stdlib.h>

const char * local_f32_oracle_id(void) {
    return "local-f32-activations-v1";
}

bool local_f32_matmul(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0];
    const struct ggml_tensor * x = dst->src[1];
    if (w->type == GGML_TYPE_F32 || x->type != GGML_TYPE_F32) {
        return false;
    }
    const bool indexed = dst->op == GGML_OP_MUL_MAT_ID;
    GGML_ASSERT(indexed || dst->op == GGML_OP_MUL_MAT);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(x->nb[0] == sizeof(float) && x->ne[0] == w->ne[0]);
    GGML_ASSERT(w->ne[1] == dst->ne[0] && w->ne[0] > 0);
    GGML_ASSERT((uint64_t) w->ne[0] <= SIZE_MAX / sizeof(float));
    const struct ggml_type_traits * traits = ggml_get_type_traits(w->type);
    GGML_ASSERT(traits->to_float != NULL);
    float * row = malloc((size_t) w->ne[0] * sizeof(float));
    GGML_ASSERT(row != NULL);
    const struct ggml_tensor * ids = indexed ? dst->src[2] : NULL;
    if (indexed) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
        GGML_ASSERT(ids->ne[0] == dst->ne[1] && ids->ne[1] == dst->ne[2]);
        GGML_ASSERT(dst->ne[3] == 1 && x->ne[2] == dst->ne[2]);
        GGML_ASSERT(x->ne[1] == 1 || x->ne[1] == dst->ne[1]);
    } else {
        GGML_ASSERT(dst->ne[2] % w->ne[2] == 0 && dst->ne[3] % w->ne[3] == 0);
    }
    for (int64_t element = params->ith; element < ggml_nelements(dst); element += params->nth) {
        int64_t row_index = element % dst->ne[0];
        int64_t remaining = element / dst->ne[0];
        int64_t i1 = remaining % dst->ne[1];
        remaining /= dst->ne[1];
        int64_t i2 = remaining % dst->ne[2];
        int64_t i3 = remaining / dst->ne[2];
        size_t w_offset, x_offset;
        if (indexed) {
            int32_t expert;
            memcpy(&expert, (const char *) ids->data + i1 * ids->nb[0] + i2 * ids->nb[1], sizeof(expert));
            GGML_ASSERT(expert >= 0 && expert < w->ne[2]);
            w_offset = expert * w->nb[2] + row_index * w->nb[1];
            x_offset = (i1 % x->ne[1]) * x->nb[1] + i2 * x->nb[2];
        } else {
            w_offset = row_index * w->nb[1]
                + (i2 / (dst->ne[2] / w->ne[2])) * w->nb[2]
                + (i3 / (dst->ne[3] / w->ne[3])) * w->nb[3];
            x_offset = i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3];
        }
        traits->to_float((const char *) w->data + w_offset, row, w->ne[0]);
        const float * input = (const float *) ((const char *) x->data + x_offset);
        double sum = 0.0;
        for (int64_t k = 0; k < w->ne[0]; ++k) {
            sum += (double) row[k] * (double) input[k];
        }
        float result = (float) sum;
        memcpy((char *) dst->data + row_index * dst->nb[0] + i1 * dst->nb[1]
            + i2 * dst->nb[2] + i3 * dst->nb[3], &result, sizeof(result));
    }
    free(row);
    return true;
}

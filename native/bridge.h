#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct li_handle li_handle;
typedef struct {
    uint64_t weight_bytes;
    uint64_t parameters;
    uint64_t tensor_type_mask;
    uint32_t context_tokens;
    uint32_t batch_tokens;
    uint32_t vocab_tokens;
} li_info;

// Buffers and handles stay owned by the caller. Errors are thread-local.
const char * li_error(void);
int li_open(const char * path, uint32_t context, uint32_t batch, int32_t threads, li_handle ** out);
void li_close(li_handle * handle);
int li_get_info(li_handle * handle, li_info * out);
int li_tokenize(li_handle * handle, const char * text, int32_t length, int32_t * out, int32_t capacity, int32_t * count);
int li_detokenize(li_handle * handle, const int32_t * tokens, int32_t count, char * out, int32_t capacity, int32_t * length);
int li_reset(li_handle * handle);
int li_eval(li_handle * handle, const int32_t * tokens, int32_t count, int32_t position);
int li_sample_greedy(li_handle * handle, int32_t * token);
int li_copy_logits(li_handle * handle, float * out, int32_t count);
int li_is_eog(li_handle * handle, int32_t token);

#ifdef __cplusplus
}
#endif

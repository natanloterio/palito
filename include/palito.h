/*
 * Palito C API: fast local LLM inference for GGUF models.
 *
 * Link with -lpalito (libpalito.so). With CUDA, libnvrtc.so.13 must be on the
 * library path at run time (the Linux bundle ships it in lib/).
 *
 * Conventions:
 *  - Initialize option structs with palito_options_init/palito_generation_init
 *    before setting fields, so new fields get their defaults.
 *  - Functions returning int return 0 on success and -1 on error. On error,
 *    when `error` is not NULL, *error receives a message to release with
 *    palito_string_free.
 *  - Strings are UTF-8. Text passed to callbacks is not NUL-terminated.
 *  - A palito_engine must not be used from two threads at the same time.
 *    Separate engines are independent.
 *  - ABI: struct layouts never change; new options come as new functions.
 */
#ifndef PALITO_H
#define PALITO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct palito_engine palito_engine;

enum {
    PALITO_DEVICE_DEFAULT = 0, /* CUDA when built with it, else CPU */
    PALITO_DEVICE_CPU = 1,
    PALITO_DEVICE_CUDA = 2
};

enum {
    PALITO_FINISH_STOP = 0,         /* the model ended its turn */
    PALITO_FINISH_LENGTH = 1,       /* max_tokens reached */
    PALITO_FINISH_CONTEXT_FULL = 2, /* no context left */
    PALITO_FINISH_CANCELLED = 3     /* the callback asked to stop */
};

typedef struct {
    int device;                      /* PALITO_DEVICE_* */
    uint32_t context_tokens;         /* 0: model's trained context, fitted to GPU memory */
    uint32_t threads;                /* 0: 4 CPU threads */
    uint32_t gpu_device;             /* CUDA device index */
    uint64_t gpu_memory_limit_bytes; /* 0: free GPU memory minus 2 GiB */
    const char *kernel_cache_dir;    /* NULL: $PALITO_CUDA_KERNEL_CACHE or ~/.cache/palito/kernels */
} palito_options;

typedef struct {
    const char *role;    /* "system", "user" or "assistant" */
    const char *content;
} palito_message;

typedef struct {
    uint32_t max_tokens;      /* 0: until end of turn or a full context */
    int enable_thinking;      /* nonzero: reasoning on (Qwen3-style templates) */
    float temperature;        /* 0: greedy (default) */
    float top_p;              /* nucleus mass in (0, 1]; 1: off */
    uint32_t top_k;           /* 0: off */
    int64_t seed;             /* negative: a different seed every call */
    const char *const *stop;  /* stop sequences; generation ends at the first, */
    size_t stop_count;        /* which is not included in the text (max 16) */
} palito_generation;

typedef struct {
    char *text;               /* full generated text; free with palito_result_free */
    int finish_reason;        /* PALITO_FINISH_* */
    uint32_t prompt_tokens;
    uint32_t generated_tokens;
    double ttft_ms;           /* time to first token (0 when cancelled) */
    double tokens_per_second; /* decode speed after the first token */
} palito_result;

/*
 * Streaming callback: receives `length` bytes of new text (never splitting a
 * UTF-8 character). Return 0 to continue, nonzero to stop generation.
 */
typedef int (*palito_text_callback)(const char *text, size_t length, void *user_data);

const char *palito_version(void);

void palito_options_init(palito_options *options);
void palito_generation_init(palito_generation *generation);

/* Opens a GGUF model; `options` may be NULL. Returns NULL on error. */
palito_engine *palito_open(const char *model_path, const palito_options *options, char **error);
void palito_close(palito_engine *engine);

/* Context tokens actually opened (may be below the requested context). */
uint32_t palito_context_tokens(const palito_engine *engine);
/* "cpu", "int8-fast", "f32-tiled" or "f32"; static string. */
const char *palito_profile(const palito_engine *engine);

/*
 * Generates the assistant reply to `messages` using the model's chat template.
 * The call is stateless: pass the whole conversation each time. `generation`,
 * `callback` and `result` may be NULL.
 */
int palito_chat(palito_engine *engine,
                const palito_message *messages, size_t message_count,
                const palito_generation *generation,
                palito_text_callback callback, void *user_data,
                palito_result *result, char **error);

/* Continues `prompt` without a chat template. Arguments as in palito_chat. */
int palito_complete(palito_engine *engine, const char *prompt,
                    const palito_generation *generation,
                    palito_text_callback callback, void *user_data,
                    palito_result *result, char **error);

/*
 * palito_chat / palito_complete whose output is JSON matching `json_schema`
 * (JSON Schema text; "{\"type\":\"object\"}" for any object; NULL:
 * unconstrained). Reasoning is turned off. A schema with unsupported keywords
 * fails before generation. Output cut by max_tokens may be incomplete JSON
 * (finish_reason PALITO_FINISH_LENGTH).
 */
int palito_chat_json(palito_engine *engine,
                     const palito_message *messages, size_t message_count,
                     const palito_generation *generation, const char *json_schema,
                     palito_text_callback callback, void *user_data,
                     palito_result *result, char **error);
int palito_complete_json(palito_engine *engine, const char *prompt,
                         const palito_generation *generation, const char *json_schema,
                         palito_text_callback callback, void *user_data,
                         palito_result *result, char **error);

void palito_result_free(palito_result *result);
void palito_string_free(char *string);

#ifdef __cplusplus
}
#endif

#endif /* PALITO_H */

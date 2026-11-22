// Run-time loading of libpalito (dlopen), so building needs no link flags.
#ifndef PALITO_GO_SHIM_H
#define PALITO_GO_SHIM_H
#include <stdint.h>
#include "palito.h"

// Loads the library once; returns NULL on success or a static error message.
const char *palito_go_load(const char *path);

const char *palito_go_version(void);
void palito_go_options_init(palito_options *options);
void palito_go_generation_init(palito_generation *generation);
palito_engine *palito_go_open(const char *path, const palito_options *options, char **error);
void palito_go_close(palito_engine *engine);
uint32_t palito_go_context_tokens(const palito_engine *engine);
const char *palito_go_profile(const palito_engine *engine);
// `handle` is a cgo.Handle passed back to the Go text callback.
int palito_go_chat(palito_engine *engine, const palito_message *messages, size_t count,
                   const palito_generation *generation, uintptr_t handle,
                   palito_result *result, char **error);
int palito_go_complete(palito_engine *engine, const char *prompt,
                       const palito_generation *generation, uintptr_t handle,
                       palito_result *result, char **error);
void palito_go_result_free(palito_result *result);
void palito_go_string_free(char *string);
#endif

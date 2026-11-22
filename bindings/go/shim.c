#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include "shim.h"

// Implemented in Go (callback.go).
extern int palitoGoText(char *text, size_t length, uintptr_t handle);

static struct {
    const char *(*version)(void);
    void (*options_init)(palito_options *);
    void (*generation_init)(palito_generation *);
    palito_engine *(*open)(const char *, const palito_options *, char **);
    void (*close)(palito_engine *);
    uint32_t (*context_tokens)(const palito_engine *);
    const char *(*profile)(const palito_engine *);
    int (*chat)(palito_engine *, const palito_message *, size_t, const palito_generation *,
                palito_text_callback, void *, palito_result *, char **);
    int (*complete)(palito_engine *, const char *, const palito_generation *,
                    palito_text_callback, void *, palito_result *, char **);
    void (*result_free)(palito_result *);
    void (*string_free)(char *);
} api;

static pthread_mutex_t load_lock = PTHREAD_MUTEX_INITIALIZER;
static int loaded;

#define BIND(field, name)                                   \
    do {                                                    \
        *(void **)(&api.field) = dlsym(library, name);      \
        if (!api.field) return "libpalito is missing " name; \
    } while (0)

static const char *bind(void *library) {
    BIND(version, "palito_version");
    BIND(options_init, "palito_options_init");
    BIND(generation_init, "palito_generation_init");
    BIND(open, "palito_open");
    BIND(close, "palito_close");
    BIND(context_tokens, "palito_context_tokens");
    BIND(profile, "palito_profile");
    BIND(chat, "palito_chat");
    BIND(complete, "palito_complete");
    BIND(result_free, "palito_result_free");
    BIND(string_free, "palito_string_free");
    return NULL;
}

const char *palito_go_load(const char *path) {
    const char *failure = NULL;
    pthread_mutex_lock(&load_lock);
    if (!loaded) {
        void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            failure = dlerror();
        } else if ((failure = bind(library)) == NULL) {
            loaded = 1;
        }
    }
    pthread_mutex_unlock(&load_lock);
    return failure;
}

static int forward(const char *text, size_t length, void *user_data) {
    return palitoGoText((char *)text, length, (uintptr_t)user_data);
}

const char *palito_go_version(void) { return api.version(); }
void palito_go_options_init(palito_options *o) { api.options_init(o); }
void palito_go_generation_init(palito_generation *g) { api.generation_init(g); }
palito_engine *palito_go_open(const char *p, const palito_options *o, char **e) { return api.open(p, o, e); }
void palito_go_close(palito_engine *e) { api.close(e); }
uint32_t palito_go_context_tokens(const palito_engine *e) { return api.context_tokens(e); }
const char *palito_go_profile(const palito_engine *e) { return api.profile(e); }
void palito_go_result_free(palito_result *r) { api.result_free(r); }
void palito_go_string_free(char *s) { api.string_free(s); }

int palito_go_chat(palito_engine *engine, const palito_message *messages, size_t count,
                   const palito_generation *generation, uintptr_t handle,
                   palito_result *result, char **error) {
    return api.chat(engine, messages, count, generation, forward, (void *)handle, result, error);
}

int palito_go_complete(palito_engine *engine, const char *prompt,
                       const palito_generation *generation, uintptr_t handle,
                       palito_result *result, char **error) {
    return api.complete(engine, prompt, generation, forward, (void *)handle, result, error);
}

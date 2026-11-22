/*
 * Streams a chat reply through the Palito C API.
 *
 *   cc examples/c/chat.c -Iinclude -Ltarget/release -lpalito -o target/palito-chat-example
 *   LD_LIBRARY_PATH=target/release:<nvrtc lib dir> target/palito-chat-example model.gguf "Hello!"
 */
#include <palito.h>
#include <stdio.h>

static int print_text(const char *text, size_t length, void *user_data) {
    (void)user_data;
    fwrite(text, 1, length, stdout);
    fflush(stdout);
    return 0; /* nonzero stops generation */
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.gguf \"message\"\n", argv[0]);
        return 2;
    }
    char *error = NULL;
    palito_engine *engine = palito_open(argv[1], NULL, &error);
    if (!engine) {
        fprintf(stderr, "palito_open: %s\n", error);
        palito_string_free(error);
        return 1;
    }
    fprintf(stderr, "palito %s, %s, context %u tokens\n", palito_version(),
            palito_profile(engine), palito_context_tokens(engine));

    palito_message messages[] = {
        {"system", "You are a concise assistant."},
        {"user", argv[2]},
    };
    palito_generation generation;
    palito_generation_init(&generation);
    generation.enable_thinking = 0;

    palito_result result;
    int status = palito_chat(engine, messages, 2, &generation, print_text, NULL, &result, &error);
    if (status != 0) {
        fprintf(stderr, "\npalito_chat: %s\n", error);
        palito_string_free(error);
    } else {
        fprintf(stderr, "\n[%u tokens, first token %.0f ms, %.1f tok/s, finish %d]\n",
                result.generated_tokens, result.ttft_ms, result.tokens_per_second,
                result.finish_reason);
        palito_result_free(&result);
    }
    palito_close(engine);
    return status == 0 ? 0 : 1;
}

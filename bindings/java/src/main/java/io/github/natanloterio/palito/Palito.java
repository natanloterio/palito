package io.github.natanloterio.palito;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

/**
 * Java binding for Palito: fast local LLM inference for GGUF models.
 *
 * <pre>{@code
 * try (Palito llm = Palito.open(Path.of("model.gguf"), new Options())) {
 *     Completion reply = llm.chat(List.of(Message.user("Hello!")),
 *             new Generation().onText(text -> { System.out.print(text); return true; }));
 * }
 * }</pre>
 *
 * <p>Uses the Foreign Function &amp; Memory API (Java 22+); run with
 * {@code --enable-native-access=ALL-UNNAMED}. libpalito is loaded from
 * {@code $PALITO_LIBRARY}, then the installed {@code palito} command
 * ({@code <bundle>/lib}), then the system library path. Requests run one at a
 * time per engine; concurrent calls wait.
 */
public final class Palito implements AutoCloseable {

    private static final StructLayout OPTIONS = MemoryLayout.structLayout(
            JAVA_INT.withName("device"),
            JAVA_INT.withName("context_tokens"),
            JAVA_INT.withName("threads"),
            JAVA_INT.withName("gpu_device"),
            JAVA_LONG.withName("gpu_memory_limit_bytes"),
            ADDRESS.withName("kernel_cache_dir"));
    private static final StructLayout MESSAGE = MemoryLayout.structLayout(
            ADDRESS.withName("role"), ADDRESS.withName("content"));
    private static final StructLayout GENERATION = MemoryLayout.structLayout(
            JAVA_INT.withName("max_tokens"),
            JAVA_INT.withName("enable_thinking"),
            JAVA_FLOAT.withName("temperature"),
            JAVA_FLOAT.withName("top_p"),
            JAVA_INT.withName("top_k"),
            MemoryLayout.paddingLayout(4),
            JAVA_LONG.withName("seed"),
            ADDRESS.withName("stop"),
            JAVA_LONG.withName("stop_count"),
            JAVA_FLOAT.withName("min_p"),
            JAVA_FLOAT.withName("repeat_penalty"),
            JAVA_INT.withName("repeat_last_n"),
            JAVA_FLOAT.withName("presence_penalty"),
            JAVA_FLOAT.withName("frequency_penalty"),
            MemoryLayout.paddingLayout(4),
            ADDRESS.withName("logit_bias"),
            JAVA_LONG.withName("logit_bias_count"));
    private static final StructLayout LOGIT_BIAS = MemoryLayout.structLayout(
            JAVA_INT.withName("token"), JAVA_FLOAT.withName("bias"));
    private static final StructLayout RESULT = MemoryLayout.structLayout(
            ADDRESS.withName("text"),
            JAVA_INT.withName("finish_reason"),
            JAVA_INT.withName("prompt_tokens"),
            JAVA_INT.withName("generated_tokens"),
            MemoryLayout.paddingLayout(4),
            JAVA_DOUBLE.withName("ttft_ms"),
            JAVA_DOUBLE.withName("tokens_per_second"));
    private static final FunctionDescriptor CALLBACK =
            FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS);

    private static volatile Native api;

    private final ReentrantLock lock = new ReentrantLock();
    private MemorySegment engine;

    private Palito(MemorySegment engine) {
        this.engine = engine;
    }

    /** Opens a GGUF model; {@code options} may be null. */
    public static Palito open(Path model, Options options) {
        Native n = api();
        Options o = options == null ? new Options() : options;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment raw = arena.allocate(OPTIONS);
            n.optionsInit.invokeExact(raw);
            raw.set(JAVA_INT, 0, o.device.ordinal());
            raw.set(JAVA_INT, 4, o.contextTokens);
            raw.set(JAVA_INT, 8, o.threads);
            raw.set(JAVA_INT, 12, o.gpuDevice);
            raw.set(JAVA_LONG, 16, o.gpuMemoryLimitBytes);
            if (o.kernelCacheDir != null) {
                raw.set(ADDRESS, 24, arena.allocateFrom(o.kernelCacheDir.toString()));
            }
            MemorySegment error = arena.allocate(ADDRESS);
            MemorySegment handle = (MemorySegment) n.open.invokeExact(
                    arena.allocateFrom(model.toString()), raw, error);
            if (handle.equals(MemorySegment.NULL)) {
                throw new PalitoException(takeError(error));
            }
            return new Palito(handle);
        } catch (PalitoException e) {
            throw e;
        } catch (Throwable e) {
            throw new PalitoException("palito_open failed", e);
        }
    }

    public static String version() {
        try {
            return string((MemorySegment) api().version.invokeExact());
        } catch (Throwable e) {
            throw new PalitoException("palito_version failed", e);
        }
    }

    /** Context tokens actually opened. */
    public int contextTokens() {
        return locked(() -> (int) api().contextTokens.invokeExact(handle()));
    }

    /** "cpu", "int8-fast", "f32-tiled" or "f32". */
    public String profile() {
        return locked(() -> string((MemorySegment) api().profile.invokeExact(handle())));
    }

    /** Replies to {@code messages} with the model's chat template. */
    public Completion chat(List<Message> messages, Generation generation) {
        return generate(generation, (arena, gen, callback, result, error) -> {
            MemorySegment rows = arena.allocate(MESSAGE, Math.max(messages.size(), 1));
            for (int i = 0; i < messages.size(); i++) {
                long base = i * MESSAGE.byteSize();
                rows.set(ADDRESS, base, arena.allocateFrom(messages.get(i).role()));
                rows.set(ADDRESS, base + 8, arena.allocateFrom(messages.get(i).content()));
            }
            return (int) api().chat.invokeExact(handle(), rows, (long) messages.size(), gen,
                    callback, MemorySegment.NULL, result, error);
        });
    }

    /** Continues {@code prompt} without a chat template. */
    public Completion complete(String prompt, Generation generation) {
        return generate(generation, (arena, gen, callback, result, error) ->
                (int) api().complete.invokeExact(handle(), arena.allocateFrom(prompt), gen,
                        callback, MemorySegment.NULL, result, error));
    }

    @FunctionalInterface
    private interface Call {
        int invoke(Arena arena, MemorySegment generation, MemorySegment callback,
                   MemorySegment result, MemorySegment error) throws Throwable;
    }

    private Completion generate(Generation generation, Call call) {
        Generation g = generation == null ? new Generation() : generation;
        return locked(() -> {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment gen = arena.allocate(GENERATION);
                api().generationInit.invokeExact(gen);
                gen.set(JAVA_INT, 0, g.maxTokens);
                gen.set(JAVA_INT, 4, g.thinking ? 1 : 0);
                gen.set(JAVA_FLOAT, 8, g.temperature);
                gen.set(JAVA_FLOAT, 12, g.topP);
                gen.set(JAVA_INT, 16, g.topK);
                gen.set(JAVA_LONG, 24, g.seed.isPresent() ? g.seed.getAsLong() & Long.MAX_VALUE : -1L);
                if (!g.stop.isEmpty()) {
                    MemorySegment stops = arena.allocate(ADDRESS, g.stop.size());
                    for (int i = 0; i < g.stop.size(); i++) {
                        stops.setAtIndex(ADDRESS, i, arena.allocateFrom(g.stop.get(i)));
                    }
                    gen.set(ADDRESS, 32, stops);
                    gen.set(JAVA_LONG, 40, g.stop.size());
                }
                gen.set(JAVA_FLOAT, 48, g.minP);
                gen.set(JAVA_FLOAT, 52, g.repeatPenalty);
                gen.set(JAVA_INT, 56, g.repeatLastN);
                gen.set(JAVA_FLOAT, 60, g.presencePenalty);
                gen.set(JAVA_FLOAT, 64, g.frequencyPenalty);
                if (!g.logitBias.isEmpty()) {
                    MemorySegment biases = arena.allocate(LOGIT_BIAS, g.logitBias.size());
                    long offset = 0;
                    for (var entry : g.logitBias.entrySet()) {
                        biases.set(JAVA_INT, offset, entry.getKey());
                        biases.set(JAVA_FLOAT, offset + 4, entry.getValue());
                        offset += LOGIT_BIAS.byteSize();
                    }
                    gen.set(ADDRESS, 72, biases);
                    gen.set(JAVA_LONG, 80, g.logitBias.size());
                }
                TextSink sink = new TextSink(g.onText);
                MemorySegment callback = Linker.nativeLinker().upcallStub(
                        SINK.bindTo(sink), CALLBACK, arena);
                MemorySegment result = arena.allocate(RESULT);
                MemorySegment error = arena.allocate(ADDRESS);
                if (call.invoke(arena, gen, callback, result, error) != 0) {
                    throw new PalitoException(takeError(error));
                }
                try {
                    if (sink.failure != null) {
                        throw sink.failure;
                    }
                    return new Completion(
                            string(result.get(ADDRESS, 0)),
                            Completion.FinishReason.values()[result.get(JAVA_INT, 8)],
                            result.get(JAVA_INT, 12),
                            result.get(JAVA_INT, 16),
                            Duration.ofNanos((long) (result.get(JAVA_DOUBLE, 24) * 1e6)),
                            result.get(JAVA_DOUBLE, 32));
                } finally {
                    api().resultFree.invokeExact(result);
                }
            }
        });
    }

    /** Forwards streamed text; exceptions must not cross into native code. */
    private static final class TextSink {
        final Predicate<String> onText;
        RuntimeException failure;

        TextSink(Predicate<String> onText) {
            this.onText = onText;
        }

        int accept(MemorySegment text, long length, MemorySegment userData) {
            if (onText == null || failure != null) {
                return failure == null ? 0 : 1;
            }
            try {
                byte[] bytes = text.reinterpret(length).toArray(JAVA_BYTE);
                return onText.test(new String(bytes, StandardCharsets.UTF_8)) ? 0 : 1;
            } catch (RuntimeException e) {
                failure = e;
                return 1;
            } catch (Throwable e) {
                failure = new PalitoException("text callback failed", e);
                return 1;
            }
        }
    }

    private static final MethodHandle SINK;

    static {
        try {
            SINK = MethodHandles.lookup().findVirtual(TextSink.class, "accept",
                    MethodType.methodType(int.class, MemorySegment.class, long.class,
                            MemorySegment.class));
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    @FunctionalInterface
    private interface Body<T> {
        T run() throws Throwable;
    }

    private <T> T locked(Body<T> body) {
        lock.lock();
        try {
            return body.run();
        } catch (RuntimeException e) {
            throw e;
        } catch (Throwable e) {
            throw new PalitoException("native call failed", e);
        } finally {
            lock.unlock();
        }
    }

    private MemorySegment handle() {
        if (engine == null) {
            throw new PalitoException("engine is closed");
        }
        return engine;
    }

    /** Releases the model; waits for a running request. */
    @Override
    public void close() {
        lock.lock();
        try {
            if (engine != null) {
                api().close.invokeExact(engine);
                engine = null;
            }
        } catch (Throwable e) {
            throw new PalitoException("palito_close failed", e);
        } finally {
            lock.unlock();
        }
    }

    private static String takeError(MemorySegment slot) throws Throwable {
        MemorySegment message = slot.get(ADDRESS, 0);
        if (message.equals(MemorySegment.NULL)) {
            return "unknown error";
        }
        String text = string(message);
        api().stringFree.invokeExact(message);
        return text;
    }

    private static String string(MemorySegment pointer) {
        return pointer.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
    }

    /** Where libpalito is loaded from. */
    public static String libraryPath() {
        String explicit = System.getenv("PALITO_LIBRARY");
        if (explicit != null && !explicit.isEmpty()) {
            return explicit;
        }
        String name = System.mapLibraryName("palito");
        String path = System.getenv("PATH");
        if (path != null) {
            for (String dir : path.split(java.io.File.pathSeparator)) {
                Path command = Path.of(dir, "palito");
                try {
                    if (Files.isExecutable(command)) {
                        Path bundled = command.toRealPath().getParent().resolve("lib").resolve(name);
                        if (Files.isRegularFile(bundled)) {
                            return bundled.toString();
                        }
                    }
                } catch (java.io.IOException | RuntimeException ignored) {
                    // not here
                }
            }
        }
        return name;
    }

    private static Native api() {
        Native n = api;
        if (n == null) {
            synchronized (Palito.class) {
                if (api == null) {
                    api = new Native(libraryPath());
                }
                n = api;
            }
        }
        return n;
    }

    private static final class Native {
        final MethodHandle version, optionsInit, generationInit, open, close, contextTokens,
                profile, chat, complete, resultFree, stringFree;

        Native(String path) {
            SymbolLookup library;
            try {
                library = path.contains("/") || path.contains("\\")
                        ? SymbolLookup.libraryLookup(Path.of(path), Arena.global())
                        : SymbolLookup.libraryLookup(path, Arena.global());
            } catch (IllegalArgumentException e) {
                throw new PalitoException("cannot load " + path
                        + "; install Palito or set PALITO_LIBRARY", e);
            }
            Linker linker = Linker.nativeLinker();
            java.util.function.BiFunction<String, FunctionDescriptor, MethodHandle> bind =
                    (name, descriptor) -> linker.downcallHandle(library.find(name).orElseThrow(
                            () -> new PalitoException("libpalito is missing " + name)), descriptor);
            version = bind.apply("palito_version", FunctionDescriptor.of(ADDRESS));
            optionsInit = bind.apply("palito_options_init", FunctionDescriptor.ofVoid(ADDRESS));
            generationInit = bind.apply("palito_generation_init", FunctionDescriptor.ofVoid(ADDRESS));
            open = bind.apply("palito_open", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            close = bind.apply("palito_close", FunctionDescriptor.ofVoid(ADDRESS));
            contextTokens = bind.apply("palito_context_tokens", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            profile = bind.apply("palito_profile", FunctionDescriptor.of(ADDRESS, ADDRESS));
            chat = bind.apply("palito_chat", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS,
                    JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            complete = bind.apply("palito_complete", FunctionDescriptor.of(JAVA_INT, ADDRESS,
                    ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            resultFree = bind.apply("palito_result_free", FunctionDescriptor.ofVoid(ADDRESS));
            stringFree = bind.apply("palito_string_free", FunctionDescriptor.ofVoid(ADDRESS));
        }
    }
}

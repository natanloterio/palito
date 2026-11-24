using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Palito;

/// <summary>
/// A GGUF model opened once; <see cref="Chat"/> and <see cref="Complete"/> run one
/// request at a time (concurrent calls wait). libpalito is loaded from
/// $PALITO_LIBRARY, then the installed <c>palito</c> command (&lt;bundle&gt;/lib),
/// then the system loader.
/// </summary>
public sealed unsafe class PalitoEngine : IDisposable
{
    private readonly object _lock = new();
    private IntPtr _engine;

    private PalitoEngine(IntPtr engine) => _engine = engine;

    /// <summary>The library version.</summary>
    public static string Version => Marshal.PtrToStringUTF8(Native.palito_version())!;

    /// <summary>Opens a GGUF model.</summary>
    public static PalitoEngine Open(string modelPath, PalitoOptions? options = null)
    {
        options ??= new PalitoOptions();
        using var strings = new Utf8Strings();
        NativeOptions raw;
        Native.palito_options_init(&raw);
        raw.Device = (int)options.Device;
        raw.ContextTokens = options.ContextTokens;
        raw.Threads = options.Threads;
        raw.GpuDevice = options.GpuDevice;
        raw.GpuMemoryLimitBytes = options.GpuMemoryLimitBytes;
        if (options.KernelCacheDir is not null) raw.KernelCacheDir = strings.Add(options.KernelCacheDir);
        IntPtr error = IntPtr.Zero;
        IntPtr engine = Native.palito_open((byte*)strings.Add(modelPath), &raw, &error);
        if (engine == IntPtr.Zero) throw new PalitoException(TakeError(error));
        return new PalitoEngine(engine);
    }

    /// <summary>Opens a model on a thread-pool thread (loading can take seconds).</summary>
    public static Task<PalitoEngine> OpenAsync(string modelPath, PalitoOptions? options = null) =>
        Task.Run(() => Open(modelPath, options));

    /// <summary>Context tokens actually opened.</summary>
    public uint ContextTokens
    {
        get { lock (_lock) return Native.palito_context_tokens(Handle); }
    }

    /// <summary>"cpu", "int8-fast", "f32-tiled" or "f32".</summary>
    public string Profile
    {
        get { lock (_lock) return Marshal.PtrToStringUTF8(Native.palito_profile(Handle))!; }
    }

    /// <summary>Replies to <paramref name="messages"/> with the model's chat template.</summary>
    public Completion Chat(IReadOnlyList<Message> messages, GenerationOptions? options = null,
        CancellationToken cancellation = default)
    {
        return Generate(options, cancellation, (strings, gen, user, result, error) =>
        {
            var rows = new NativeMessage[Math.Max(messages.Count, 1)];
            for (int i = 0; i < messages.Count; i++)
            {
                rows[i].Role = strings.Add(messages[i].Role);
                rows[i].Content = strings.Add(messages[i].Content);
            }
            fixed (NativeMessage* pointer = rows)
            {
                return Native.palito_chat(Handle, pointer, (nuint)messages.Count, gen, &OnText,
                    user, result, error);
            }
        });
    }

    /// <summary>Continues <paramref name="prompt"/> without a chat template.</summary>
    public Completion Complete(string prompt, GenerationOptions? options = null,
        CancellationToken cancellation = default)
    {
        return Generate(options, cancellation, (strings, gen, user, result, error) =>
            Native.palito_complete(Handle, (byte*)strings.Add(prompt), gen, &OnText, user, result, error));
    }

    /// <summary><see cref="Chat"/> on a thread-pool thread.</summary>
    public Task<Completion> ChatAsync(IReadOnlyList<Message> messages, GenerationOptions? options = null,
        CancellationToken cancellation = default) =>
        Task.Run(() => Chat(messages, options, cancellation), CancellationToken.None);

    /// <summary><see cref="Complete"/> on a thread-pool thread.</summary>
    public Task<Completion> CompleteAsync(string prompt, GenerationOptions? options = null,
        CancellationToken cancellation = default) =>
        Task.Run(() => Complete(prompt, options, cancellation), CancellationToken.None);

    /// <summary>The reply text as it is generated; stopping the enumeration stops generation.</summary>
    public IAsyncEnumerable<string> StreamChatAsync(IReadOnlyList<Message> messages,
        GenerationOptions? options = null, CancellationToken cancellation = default) =>
        TextStream.Run((onText, token) => Chat(messages, With(options, onText), token), cancellation);

    /// <summary>Streaming <see cref="Complete"/>.</summary>
    public IAsyncEnumerable<string> StreamCompleteAsync(string prompt,
        GenerationOptions? options = null, CancellationToken cancellation = default) =>
        TextStream.Run((onText, token) => Complete(prompt, With(options, onText), token), cancellation);

    private static GenerationOptions With(GenerationOptions? options, Func<string, bool> onText)
    {
        options ??= new GenerationOptions();
        return new GenerationOptions
        {
            MaxTokens = options.MaxTokens,
            Thinking = options.Thinking,
            Temperature = options.Temperature,
            TopP = options.TopP,
            TopK = options.TopK,
            MinP = options.MinP,
            RepeatPenalty = options.RepeatPenalty,
            RepeatLastN = options.RepeatLastN,
            PresencePenalty = options.PresencePenalty,
            FrequencyPenalty = options.FrequencyPenalty,
            LogitBias = options.LogitBias,
            Seed = options.Seed,
            Stop = options.Stop,
            OnText = onText,
        };
    }

    private delegate int Call(Utf8Strings strings, NativeGeneration* generation, IntPtr user,
        NativeResult* result, IntPtr* error);

    private Completion Generate(GenerationOptions? options, CancellationToken cancellation, Call call)
    {
        options ??= new GenerationOptions();
        using var strings = new Utf8Strings();
        NativeGeneration gen;
        Native.palito_generation_init(&gen);
        gen.MaxTokens = options.MaxTokens;
        gen.EnableThinking = options.Thinking ? 1 : 0;
        gen.Temperature = options.Temperature;
        gen.TopP = options.TopP;
        gen.TopK = options.TopK;
        gen.MinP = options.MinP;
        gen.RepeatPenalty = options.RepeatPenalty;
        gen.RepeatLastN = options.RepeatLastN;
        gen.PresencePenalty = options.PresencePenalty;
        gen.FrequencyPenalty = options.FrequencyPenalty;
        var biases = options.LogitBias
            .Select(entry => new NativeLogitBias { Token = entry.Key, Bias = entry.Value })
            .DefaultIfEmpty()
            .ToArray();
        gen.Seed = options.Seed is ulong seed ? (long)(seed & long.MaxValue) : -1;
        var stops = new IntPtr[Math.Max(options.Stop.Count, 1)];
        for (int i = 0; i < options.Stop.Count; i++) stops[i] = strings.Add(options.Stop[i]);
        var sink = new Sink(options.OnText, cancellation);
        var handle = GCHandle.Alloc(sink);
        try
        {
            fixed (IntPtr* stopPointer = stops)
            fixed (NativeLogitBias* biasPointer = biases)
            {
                gen.Stop = options.Stop.Count > 0 ? (IntPtr)stopPointer : IntPtr.Zero;
                gen.StopCount = (nuint)options.Stop.Count;
                gen.LogitBias = options.LogitBias.Count > 0 ? (IntPtr)biasPointer : IntPtr.Zero;
                gen.LogitBiasCount = (nuint)options.LogitBias.Count;
                NativeResult result;
                IntPtr error = IntPtr.Zero;
                int status;
                lock (_lock)
                {
                    status = call(strings, &gen, GCHandle.ToIntPtr(handle), &result, &error);
                }
                if (status != 0) throw new PalitoException(TakeError(error));
                try
                {
                    if (sink.Failure is not null) throw sink.Failure;
                    cancellation.ThrowIfCancellationRequested();
                    return new Completion(
                        Marshal.PtrToStringUTF8(result.Text) ?? "",
                        (FinishReason)result.FinishReason,
                        (int)result.PromptTokens,
                        (int)result.GeneratedTokens,
                        TimeSpan.FromMilliseconds(result.TtftMs),
                        result.TokensPerSecond);
                }
                finally
                {
                    Native.palito_result_free(&result);
                }
            }
        }
        finally
        {
            handle.Free();
        }
    }

    private sealed class Sink(Func<string, bool>? onText, CancellationToken cancellation)
    {
        public Exception? Failure;

        public int Accept(string text)
        {
            if (Failure is not null || cancellation.IsCancellationRequested) return 1;
            if (onText is null) return 0;
            try
            {
                return onText(text) ? 0 : 1;
            }
            catch (Exception error)
            {
                Failure = error;
                return 1;
            }
        }
    }

    // Exceptions must not cross into native code: Sink.Accept catches them.
    [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static int OnText(byte* text, nuint length, IntPtr user)
    {
        var sink = (Sink)GCHandle.FromIntPtr(user).Target!;
        return sink.Accept(Encoding.UTF8.GetString(text, checked((int)length)));
    }

    private IntPtr Handle =>
        _engine != IntPtr.Zero ? _engine : throw new ObjectDisposedException(nameof(PalitoEngine));

    /// <summary>Releases the model; waits for a running request.</summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_engine != IntPtr.Zero)
            {
                Native.palito_close(_engine);
                _engine = IntPtr.Zero;
            }
        }
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the model if Dispose was not called.</summary>
    ~PalitoEngine()
    {
        if (_engine != IntPtr.Zero) Native.palito_close(_engine);
    }

    /// <summary>Where libpalito is loaded from.</summary>
    public static string LibraryPath()
    {
        string? explicitPath = Environment.GetEnvironmentVariable("PALITO_LIBRARY");
        if (!string.IsNullOrEmpty(explicitPath)) return explicitPath;
        string name = OperatingSystem.IsWindows() ? "palito.dll"
            : OperatingSystem.IsMacOS() ? "libpalito.dylib" : "libpalito.so";
        foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            try
            {
                var command = new FileInfo(Path.Combine(dir, "palito"));
                if (!command.Exists) continue;
                string real = command.ResolveLinkTarget(true)?.FullName ?? command.FullName;
                string bundled = Path.Combine(Path.GetDirectoryName(real)!, "lib", name);
                if (File.Exists(bundled)) return bundled;
            }
            catch (IOException)
            {
                // not here
            }
        }
        return name;
    }

    private static string TakeError(IntPtr error)
    {
        if (error == IntPtr.Zero) return "unknown error";
        string message = Marshal.PtrToStringUTF8(error) ?? "unknown error";
        Native.palito_string_free(error);
        return message;
    }

    /// <summary>NUL-terminated UTF-8 copies freed together.</summary>
    private sealed class Utf8Strings : IDisposable
    {
        private readonly List<IntPtr> _items = new();

        public IntPtr Add(string text)
        {
            IntPtr pointer = Marshal.StringToCoTaskMemUTF8(text);
            _items.Add(pointer);
            return pointer;
        }

        public void Dispose()
        {
            foreach (IntPtr pointer in _items) Marshal.FreeCoTaskMem(pointer);
            _items.Clear();
        }
    }
}

namespace Palito;

/// <summary>Where the model runs.</summary>
public enum Device
{
    /// <summary>CUDA when the library has it, else CPU.</summary>
    Default = 0,
    /// <summary>CPU.</summary>
    Cpu = 1,
    /// <summary>NVIDIA GPU.</summary>
    Cuda = 2,
    /// <summary>Vulkan GPU (AMD, Intel, NVIDIA).</summary>
    Vulkan = 3,
    /// <summary>Apple Silicon GPU (macOS Metal).</summary>
    Metal = 4,
}

/// <summary>Why generation ended.</summary>
public enum FinishReason
{
    /// <summary>The model ended its turn, or a stop sequence appeared.</summary>
    Stop = 0,
    /// <summary>MaxTokens was reached.</summary>
    Length = 1,
    /// <summary>No context left.</summary>
    ContextFull = 2,
    /// <summary>The text callback or the cancellation token stopped it.</summary>
    Cancelled = 3,
}

/// <summary>One chat turn: role "system", "user" or "assistant".</summary>
public sealed record Message(string Role, string Content)
{
    /// <summary>A system message.</summary>
    public static Message System(string content) => new("system", content);
    /// <summary>A user message.</summary>
    public static Message User(string content) => new("user", content);
    /// <summary>An assistant message.</summary>
    public static Message Assistant(string content) => new("assistant", content);
}

/// <summary>The result of Chat or Complete.</summary>
public sealed record Completion(
    string Text,
    FinishReason FinishReason,
    int PromptTokens,
    int GeneratedTokens,
    TimeSpan TimeToFirstToken,
    double TokensPerSecond);

/// <summary>Settings for <see cref="PalitoEngine.Open"/>; defaults are Palito's.</summary>
public sealed class PalitoOptions
{
    /// <summary>Default: CUDA when available, else CPU.</summary>
    public Device Device { get; init; } = Device.Default;
    /// <summary>0: the model's trained context, fitted to GPU memory.</summary>
    public uint ContextTokens { get; init; }
    /// <summary>0: 4 CPU threads.</summary>
    public uint Threads { get; init; }
    /// <summary>CUDA device index.</summary>
    public uint GpuDevice { get; init; }
    /// <summary>0: free GPU memory minus 2 GiB.</summary>
    public ulong GpuMemoryLimitBytes { get; init; }
    /// <summary>null: $PALITO_CUDA_KERNEL_CACHE or ~/.cache/palito/kernels.</summary>
    public string? KernelCacheDir { get; init; }
}

/// <summary>Settings for one request: greedy, reasoning on, no cap and no stops by default.</summary>
public sealed class GenerationOptions
{
    /// <summary>0: until the model stops or the context is full.</summary>
    public uint MaxTokens { get; init; }
    /// <summary>Reasoning for templates that support it (Qwen3).</summary>
    public bool Thinking { get; init; } = true;
    /// <summary>0: greedy.</summary>
    public float Temperature { get; init; }
    /// <summary>Nucleus mass in (0, 1]; 1: off.</summary>
    public float TopP { get; init; } = 1f;
    /// <summary>0: off.</summary>
    public uint TopK { get; init; }
    /// <summary>In [0, 1]: drop tokens below MinP times the most likely one's probability; 0: off.</summary>
    public float MinP { get; init; }
    /// <summary>llama.cpp repeat penalty (&gt; 0) over the last RepeatLastN tokens, prompt included; 1: off.</summary>
    public float RepeatPenalty { get; init; } = 1f;
    /// <summary>Window of the penalties; 0 turns them off.</summary>
    public uint RepeatLastN { get; init; } = 64;
    /// <summary>Subtracted once from every token in the window; 0: off.</summary>
    public float PresencePenalty { get; init; }
    /// <summary>Subtracted per occurrence in the window; 0: off.</summary>
    public float FrequencyPenalty { get; init; }
    /// <summary>Token id to a bias added to its logit; float.NegativeInfinity bans the token.</summary>
    public IReadOnlyDictionary<int, float> LogitBias { get; init; } = new Dictionary<int, float>();
    /// <summary>null: a different seed each call.</summary>
    public ulong? Seed { get; init; }
    /// <summary>Generation ends at the first of these; it is not part of the text.</summary>
    public IReadOnlyList<string> Stop { get; init; } = Array.Empty<string>();
    /// <summary>Receives text as it is generated; returning false stops generation.</summary>
    public Func<string, bool>? OnText { get; init; }
}

/// <summary>An error reported by the Palito library.</summary>
public sealed class PalitoException : Exception
{
    /// <summary>Creates the exception.</summary>
    public PalitoException(string message, Exception? inner = null) : base(message, inner) { }
}

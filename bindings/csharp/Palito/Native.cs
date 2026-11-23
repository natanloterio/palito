using System.Reflection;
using System.Runtime.InteropServices;

namespace Palito;

// Mirrors include/palito.h. Sequential layout gives the same padding as C.
[StructLayout(LayoutKind.Sequential)]
internal struct NativeOptions
{
    public int Device;
    public uint ContextTokens;
    public uint Threads;
    public uint GpuDevice;
    public ulong GpuMemoryLimitBytes;
    public IntPtr KernelCacheDir;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeMessage
{
    public IntPtr Role;
    public IntPtr Content;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeGeneration
{
    public uint MaxTokens;
    public int EnableThinking;
    public float Temperature;
    public float TopP;
    public uint TopK;
    public long Seed;
    public IntPtr Stop;
    public nuint StopCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeResult
{
    public IntPtr Text;
    public int FinishReason;
    public uint PromptTokens;
    public uint GeneratedTokens;
    public double TtftMs;
    public double TokensPerSecond;
}

internal static unsafe class Native
{
    private const string Library = "palito";

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? path)
    {
        if (name != Library) return IntPtr.Zero;
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("PALITO_LIBRARY"))
            && NativeLibrary.TryLoad(name, assembly, path, out IntPtr packaged))
        {
            return packaged;
        }
        string location = PalitoEngine.LibraryPath();
        try
        {
            return NativeLibrary.Load(location);
        }
        catch (DllNotFoundException error)
        {
            throw new PalitoException(
                $"cannot load {location}: install Palito or set PALITO_LIBRARY", error);
        }
    }

    [DllImport(Library)] public static extern IntPtr palito_version();
    [DllImport(Library)] public static extern void palito_options_init(NativeOptions* options);
    [DllImport(Library)] public static extern void palito_generation_init(NativeGeneration* generation);
    [DllImport(Library)] public static extern IntPtr palito_open(byte* path, NativeOptions* options, IntPtr* error);
    [DllImport(Library)] public static extern void palito_close(IntPtr engine);
    [DllImport(Library)] public static extern uint palito_context_tokens(IntPtr engine);
    [DllImport(Library)] public static extern IntPtr palito_profile(IntPtr engine);
    [DllImport(Library)]
    public static extern int palito_chat(IntPtr engine, NativeMessage* messages, nuint count,
        NativeGeneration* generation, delegate* unmanaged[Cdecl]<byte*, nuint, IntPtr, int> callback,
        IntPtr userData, NativeResult* result, IntPtr* error);
    [DllImport(Library)]
    public static extern int palito_complete(IntPtr engine, byte* prompt,
        NativeGeneration* generation, delegate* unmanaged[Cdecl]<byte*, nuint, IntPtr, int> callback,
        IntPtr userData, NativeResult* result, IntPtr* error);
    [DllImport(Library)] public static extern void palito_result_free(NativeResult* result);
    [DllImport(Library)] public static extern void palito_string_free(IntPtr text);
}

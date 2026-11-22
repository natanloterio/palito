// Run: PALITO_TEST_MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf dotnet test
using Palito;
using Xunit;

public sealed class EngineFixture : IDisposable
{
    public PalitoEngine? Engine { get; }

    public EngineFixture()
    {
        string? model = Environment.GetEnvironmentVariable("PALITO_TEST_MODEL");
        if (!string.IsNullOrEmpty(model))
            Engine = PalitoEngine.Open(model, new PalitoOptions { Device = Device.Cpu, ContextTokens = 512 });
    }

    public void Dispose() => Engine?.Dispose();
}

public class PalitoTests(EngineFixture fixture) : IClassFixture<EngineFixture>
{
    private static readonly Message[] Hello = { Message.User("Say hello.") };

    private PalitoEngine Llm => fixture.Engine ?? throw new InvalidOperationException("set PALITO_TEST_MODEL");

    [Fact]
    public async Task StreamedTextEqualsTheReply()
    {
        var streamed = new System.Text.StringBuilder();
        var reply = Llm.Chat(Hello, new GenerationOptions { MaxTokens = 24, OnText = t => { streamed.Append(t); return true; } });
        Assert.Equal(reply.Text, streamed.ToString());
        Assert.InRange(reply.GeneratedTokens, 1, 24);
        var iterated = new System.Text.StringBuilder();
        await foreach (var piece in Llm.StreamChatAsync(Hello, new GenerationOptions { MaxTokens = 24 }))
            iterated.Append(piece);
        Assert.Equal(reply.Text, iterated.ToString());
        Assert.Equal("cpu", Llm.Profile);
    }

    [Fact]
    public async Task StopSamplingAndCancel()
    {
        var cut = Llm.Complete("1, 2, 3, 4,", new GenerationOptions { MaxTokens = 24, Stop = new[] { " 7" } });
        Assert.Equal(FinishReason.Stop, cut.FinishReason);
        Assert.DoesNotContain(" 7", cut.Text);
        var sampled = new GenerationOptions { MaxTokens = 16, Temperature = 0.9f, Seed = 4 };
        var both = await Task.WhenAll(Llm.ChatAsync(Hello, sampled), Llm.ChatAsync(Hello, sampled));
        Assert.Equal(both[0].Text, both[1].Text);
        Assert.Equal(FinishReason.Cancelled,
            Llm.Complete("Once upon a time", new GenerationOptions { OnText = _ => false }).FinishReason);
        using var source = new CancellationTokenSource();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Llm.CompleteAsync("Once upon a time",
            new GenerationOptions { OnText = _ => { source.Cancel(); return true; } }, source.Token));
        await foreach (var _ in Llm.StreamCompleteAsync("Once upon a time", new GenerationOptions { MaxTokens = 200 }))
            break;
        Assert.Equal(2, Llm.Complete("Hi", new GenerationOptions { MaxTokens = 2 }).GeneratedTokens);
    }

    [Fact]
    public void Errors()
    {
        Assert.Throws<PalitoException>(() => PalitoEngine.Open("/nonexistent.gguf"));
        Assert.Throws<PalitoException>(() => Llm.Chat(Hello, new GenerationOptions { Temperature = -1 }));
        Assert.Throws<DivideByZeroException>(() => Llm.Complete("Hi", new GenerationOptions
        {
            OnText = _ => { int zero = 0; return 1 / zero == 0; },
        }));
    }
}

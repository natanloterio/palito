using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace Palito;

/// <summary>Turns a callback-streaming call into an async sequence.</summary>
internal static class TextStream
{
    public static async IAsyncEnumerable<string> Run(
        Func<Func<string, bool>, CancellationToken, Completion> run,
        [EnumeratorCancellation] CancellationToken cancellation = default)
    {
        var channel = Channel.CreateUnbounded<string>();
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        var task = Task.Run(() =>
        {
            try
            {
                run(text => channel.Writer.TryWrite(text), stop.Token);
                channel.Writer.Complete();
            }
            catch (Exception error)
            {
                channel.Writer.Complete(error);
            }
        }, CancellationToken.None);
        try
        {
            await foreach (var text in channel.Reader.ReadAllAsync(CancellationToken.None))
            {
                yield return text;
            }
        }
        finally
        {
            stop.Cancel();
            await task;
        }
    }
}

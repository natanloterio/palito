# Palito (C#)

C# binding for [Palito](../../README.md). It targets .NET 8+, uses P/Invoke, and has no dependencies.

```sh
dotnet add reference bindings/csharp/Palito/Palito.csproj
```

```csharp
using Palito;

using var llm = await PalitoEngine.OpenAsync("model.gguf");
var reply = await llm.ChatAsync(new[] { Message.User("Hello!") },
    new GenerationOptions { OnText = text => { Console.Write(text); return true; } });
await foreach (var text in llm.StreamChatAsync(new[] { Message.User("A haiku about tea") },
                   new GenerationOptions { Temperature = 0.7f, Stop = new[] { "\n\n" } }))
    Console.Write(text);
```

- **Cancellation:** a `CancellationToken` stops generation.
- **Concurrency:** calls on one engine wait their turn.

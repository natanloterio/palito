// usage: dotnet run --project Example -- model.gguf "message"
using Palito;

if (args.Length != 2)
{
    Console.Error.WriteLine("usage: Example model.gguf \"message\"");
    return 2;
}
using var llm = await PalitoEngine.OpenAsync(args[0]);
Console.Error.WriteLine($"palito {PalitoEngine.Version}, {llm.Profile}, context {llm.ContextTokens}");
await foreach (var text in llm.StreamChatAsync(new[] { Message.User(args[1]) }, new GenerationOptions { Thinking = false }))
    Console.Write(text);
Console.WriteLine();
return 0;

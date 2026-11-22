// usage (Java 22+): java --enable-native-access=ALL-UNNAMED -cp target/classes examples/Chat.java model.gguf "message"
import io.github.natanloterio.palito.*;
import java.nio.file.Path;
import java.util.List;

public class Chat {
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("usage: Chat model.gguf \"message\"");
            System.exit(2);
        }
        try (Palito llm = Palito.open(Path.of(args[0]), new Options())) {
            System.err.printf("palito %s, %s, context %d%n", Palito.version(), llm.profile(), llm.contextTokens());
            Completion reply = llm.chat(List.of(Message.user(args[1])), new Generation().thinking(false)
                    .onText(text -> { System.out.print(text); System.out.flush(); return true; }));
            System.err.printf("%n[%d tokens, %.1f tok/s, %s]%n", reply.generatedTokens(),
                    reply.tokensPerSecond(), reply.finishReason());
        }
    }
}

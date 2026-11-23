# palito (Java)

Java binding for [Palito](../../README.md). It uses the Foreign Function & Memory API (Java 22+), so it needs no JNI and no dependencies.

```sh
mvn -f bindings/java/pom.xml install   # io.github.natanloterio:palito:0.7.0
```

```java
try (Palito llm = Palito.open(Path.of("model.gguf"), new Options())) {
    Completion reply = llm.chat(List.of(Message.user("Hello!")), new Generation()
            .temperature(0.7f).stop(List.of("\n\n"))
            .onText(text -> { System.out.print(text); return true; }));
    System.out.println(reply.finishReason() + " " + reply.tokensPerSecond());
}
```

Run with `--enable-native-access=ALL-UNNAMED`. Without it, the JVM prints a warning on Java 22+. The example is `examples/Chat.java`.

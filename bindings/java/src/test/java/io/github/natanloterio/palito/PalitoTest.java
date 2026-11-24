package io.github.natanloterio.palito;

// Run: PALITO_TEST_MODEL=/path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf mvn test

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class PalitoTest {
    static final List<Message> HELLO = List.of(Message.user("Say hello."));
    static Palito llm;

    @BeforeAll
    static void open() {
        String model = System.getenv("PALITO_TEST_MODEL");
        assumeTrue(model != null && !model.isEmpty(), "set PALITO_TEST_MODEL");
        llm = Palito.open(Path.of(model),
                new Options().device(Options.Device.CPU).contextTokens(512));
    }

    @AfterAll
    static void close() {
        if (llm != null) llm.close();
    }

    @Test
    void streamedTextEqualsTheReply() {
        StringBuilder streamed = new StringBuilder();
        Completion reply = llm.chat(HELLO, new Generation().maxTokens(24)
                .onText(text -> { streamed.append(text); return true; }));
        assertEquals(reply.text(), streamed.toString());
        assertTrue(reply.generatedTokens() > 0 && reply.generatedTokens() <= 24);
        assertEquals(512, llm.contextTokens());
        assertEquals("cpu", llm.profile());
    }

    @Test
    void stopSamplingAndCancel() {
        Completion cut = llm.complete("1, 2, 3, 4,", new Generation().maxTokens(24).stop(List.of(" 7")));
        assertEquals(Completion.FinishReason.STOP, cut.finishReason());
        assertFalse(cut.text().contains(" 7"));
        Generation sampled = new Generation().maxTokens(16).temperature(0.9f).seed(4);
        assertEquals(llm.chat(HELLO, sampled).text(), llm.chat(HELLO, sampled).text());
        Completion cancelled = llm.complete("Once upon a time", new Generation().onText(t -> false));
        assertEquals(Completion.FinishReason.CANCELLED, cancelled.finishReason());
    }

    @Test
    void penaltiesMinPAndLogitBias() {
        String ones = "1, 1, 1, 1, 1, 1,";
        assertTrue(llm.complete(ones, new Generation().maxTokens(4)).text().startsWith(" 1"));
        Completion penalized = llm.complete(ones, new Generation().maxTokens(4)
                .frequencyPenalty(2f).presencePenalty(0.5f).repeatPenalty(1.1f));
        assertFalse(penalized.text().startsWith(" 1"), penalized.text());
        Generation seeded = new Generation().maxTokens(8).temperature(1f).minP(0.05f).seed(3);
        assertEquals(llm.complete(ones, seeded).text(), llm.complete(ones, seeded).text());
        assertThrows(PalitoException.class, () -> llm.complete(ones,
                new Generation().maxTokens(2).logitBias(Map.of(1 << 30, 1f))));
        assertThrows(PalitoException.class, () -> llm.complete(ones, new Generation().minP(2f)));
    }

    @Test
    void errors() {
        assertThrows(PalitoException.class, () -> Palito.open(Path.of("/nonexistent.gguf"), null));
        assertThrows(PalitoException.class, () -> llm.chat(HELLO, new Generation().temperature(-1)));
        assertThrows(ArithmeticException.class, () -> llm.complete("Hi",
                new Generation().onText(t -> 1 / 0 == 0)));
        assertEquals(2, llm.complete("Hi", new Generation().maxTokens(2)).generatedTokens());
    }
}

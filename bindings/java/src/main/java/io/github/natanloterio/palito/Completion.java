package io.github.natanloterio.palito;

import java.time.Duration;

/** The result of {@link Palito#chat} or {@link Palito#complete}. */
public record Completion(
        String text,
        FinishReason finishReason,
        int promptTokens,
        int generatedTokens,
        Duration timeToFirstToken,
        double tokensPerSecond) {

    public enum FinishReason {
        /** The model ended its turn, or a stop sequence appeared. */
        STOP,
        /** maxTokens was reached. */
        LENGTH,
        /** No context left. */
        CONTEXT_FULL,
        /** The text callback asked to stop. */
        CANCELLED
    }
}

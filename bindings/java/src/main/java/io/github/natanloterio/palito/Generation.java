package io.github.natanloterio.palito;

import java.util.List;
import java.util.OptionalLong;
import java.util.function.Predicate;

/** Settings for one request; the defaults are greedy, reasoning on, no cap, no stops. */
public final class Generation {
    int maxTokens;
    boolean thinking = true;
    float temperature;
    float topP = 1f;
    int topK;
    OptionalLong seed = OptionalLong.empty();
    List<String> stop = List.of();
    Predicate<String> onText;

    /** 0: until the model stops or the context is full. */
    public Generation maxTokens(int tokens) { this.maxTokens = tokens; return this; }
    /** Reasoning for templates that support it (Qwen3). */
    public Generation thinking(boolean on) { this.thinking = on; return this; }
    /** 0: greedy. */
    public Generation temperature(float temperature) { this.temperature = temperature; return this; }
    public Generation topP(float topP) { this.topP = topP; return this; }
    public Generation topK(int topK) { this.topK = topK; return this; }
    public Generation seed(long seed) { this.seed = OptionalLong.of(seed); return this; }
    /** Generation ends at the first of these; it is not part of the text. */
    public Generation stop(List<String> stop) { this.stop = List.copyOf(stop); return this; }
    /** Receives text as it is generated; returning false stops generation. */
    public Generation onText(Predicate<String> onText) { this.onText = onText; return this; }
}

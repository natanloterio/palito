package io.github.natanloterio.palito;

/** An error reported by the Palito library. */
public class PalitoException extends RuntimeException {
    public PalitoException(String message) { super(message); }
    public PalitoException(String message, Throwable cause) { super(message, cause); }
}

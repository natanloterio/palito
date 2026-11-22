package io.github.natanloterio.palito;

import java.nio.file.Path;

/** Settings for {@link Palito#open}; the defaults are Palito's. */
public final class Options {
    public enum Device { DEFAULT, CPU, CUDA }

    Device device = Device.DEFAULT;
    int contextTokens;
    int threads;
    int gpuDevice;
    long gpuMemoryLimitBytes;
    Path kernelCacheDir;

    /** DEFAULT: CUDA when the library has it, else CPU. */
    public Options device(Device device) { this.device = device; return this; }
    /** 0: the model's trained context, fitted to GPU memory. */
    public Options contextTokens(int tokens) { this.contextTokens = tokens; return this; }
    public Options threads(int threads) { this.threads = threads; return this; }
    public Options gpuDevice(int index) { this.gpuDevice = index; return this; }
    /** 0: free GPU memory minus 2 GiB. */
    public Options gpuMemoryLimitBytes(long bytes) { this.gpuMemoryLimitBytes = bytes; return this; }
    public Options kernelCacheDir(Path dir) { this.kernelCacheDir = dir; return this; }
}

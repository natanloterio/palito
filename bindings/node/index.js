'use strict';
// Node.js binding for Palito: fast local LLM inference for GGUF models.
//
// Generation runs on a worker thread (the event loop stays free); streamed text
// arrives through `onText` or the async iterators. The library is found via
// $PALITO_LIBRARY, then the installed `palito` command (<bundle>/lib), then the
// system loader.

const fs = require('node:fs');
const path = require('node:path');
const koffi = require('koffi');

const DEVICES = { default: 0, cpu: 1, cuda: 2 };
const FINISH_REASONS = ['stop', 'length', 'context_full', 'cancelled'];
const LIBRARY_NAME = { darwin: 'libpalito.dylib', win32: 'palito.dll' }[process.platform] || 'libpalito.so';

class PalitoError extends Error {
  constructor(message) {
    super(message);
    this.name = 'PalitoError';
  }
}

function libraryPath() {
  if (process.env.PALITO_LIBRARY) return process.env.PALITO_LIBRARY;
  const separator = process.platform === 'win32' ? ';' : ':';
  for (const dir of (process.env.PATH || '').split(separator)) {
    const command = path.join(dir, 'palito');
    try {
      const bundled = path.join(path.dirname(fs.realpathSync(command)), 'lib', LIBRARY_NAME);
      if (fs.statSync(bundled).isFile()) return bundled;
    } catch {
      // not here
    }
  }
  return LIBRARY_NAME;
}

let api = null;

function load() {
  if (api) return api;
  const file = libraryPath();
  let lib;
  try {
    lib = koffi.load(file);
  } catch (error) {
    throw new PalitoError(`cannot load ${file}: ${error.message}. Install Palito or set PALITO_LIBRARY`);
  }
  const Options = koffi.struct('palito_options', {
    device: 'int',
    context_tokens: 'uint32_t',
    threads: 'uint32_t',
    gpu_device: 'uint32_t',
    gpu_memory_limit_bytes: 'uint64_t',
    kernel_cache_dir: 'const char *',
  });
  const Message = koffi.struct('palito_message', { role: 'const char *', content: 'const char *' });
  const Generation = koffi.struct('palito_generation', {
    max_tokens: 'uint32_t',
    enable_thinking: 'int',
    temperature: 'float',
    top_p: 'float',
    top_k: 'uint32_t',
    seed: 'int64_t',
    stop: 'const char **',
    stop_count: 'size_t',
  });
  const Result = koffi.struct('palito_result', {
    text: 'void *',
    finish_reason: 'int',
    prompt_tokens: 'uint32_t',
    generated_tokens: 'uint32_t',
    ttft_ms: 'double',
    tokens_per_second: 'double',
  });
  const TextCallback = koffi.proto('int palito_text_callback(const uint8_t *text, size_t length, void *user_data)');
  api = {
    Options,
    Generation,
    Result,
    TextCallback,
    version: lib.func('const char *palito_version(void)'),
    optionsInit: lib.func('void palito_options_init(_Out_ palito_options *options)'),
    generationInit: lib.func('void palito_generation_init(_Out_ palito_generation *generation)'),
    open: lib.func('void *palito_open(const char *path, const palito_options *options, _Out_ void **error)'),
    close: lib.func('void palito_close(void *engine)'),
    contextTokens: lib.func('uint32_t palito_context_tokens(void *engine)'),
    profile: lib.func('const char *palito_profile(void *engine)'),
    chat: lib.func('int palito_chat(void *engine, const palito_message *messages, size_t count, ' +
      'const palito_generation *generation, palito_text_callback *callback, void *user_data, ' +
      '_Out_ palito_result *result, _Out_ void **error)'),
    complete: lib.func('int palito_complete(void *engine, const char *prompt, ' +
      'const palito_generation *generation, palito_text_callback *callback, void *user_data, ' +
      '_Out_ palito_result *result, _Out_ void **error)'),
    resultFree: lib.func('void palito_result_free(_Inout_ palito_result *result)'),
    stringFree: lib.func('void palito_string_free(void *string)'),
  };
  return api;
}

function takeError(pointer) {
  if (!pointer) return 'unknown error';
  const message = koffi.decode(pointer, 'char', -1);
  load().stringFree(pointer);
  return message;
}

function callAsync(fn, ...args) {
  return new Promise((resolve, reject) => {
    fn.async(...args, (error, value) => (error ? reject(error) : resolve(value)));
  });
}

/** A model opened once; chat() and complete() run one at a time per engine. */
class Palito {
  /**
   * Opens a GGUF model on a worker thread.
   * options: { device: 'default'|'cpu'|'cuda', contextTokens, threads, gpuDevice,
   *            gpuMemoryLimitBytes, kernelCacheDir }
   */
  static async open(modelPath, options = {}) {
    const native = load();
    const raw = {};
    native.optionsInit(raw);
    const device = DEVICES[options.device || 'default'];
    if (device === undefined) throw new PalitoError(`unknown device ${options.device}`);
    Object.assign(raw, {
      device,
      context_tokens: options.contextTokens || 0,
      threads: options.threads || 0,
      gpu_device: options.gpuDevice || 0,
      gpu_memory_limit_bytes: options.gpuMemoryLimitBytes || 0,
      kernel_cache_dir: options.kernelCacheDir || null,
    });
    const error = [null];
    const engine = await callAsync(native.open, modelPath, raw, error);
    if (!engine) throw new PalitoError(takeError(error[0]));
    return new Palito(engine);
  }

  constructor(engine) {
    this._engine = engine;
    this._queue = Promise.resolve();
  }

  get contextTokens() {
    return load().contextTokens(this._handle());
  }

  /** 'cpu', 'int8-fast', 'f32-tiled' or 'f32'. */
  get profile() {
    return load().profile(this._handle());
  }

  /**
   * Replies to messages ([{role, content}]) with the model's chat template.
   * options: { maxTokens, thinking (true), temperature (0: greedy), topP, topK, seed,
   *            stop: string[], onText(text) → return false to stop }
   */
  chat(messages, options = {}) {
    const rows = messages.map((m) => ({ role: m.role, content: m.content }));
    return this._run(options, (native, generation, callback, result, error) =>
      callAsync(native.chat, this._handle(), rows, rows.length, generation, callback, null, result, error));
  }

  /** Continues prompt without a chat template; options as in chat(). */
  complete(prompt, options = {}) {
    return this._run(options, (native, generation, callback, result, error) =>
      callAsync(native.complete, this._handle(), prompt, generation, callback, null, result, error));
  }

  /** Async iterator over the reply text; breaking out of the loop stops generation. */
  streamChat(messages, options = {}) {
    return this._stream((onText) => this.chat(messages, { ...options, onText }));
  }

  streamComplete(prompt, options = {}) {
    return this._stream((onText) => this.complete(prompt, { ...options, onText }));
  }

  async *_stream(start) {
    const pieces = [];
    let wake = null;
    let stopped = false;
    let finished = false;
    let failure = null;
    const notify = () => wake && wake();
    const run = start((text) => {
      pieces.push(text);
      notify();
      return !stopped;
    }).then(
      () => { finished = true; notify(); },
      (error) => { failure = error; finished = true; notify(); },
    );
    try {
      while (true) {
        if (pieces.length) {
          yield pieces.shift();
          continue;
        }
        if (finished) break;
        await new Promise((resolve) => { wake = resolve; });
        wake = null;
      }
      if (failure) throw failure;
    } finally {
      stopped = true;
      await run;
    }
  }

  _run(options, call) {
    const task = this._queue.then(() => this._generate(options, call));
    this._queue = task.catch(() => {});
    return task;
  }

  async _generate(options, call) {
    const native = load();
    const generation = {};
    native.generationInit(generation);
    const stop = options.stop || [];
    Object.assign(generation, {
      max_tokens: options.maxTokens || 0,
      enable_thinking: options.thinking === false ? 0 : 1,
      temperature: options.temperature ?? 0,
      top_p: options.topP ?? 1,
      top_k: options.topK ?? 0,
      seed: options.seed ?? -1,
      stop: stop.length ? stop : null,
      stop_count: stop.length,
    });
    let failure = null;
    const onText = options.onText;
    const callback = koffi.register((text, length) => {
      if (!onText) return 0;
      try {
        const bytes = koffi.decode(text, 'uint8_t', Number(length));
        return onText(Buffer.from(bytes).toString('utf8')) === false ? 1 : 0;
      } catch (error) {
        failure = error;
        return 1;
      }
    }, koffi.pointer(native.TextCallback));
    const result = {};
    const error = [null];
    let status;
    try {
      status = await call(native, generation, callback, result, error);
    } finally {
      koffi.unregister(callback);
    }
    if (status !== 0) throw new PalitoError(takeError(error[0]));
    let completion;
    try {
      completion = {
        text: koffi.decode(result.text, 'char', -1),
        finishReason: FINISH_REASONS[result.finish_reason],
        promptTokens: result.prompt_tokens,
        generatedTokens: result.generated_tokens,
        ttftMs: result.ttft_ms,
        tokensPerSecond: result.tokens_per_second,
      };
    } finally {
      native.resultFree(result);
    }
    if (failure) throw failure;
    return completion;
  }

  _handle() {
    if (!this._engine) throw new PalitoError('engine is closed');
    return this._engine;
  }

  /** Waits for queued requests, then releases the model. */
  async close() {
    await this._queue;
    if (this._engine) {
      load().close(this._engine);
      this._engine = null;
    }
  }

  async [Symbol.asyncDispose]() {
    await this.close();
  }
}

function version() {
  return load().version();
}

module.exports = { Palito, PalitoError, version, libraryPath };

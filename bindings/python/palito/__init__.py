"""Python binding for Palito: fast local LLM inference for GGUF models.

Standard library only (ctypes) on top of the C API in libpalito.so.

    import palito

    with palito.Palito("model.gguf") as llm:
        reply = llm.chat([{"role": "user", "content": "Hello!"}],
                         on_text=lambda text: print(text, end="", flush=True))
        print(reply.finish_reason, reply.tokens_per_second)

        for text in llm.stream_chat([{"role": "user", "content": "Tell a joke"}],
                                    temperature=0.7):
            print(text, end="", flush=True)

        person = llm.chat([{"role": "user", "content": "Invent a person."}],
                          json_schema={"type": "object",
                                       "properties": {"name": {"type": "string"}},
                                       "required": ["name"]})
        print(json.loads(person.text)["name"])

The library is found via $PALITO_LIBRARY, then the installed `palito` command
(<bundle>/lib/libpalito.so), then the system loader.
"""

import ctypes
import json
import os
import queue
import shutil
import sys
import threading
from ctypes import (CFUNCTYPE, POINTER, Structure, c_char_p, c_double, c_float, c_int32,
                    c_int, c_int64, c_size_t, c_uint32, c_uint64, c_void_p)
from dataclasses import dataclass

__all__ = ["Palito", "PalitoError", "Completion", "DEVICE_DEFAULT", "DEVICE_CPU",
           "DEVICE_CUDA", "version", "library_path"]

DEVICE_DEFAULT, DEVICE_CPU, DEVICE_CUDA = 0, 1, 2
_FINISH_REASONS = {0: "stop", 1: "length", 2: "context_full", 3: "cancelled"}


class PalitoError(RuntimeError):
    """An error reported by the Palito library."""


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str  # "stop", "length", "context_full" or "cancelled"
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    tokens_per_second: float


class _Options(Structure):
    _fields_ = [("device", c_int), ("context_tokens", c_uint32), ("threads", c_uint32),
                ("gpu_device", c_uint32), ("gpu_memory_limit_bytes", c_uint64),
                ("kernel_cache_dir", c_char_p)]


class _Message(Structure):
    _fields_ = [("role", c_char_p), ("content", c_char_p)]


class _LogitBias(Structure):
    _fields_ = [("token", c_int32), ("bias", c_float)]


class _Generation(Structure):
    _fields_ = [("max_tokens", c_uint32), ("enable_thinking", c_int),
                ("temperature", c_float), ("top_p", c_float), ("top_k", c_uint32),
                ("seed", c_int64), ("stop", POINTER(c_char_p)), ("stop_count", c_size_t),
                ("min_p", c_float), ("repeat_penalty", c_float), ("repeat_last_n", c_uint32),
                ("presence_penalty", c_float), ("frequency_penalty", c_float),
                ("logit_bias", POINTER(_LogitBias)), ("logit_bias_count", c_size_t)]


class _Result(Structure):
    _fields_ = [("text", c_void_p), ("finish_reason", c_int), ("prompt_tokens", c_uint32),
                ("generated_tokens", c_uint32), ("ttft_ms", c_double),
                ("tokens_per_second", c_double)]


_Callback = CFUNCTYPE(c_int, c_void_p, c_size_t, c_void_p)

_LIBRARY_NAME = {"darwin": "libpalito.dylib", "win32": "palito.dll"}.get(sys.platform,
                                                                         "libpalito.so")


def library_path():
    """Where libpalito is loaded from (see the module docstring)."""
    explicit = os.environ.get("PALITO_LIBRARY")
    if explicit:
        return explicit
    command = shutil.which("palito")
    if command:
        bundled = os.path.join(os.path.dirname(os.path.realpath(command)), "lib", _LIBRARY_NAME)
        if os.path.isfile(bundled):
            return bundled
    return _LIBRARY_NAME


_lib = None
_lib_lock = threading.Lock()


def _load():
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        path = library_path()
        try:
            lib = ctypes.CDLL(path)
        except OSError as error:
            raise PalitoError(f"cannot load {path}: {error}. Install Palito or set "
                              "PALITO_LIBRARY to libpalito.so") from error
        lib.palito_version.restype = c_char_p
        lib.palito_options_init.argtypes = [POINTER(_Options)]
        lib.palito_generation_init.argtypes = [POINTER(_Generation)]
        lib.palito_open.argtypes = [c_char_p, POINTER(_Options), POINTER(c_void_p)]
        lib.palito_open.restype = c_void_p
        lib.palito_close.argtypes = [c_void_p]
        lib.palito_context_tokens.argtypes = [c_void_p]
        lib.palito_context_tokens.restype = c_uint32
        lib.palito_profile.argtypes = [c_void_p]
        lib.palito_profile.restype = c_char_p
        common = [POINTER(_Generation), _Callback, c_void_p, POINTER(_Result),
                  POINTER(c_void_p)]
        lib.palito_chat.argtypes = [c_void_p, POINTER(_Message), c_size_t] + common
        lib.palito_complete.argtypes = [c_void_p, c_char_p] + common
        if hasattr(lib, "palito_chat_json"):  # newer than 0.5.0
            lib.palito_chat_json.argtypes = ([c_void_p, POINTER(_Message), c_size_t]
                                             + common[:1] + [c_char_p] + common[1:])
            lib.palito_complete_json.argtypes = ([c_void_p, c_char_p] + common[:1]
                                                 + [c_char_p] + common[1:])
        lib.palito_result_free.argtypes = [POINTER(_Result)]
        lib.palito_string_free.argtypes = [c_void_p]
        _lib = lib
        return lib


def version():
    return _load().palito_version().decode()


def _take_error(lib, error):
    message = ctypes.string_at(error.value).decode() if error.value else "unknown error"
    lib.palito_string_free(error)
    return message


def _schema_bytes(json_schema):
    if json_schema is None:
        return None
    if isinstance(json_schema, (bytes, str)):
        return json_schema.encode() if isinstance(json_schema, str) else json_schema
    return json.dumps(json_schema).encode()


class Palito:
    """A model opened once; chat() and complete() can be called many times.

    One request runs at a time per engine (calls from several threads wait).
    """

    def __init__(self, model_path, device=DEVICE_DEFAULT, context_tokens=0, threads=0,
                 gpu_device=0, gpu_memory_limit_bytes=0, kernel_cache_dir=None):
        self._lib = _load()
        self._lock = threading.Lock()
        options = _Options()
        self._lib.palito_options_init(ctypes.byref(options))
        options.device = device
        options.context_tokens = context_tokens
        options.threads = threads
        options.gpu_device = gpu_device
        options.gpu_memory_limit_bytes = gpu_memory_limit_bytes
        if kernel_cache_dir is not None:
            options.kernel_cache_dir = os.fsencode(kernel_cache_dir)
        error = c_void_p()
        self._engine = self._lib.palito_open(os.fsencode(model_path), ctypes.byref(options),
                                             ctypes.byref(error))
        if not self._engine:
            raise PalitoError(_take_error(self._lib, error))

    @property
    def context_tokens(self):
        return self._lib.palito_context_tokens(self._handle())

    @property
    def profile(self):
        """"cpu", "int8-fast", "f32-tiled" or "f32"."""
        return self._lib.palito_profile(self._handle()).decode()

    def chat(self, messages, on_text=None, **generation):
        """Replies to messages ([{"role": ..., "content": ...}]) with the model's chat template.

        on_text(str) receives the reply as it is generated; returning False stops it.
        Generation keywords: max_tokens=0 (no cap), thinking=True, temperature=0.0
        (greedy), top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0, repeat_last_n=64,
        presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None ({token id: bias},
        float("-inf") bans a token), seed=None, stop=() (strings), json_schema=None
        (a JSON Schema dict or string: the reply is JSON matching it, reasoning off;
        {"type": "object"} for any object).
        """
        array = (_Message * len(messages))(
            *[_Message(m["role"].encode(), m["content"].encode()) for m in messages])
        return self._run(lambda gen, schema, cb, res, err: (
            self._lib.palito_chat(self._handle(), array, len(messages), gen, cb, None, res, err)
            if schema is None else
            self._lib.palito_chat_json(self._handle(), array, len(messages), gen, schema, cb,
                                       None, res, err)),
            on_text, **generation)

    def complete(self, prompt, on_text=None, **generation):
        """Continues prompt without a chat template; keywords as in chat()."""
        return self._run(lambda gen, schema, cb, res, err: (
            self._lib.palito_complete(self._handle(), prompt.encode(), gen, cb, None, res, err)
            if schema is None else
            self._lib.palito_complete_json(self._handle(), prompt.encode(), gen, schema, cb,
                                           None, res, err)),
            on_text, **generation)

    def stream_chat(self, messages, **generation):
        """Yields the reply text as it is generated. Closing the generator stops it."""
        return self._stream(lambda on_text: self.chat(messages, on_text, **generation))

    def stream_complete(self, prompt, **generation):
        return self._stream(lambda on_text: self.complete(prompt, on_text, **generation))

    def _stream(self, run):
        pieces = queue.Queue()
        stop = threading.Event()
        done = object()

        def on_text(text):
            pieces.put(text)
            return not stop.is_set()

        def worker():
            try:
                run(on_text)
                pieces.put(done)
            except BaseException as error:  # re-raised in the consumer
                pieces.put(error)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            while True:
                item = pieces.get()
                if item is done:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            thread.join()

    def _run(self, call, on_text, max_tokens=0, thinking=True, temperature=0.0,
             top_p=1.0, top_k=0, min_p=0.0, repeat_penalty=1.0, repeat_last_n=64,
             presence_penalty=0.0, frequency_penalty=0.0, logit_bias=None, seed=None, stop=(),
             json_schema=None):
        schema = _schema_bytes(json_schema)
        if schema is not None and not hasattr(self._lib, "palito_chat_json"):
            raise PalitoError("json_schema needs a libpalito newer than 0.5.0")
        generation = _Generation()
        self._lib.palito_generation_init(ctypes.byref(generation))
        generation.max_tokens = max_tokens
        generation.enable_thinking = 1 if thinking else 0
        generation.temperature = temperature
        generation.top_p = top_p
        generation.top_k = top_k
        generation.min_p = min_p
        generation.repeat_penalty = repeat_penalty
        generation.repeat_last_n = repeat_last_n
        generation.presence_penalty = presence_penalty
        generation.frequency_penalty = frequency_penalty
        biases = list((logit_bias or {}).items())
        bias_array = (_LogitBias * len(biases))(*[_LogitBias(t, b) for t, b in biases])
        generation.logit_bias = bias_array
        generation.logit_bias_count = len(biases)
        generation.seed = -1 if seed is None else seed
        stops = (c_char_p * len(stop))(*[s.encode() for s in stop])
        generation.stop = stops
        generation.stop_count = len(stop)
        failure = []

        def forward(text, length, _user):
            if on_text is None:
                return 0
            try:
                return 0 if on_text(ctypes.string_at(text, length).decode()) is not False else 1
            except BaseException as error:  # stop generation, re-raise after the call
                failure.append(error)
                return 1

        callback = _Callback(forward)
        result = _Result()
        error = c_void_p()
        with self._lock:
            status = call(ctypes.byref(generation), schema, callback, ctypes.byref(result),
                          ctypes.byref(error))
        if status:
            raise PalitoError(_take_error(self._lib, error))
        try:
            completion = Completion(
                text=ctypes.string_at(result.text).decode(),
                finish_reason=_FINISH_REASONS[result.finish_reason],
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                ttft_ms=result.ttft_ms,
                tokens_per_second=result.tokens_per_second,
            )
        finally:
            self._lib.palito_result_free(ctypes.byref(result))
        if failure:
            raise failure[0]
        return completion

    def _handle(self):
        if not self._engine:
            raise PalitoError("engine is closed")
        return self._engine

    def close(self):
        with self._lock:
            if self._engine:
                self._lib.palito_close(self._engine)
                self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_engine", None):
            self.close()

"""Minimal Python binding for the Palito C API (ctypes, no dependencies).

    from palito import Palito
    with Palito("model.gguf") as llm:
        reply = llm.chat([{"role": "user", "content": "Hello!"}],
                         on_text=lambda t: print(t, end="", flush=True))

Set PALITO_LIBRARY to the path of libpalito.so if it is not on the loader path.
"""

import ctypes
import os
from ctypes import (CFUNCTYPE, POINTER, Structure, c_char_p, c_double, c_int,
                    c_size_t, c_uint32, c_uint64, c_void_p)

DEVICE_DEFAULT, DEVICE_CPU, DEVICE_CUDA = 0, 1, 2
FINISH_REASONS = {0: "stop", 1: "length", 2: "context_full", 3: "cancelled"}


class _Options(Structure):
    _fields_ = [("device", c_int), ("context_tokens", c_uint32), ("threads", c_uint32),
                ("gpu_device", c_uint32), ("gpu_memory_limit_bytes", c_uint64),
                ("kernel_cache_dir", c_char_p)]


class _Message(Structure):
    _fields_ = [("role", c_char_p), ("content", c_char_p)]


class _Generation(Structure):
    _fields_ = [("max_tokens", c_uint32), ("enable_thinking", c_int)]


class _Result(Structure):
    _fields_ = [("text", c_void_p), ("finish_reason", c_int), ("prompt_tokens", c_uint32),
                ("generated_tokens", c_uint32), ("ttft_ms", c_double),
                ("tokens_per_second", c_double)]


_Callback = CFUNCTYPE(c_int, c_void_p, c_size_t, c_void_p)


def _load():
    lib = ctypes.CDLL(os.environ.get("PALITO_LIBRARY", "libpalito.so"))
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
    common = [POINTER(_Generation), _Callback, c_void_p, POINTER(_Result), POINTER(c_void_p)]
    lib.palito_chat.argtypes = [c_void_p, POINTER(_Message), c_size_t] + common
    lib.palito_complete.argtypes = [c_void_p, c_char_p] + common
    lib.palito_result_free.argtypes = [POINTER(_Result)]
    lib.palito_string_free.argtypes = [c_void_p]
    return lib


class PalitoError(RuntimeError):
    pass


class Palito:
    def __init__(self, model_path, device=DEVICE_DEFAULT, context_tokens=0,
                 gpu_memory_limit_bytes=0):
        self._lib = _load()
        options = _Options()
        self._lib.palito_options_init(ctypes.byref(options))
        options.device = device
        options.context_tokens = context_tokens
        options.gpu_memory_limit_bytes = gpu_memory_limit_bytes
        error = c_void_p()
        self._engine = self._lib.palito_open(os.fsencode(model_path), ctypes.byref(options),
                                             ctypes.byref(error))
        if not self._engine:
            raise PalitoError(self._take(error))

    @property
    def version(self):
        return self._lib.palito_version().decode()

    @property
    def context_tokens(self):
        return self._lib.palito_context_tokens(self._engine)

    @property
    def profile(self):
        return self._lib.palito_profile(self._engine).decode()

    def chat(self, messages, max_tokens=0, thinking=True, on_text=None):
        """messages: [{"role": ..., "content": ...}]. on_text(str) may return False to stop."""
        array = (_Message * len(messages))(
            *[_Message(m["role"].encode(), m["content"].encode()) for m in messages])
        return self._run(lambda gen, cb, res, err: self._lib.palito_chat(
            self._engine, array, len(messages), gen, cb, None, res, err),
            max_tokens, thinking, on_text)

    def complete(self, prompt, max_tokens=0, on_text=None):
        return self._run(lambda gen, cb, res, err: self._lib.palito_complete(
            self._engine, prompt.encode(), gen, cb, None, res, err),
            max_tokens, True, on_text)

    def _run(self, call, max_tokens, thinking, on_text):
        generation = _Generation()
        self._lib.palito_generation_init(ctypes.byref(generation))
        generation.max_tokens = max_tokens
        generation.enable_thinking = 1 if thinking else 0

        def forward(text, length, _user):
            if on_text is None:
                return 0
            return 0 if on_text(ctypes.string_at(text, length).decode()) is not False else 1

        callback = _Callback(forward)
        result = _Result()
        error = c_void_p()
        if call(ctypes.byref(generation), callback, ctypes.byref(result), ctypes.byref(error)):
            raise PalitoError(self._take(error))
        try:
            return {
                "text": ctypes.string_at(result.text).decode(),
                "finish_reason": FINISH_REASONS[result.finish_reason],
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "ttft_ms": result.ttft_ms,
                "tokens_per_second": result.tokens_per_second,
            }
        finally:
            self._lib.palito_result_free(ctypes.byref(result))

    def _take(self, error):
        message = ctypes.string_at(error.value).decode() if error.value else "unknown error"
        self._lib.palito_string_free(error)
        return message

    def close(self):
        if self._engine:
            self._lib.palito_close(self._engine)
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python3 palito.py model.gguf 'message'")
    with Palito(sys.argv[1]) as llm:
        print(f"palito {llm.version}, {llm.profile}, context {llm.context_tokens}",
              file=sys.stderr)
        out = llm.chat([{"role": "user", "content": sys.argv[2]}], thinking=False,
                       on_text=lambda t: print(t, end="", flush=True))
        print(f"\n[{out['generated_tokens']} tokens, {out['tokens_per_second']:.1f} tok/s,"
              f" {out['finish_reason']}]", file=sys.stderr)

#!/usr/bin/env python3
"""Checks `palito serve --parallel N` with the official OpenAI SDK.

Writes one JSON report and exits non-zero if a case fails. A throughput below
the target is recorded as `target_missed` and does not fail the run.
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

COUNTING = ", ".join(str(n) for n in range(1, 60)) + ","
LONG_PROMPT = " ".join(f"Item {n} is on the shelf." for n in range(1, 160))
PROMPTS = [
    "The capital of France is",
    "Write a short poem about the rain:\n",
    "def fibonacci(n):\n",
    LONG_PROMPT + "\nThe list above describes",
]
THROUGHPUT_TARGET = 0.80


def complete(client, model, prompt, **options):
    return client.completions.create(model=model, prompt=prompt, **options)


def texts(client, model, options):
    return [complete(client, model, p, **options).choices[0].text for p in PROMPTS]


def concurrently(calls):
    with ThreadPoolExecutor(len(calls)) as pool:
        return [future.result() for future in [pool.submit(call) for call in calls]]


def short_beats_long(client, model):
    finished = {}
    started = threading.Event()

    def long_request():
        stream = complete(client, model, COUNTING, max_tokens=1800, temperature=0, stream=True)
        for _ in stream:
            started.set()
        finished["long"] = time.monotonic()

    thread = threading.Thread(target=long_request)
    thread.start()
    started.wait(120)
    begin = time.monotonic()
    first = None
    stream = complete(client, model, "Hello, my name is", max_tokens=16, temperature=0, stream=True)
    for _ in stream:
        first = first or time.monotonic() - begin
    finished["short"] = time.monotonic()
    long_was_running = "long" not in finished
    thread.join()
    passed = first is not None and first < 1.0 and long_was_running
    return {"pass": passed, "first_token_s": first, "long_was_running": long_was_running}


def isolation(client, model):
    results = {}
    for name, options in (
        ("greedy", {"max_tokens": 48, "temperature": 0}),
        ("seeded", {"max_tokens": 48, "temperature": 0.8, "seed": 42}),
    ):
        alone = texts(client, model, options)
        calls = [
            (lambda p=p: complete(client, model, p, **options).choices[0].text) for p in PROMPTS
        ]
        mixed = concurrently(calls)
        results[name] = {"alone": alone, "same": alone == mixed}
    return {"pass": all(r["same"] for r in results.values()), **results}


def rate(calls):
    begin = time.monotonic()
    replies = concurrently(calls)
    elapsed = time.monotonic() - begin
    tokens = sum(reply.usage.completion_tokens for reply in replies)
    return {"tokens": tokens, "seconds": elapsed, "tokens_per_second": tokens / elapsed}


def throughput(client, model):
    call = lambda start: (
        lambda: complete(
            client,
            model,
            ", ".join(str(start + n) for n in range(40)) + ",",
            max_tokens=600,
            temperature=0,
        )
    )
    one = rate([call(1)])
    four = rate([call(1000 * n) for n in range(1, 5)])
    ratio = four["tokens_per_second"] / one["tokens_per_second"]
    return {
        "pass": True,
        "one": one,
        "four": four,
        "ratio": ratio,
        "target": THROUGHPUT_TARGET,
        "target_missed": ratio < THROUGHPUT_TARGET,
    }


def disconnect(client, model):
    survivor = {}

    def keep():
        reply = complete(client, model, COUNTING, max_tokens=200, temperature=0)
        survivor["tokens"] = reply.usage.completion_tokens

    thread = threading.Thread(target=keep)
    thread.start()
    stream = complete(client, model, COUNTING, max_tokens=1800, temperature=0, stream=True)
    for index, _ in enumerate(stream):
        if index == 4:
            stream.close()
            break
    thread.join()
    begin = time.monotonic()
    after = concurrently(
        [lambda p=p: complete(client, model, p, max_tokens=8, temperature=0) for p in PROMPTS[:3]]
    )
    elapsed = time.monotonic() - begin
    passed = survivor.get("tokens", 0) > 0 and all(r.usage.completion_tokens > 0 for r in after)
    return {"pass": passed, "survivor_tokens": survivor.get("tokens"), "next_requests_s": elapsed}


def features(client, model):
    thinking_off = {"chat_template_kwargs": {"enable_thinking": False}}
    background = threading.Thread(
        target=lambda: complete(client, model, COUNTING, max_tokens=800, temperature=0)
    )
    background.start()
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    def json_case():
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Name a city in Brazil as JSON."}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": schema},
            },
            max_tokens=48,
        )
        value = json.loads(reply.choices[0].message.content)
        return set(value) == {"city"}

    def tool_case():
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Recife?"}],
            tools=[tool],
            tool_choice="required",
            max_tokens=96,
            extra_body=thinking_off,
        )
        calls = reply.choices[0].message.tool_calls or []
        return bool(calls) and "city" in json.loads(calls[0].function.arguments)

    def stop_case():
        reply = complete(client, model, "1, 2, 3, 4,", max_tokens=32, temperature=0, stop=["7"])
        return "7" not in reply.choices[0].text and reply.choices[0].finish_reason == "stop"

    json_ok, tool_ok, stop_ok = concurrently([json_case, tool_case, stop_case])
    background.join()
    return {"pass": json_ok and tool_ok and stop_ok, "json": json_ok, "tools": tool_ok, "stop": stop_ok}


CASES = {
    "short_beats_long": short_beats_long,
    "isolation": isolation,
    "throughput": throughput,
    "disconnect": disconnect,
    "features": features,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", choices=sorted(CASES), action="append")
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="unused", timeout=300, max_retries=0)
    report = {}
    for name in args.only or CASES:
        try:
            report[name] = CASES[name](client, args.model)
        except Exception as error:  # A failed case must not hide the others.
            report[name] = {"pass": False, "error": repr(error)}
        print(name, "pass" if report[name]["pass"] else "FAIL", flush=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return 0 if all(case["pass"] for case in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

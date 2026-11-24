#!/usr/bin/env python3
"""Validates tool calling on a running server with the official OpenAI SDK.

For each tool_choice, reasoning setting and transport: asks for a call, checks
its name and arguments against the schema, returns a tool result and requires
a final text answer. Exits 0 only if every case passes. A model that declines
to call a tool under `auto` is reported, not failed: that is model quality.
"""

import argparse
import itertools
import json
import sys

import jsonschema
from openai import OpenAI

CALL_TOKENS = 512
ANSWER_TOKENS = 1024
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
TIME_SCHEMA = {
    "type": "object",
    "properties": {"timezone": {"type": "string"}},
    "required": ["timezone"],
    "additionalProperties": False,
}
SCHEMAS = {"get_weather": WEATHER_SCHEMA, "get_time": TIME_SCHEMA}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Calls {name.replace('_', ' ')}.",
            "parameters": schema,
        },
    }
    for name, schema in SCHEMAS.items()
]
NAMED = {"type": "function", "function": {"name": "get_weather"}}
ONE_CITY = "What is the weather in Lisbon right now? Use the tool."
TWO_CITIES = "Get the weather in Lisbon and in Tokyo. Call the tool once per city."
CASES = {
    "auto": ("auto", ONE_CITY, None),
    "required": ("required", ONE_CITY, None),
    "named": (NAMED, ONE_CITY, "get_weather"),
    "parallel": ("required", TWO_CITIES, None),
}


def read_stream(chunks):
    """(content, calls, finish_reason) rebuilt from streamed deltas."""
    content, calls, finish = [], {}, None
    for chunk in chunks:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        finish = choice.finish_reason or finish
        content.append(choice.delta.content or "")
        for delta in choice.delta.tool_calls or []:
            call = calls.setdefault(delta.index, {"id": None, "name": None, "arguments": ""})
            call["id"] = delta.id or call["id"]
            if delta.function:
                call["name"] = delta.function.name or call["name"]
                call["arguments"] += delta.function.arguments or ""
    return "".join(content), [calls[index] for index in sorted(calls)], finish


def request(client, args, messages, stream, thinking, choice, max_tokens):
    reply = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=TOOLS,
        tool_choice=choice,
        max_tokens=max_tokens,
        stream=stream,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    if stream:
        return read_stream(reply)
    choice = reply.choices[0]
    calls = [
        {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
        for c in choice.message.tool_calls or []
    ]
    return choice.message.content or "", calls, choice.finish_reason


def check_calls(calls, finish, wanted):
    if finish != "tool_calls":
        raise ValueError(f"finish_reason {finish!r} with calls")
    for call in calls:
        if wanted and call["name"] != wanted:
            raise ValueError(f"called {call['name']!r}, wanted {wanted!r}")
        if call["name"] not in SCHEMAS or not call["id"]:
            raise ValueError(f"bad call {call}")
        jsonschema.validate(json.loads(call["arguments"]), SCHEMAS[call["name"]])


def history(question, content, calls):
    turn = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in calls
        ],
    }
    results = [
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"result": "22 C, sunny"})}
        for call in calls
    ]
    return [{"role": "user", "content": question}, turn, *results]


def run_case(client, args, case, stream, thinking):
    choice, question, wanted = CASES[case]
    first = [{"role": "user", "content": question}]
    content, calls, finish = request(client, args, first, stream, thinking, choice, CALL_TOKENS)
    if not calls:
        if case == "auto" and content.strip():
            return "no call (model chose text)"
        raise ValueError(f"no call, finish_reason {finish!r}")
    check_calls(calls, finish, wanted)
    follow_up = history(question, content, calls)
    answer, more, finish = request(client, args, follow_up, stream, thinking, "none", ANSWER_TOKENS)
    if more or not answer.strip():
        raise ValueError(f"no final answer, finish_reason {finish!r}, calls {more}")
    return f"{len(calls)} call(s), answered"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="unused", timeout=300)
    failures = 0
    for case, stream, thinking in itertools.product(CASES, (False, True), (True, False)):
        label = f"{case:9} stream={stream!s:5} thinking={thinking!s:5}"
        try:
            print(f"ok   {label} {run_case(client, args, case, stream, thinking)}", flush=True)
        except Exception as error:  # Every failure is reported; the run continues.
            failures += 1
            print(f"FAIL {label} {error}", flush=True)
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

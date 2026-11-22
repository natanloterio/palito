// Run: PALITO_TEST_MODEL=models/SmolLM2-135M-Instruct-Q4_K_M.gguf npm test
'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert');
const { Palito, PalitoError } = require('..');

const MODEL = process.env.PALITO_TEST_MODEL;
const HELLO = [{ role: 'user', content: 'Say hello in French and Portuguese.' }];
let llm;

before(async () => {
  if (MODEL) llm = await Palito.open(MODEL, { device: 'cpu', contextTokens: 512 });
});
after(async () => llm && (await llm.close()));

test('streamed text equals the reply', { skip: !MODEL }, async () => {
  let streamed = '';
  const reply = await llm.chat(HELLO, { maxTokens: 24, onText: (t) => { streamed += t; } });
  assert.strictEqual(streamed, reply.text);
  assert.ok(reply.generatedTokens > 0 && reply.generatedTokens <= 24);
  let iterated = '';
  for await (const piece of llm.streamChat(HELLO, { maxTokens: 24 })) iterated += piece;
  assert.strictEqual(iterated, reply.text);
});

test('stop, seeded sampling, cancel and queueing', { skip: !MODEL }, async () => {
  const cut = await llm.complete('1, 2, 3, 4,', { maxTokens: 24, stop: [' 7'] });
  assert.strictEqual(cut.finishReason, 'stop');
  assert.ok(!cut.text.includes(' 7'));
  const options = { maxTokens: 16, temperature: 0.9, seed: 4 };
  const [a, b] = await Promise.all([llm.chat(HELLO, options), llm.chat(HELLO, options)]);
  assert.strictEqual(a.text, b.text);
  const cancelled = await llm.complete('Once upon a time', { onText: () => false });
  assert.strictEqual(cancelled.finishReason, 'cancelled');
  for await (const _ of llm.streamComplete('Once upon a time', { maxTokens: 200 })) break;
  assert.strictEqual((await llm.complete('Hi', { maxTokens: 2 })).generatedTokens, 2);
});

test('penalties, min-p and logit bias', { skip: !MODEL }, async () => {
  const ones = '1, 1, 1, 1, 1, 1,';
  assert.ok((await llm.complete(ones, { maxTokens: 4 })).text.startsWith(' 1'));
  const penalized = await llm.complete(ones, {
    maxTokens: 4, frequencyPenalty: 2, presencePenalty: 0.5, repeatPenalty: 1.1,
  });
  assert.ok(!penalized.text.startsWith(' 1'), penalized.text);
  const seeded = { maxTokens: 8, temperature: 1, minP: 0.05, seed: 3 };
  assert.strictEqual((await llm.complete(ones, seeded)).text, (await llm.complete(ones, seeded)).text);
  await assert.rejects(llm.complete(ones, { maxTokens: 2, logitBias: { 1000000000: 1 } }), PalitoError);
  await assert.rejects(llm.complete(ones, { maxTokens: 2, minP: 2 }), PalitoError);
});

test('errors', { skip: !MODEL }, async () => {
  await assert.rejects(Palito.open('/nonexistent.gguf'), PalitoError);
  await assert.rejects(llm.chat(HELLO, { temperature: -1 }), PalitoError);
  await assert.rejects(llm.complete('Hi', { onText: () => { throw new RangeError('boom'); } }), RangeError);
});

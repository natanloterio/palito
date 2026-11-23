'use strict';
const assert = require('node:assert');
const test = require('node:test');
const manifest = require('../package.json');

const PREBUILT = '@koromix/koffi-linux-x64';

test('depends on the prebuilt koffi, which has no install script', () => {
  assert.strictEqual(manifest.dependencies.koffi, undefined);
  assert.match(manifest.dependencies[PREBUILT], /^\d+\.\d+\.\d+$/);
});

test('the prebuilt koffi exposes every call the binding uses', () => {
  const koffi = require(PREBUILT);
  for (const name of ['load', 'struct', 'proto', 'pointer', 'decode', 'register', 'unregister']) {
    assert.strictEqual(typeof koffi[name], 'function', name);
  }
});

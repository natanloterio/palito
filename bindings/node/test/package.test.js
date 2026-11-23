'use strict';
const assert = require('node:assert');
const test = require('node:test');
const manifest = require('../package.json');
const { koffiPackage, loadKoffi } = require('../koffi.js');

const prebuilt = manifest.optionalDependencies || {};

test('uses optional prebuilt koffi packages, which have no install script', () => {
  assert.strictEqual((manifest.dependencies || {}).koffi, undefined);
  for (const name of ['linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64', 'win32-arm64']) {
    assert.ok(prebuilt[`@koromix/koffi-${name}`], name);
  }
});

test('prebuilt packages share one compatible version range', () => {
  const ranges = new Set(Object.values(prebuilt));
  assert.strictEqual(ranges.size, 1);
  assert.match([...ranges][0], /^\^\d+\.\d+\.\d+$/);
});

test('publishes the koffi loader', () => {
  assert.ok(manifest.files.includes('koffi.js'));
});

test('picks the package for the platform', () => {
  assert.strictEqual(koffiPackage('darwin', 'arm64'), '@koromix/koffi-darwin-arm64');
});

test('an unsupported platform gets a clear error when loading', () => {
  const missing = () => { throw new Error('Cannot find module'); };
  assert.throws(() => loadKoffi('plan9', 'mips', missing), /plan9-mips/);
});

test('the prebuilt koffi exposes every call the binding uses', () => {
  const koffi = loadKoffi();
  for (const name of ['load', 'struct', 'proto', 'pointer', 'decode', 'register', 'unregister']) {
    assert.strictEqual(typeof koffi[name], 'function', name);
  }
});

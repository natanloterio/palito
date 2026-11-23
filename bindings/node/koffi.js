'use strict';
// Loads koffi's prebuilt package for this platform. Palito depends on these
// directly because the `koffi` package's install script fails pnpm's default install.

function koffiPackage(platform, arch) {
  return `@koromix/koffi-${platform}-${arch}`;
}

function loadKoffi(platform = process.platform, arch = process.arch, load = require) {
  const name = koffiPackage(platform, arch);
  try {
    return load(name);
  } catch (error) {
    throw new Error(`no prebuilt koffi for ${platform}-${arch} (${name}): ${error.message}`);
  }
}

module.exports = { koffiPackage, loadKoffi };

// usage: node example.js model.gguf "message"
'use strict';
const { Palito, version } = require('.');

(async () => {
  const [model, message] = process.argv.slice(2);
  if (!message) throw new Error('usage: node example.js model.gguf "message"');
  const llm = await Palito.open(model);
  console.error(`palito ${version()}, ${llm.profile}, context ${llm.contextTokens}`);
  for await (const text of llm.streamChat([{ role: 'user', content: message }], { thinking: false })) {
    process.stdout.write(text);
  }
  process.stdout.write('\n');
  await llm.close();
})().catch((error) => { console.error(error.message); process.exit(1); });

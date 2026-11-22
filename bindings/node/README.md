# palito (Node.js)

Node.js binding for [Palito](../../README.md), built on [koffi](https://koffi.dev). Generation runs on a worker thread, so the event loop stays free.

```sh
npm install ./bindings/node
```

```js
const { Palito } = require('palito');

const llm = await Palito.open('model.gguf');
const reply = await llm.chat([{ role: 'user', content: 'Hello!' }], {
  onText: (text) => process.stdout.write(text),
});
for await (const text of llm.streamChat([{ role: 'user', content: 'A haiku about tea' }],
                                        { temperature: 0.7, stop: ['\n\n'] })) {
  process.stdout.write(text);
}
await llm.close();
```

- **Types:** included (`index.d.ts`).
- **Concurrency:** calls on one engine are queued.
- **Stopping early:** a `break` in `for await` stops generation.

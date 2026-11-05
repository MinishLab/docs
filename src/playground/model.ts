import init, { Model } from './wasm/m2v_wasm.js';

const FILES = ['tokenizer.json', 'model.safetensors', 'config.json'];

// Downloads a model2vec model from Hugging Face once (kept in the Cache API) and loads it in WASM.
export async function loadModel(name: string, onProgress: (done: number, total: number) => void) {
  await init();
  const cache = await caches.open('minish-playground').catch(() => undefined);
  const responses = await Promise.all(
    FILES.map(async (file) => {
      const url = `https://huggingface.co/minishlab/${name}/resolve/main/${file}`;
      const cached = await cache?.match(url);
      if (cached) return cached;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not download ${name}/${file} (${response.status})`);
      await cache?.put(url, response.clone()).catch(() => {});
      return response;
    }),
  );
  const total = responses.reduce((n, r) => n + Number(r.headers.get('content-length') ?? 0), 0);
  let done = 0;
  const [tokenizer, weights, config] = await Promise.all(
    responses.map(async (response) => {
      const reader = response.body!.getReader();
      const parts: Uint8Array[] = [];
      for (let r = await reader.read(); !r.done; r = await reader.read()) {
        parts.push(r.value);
        onProgress((done += r.value.length), Math.max(total, done));
      }
      return concat(parts, Uint8Array);
    }),
  );
  return new Model(tokenizer, weights, config);
}

export function concat<T extends Uint8Array | Float32Array>(parts: T[], Type: { new (length: number): T }) {
  const out = new Type(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((offset, p) => (out.set(p, offset), offset + p.length), 0);
  return out;
}

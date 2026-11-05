import type { Model } from './wasm/m2v_wasm.js';
import { loadModel } from './model';

const MODEL_NAME = 'potion-base-8M';

export type Request =
  | { type: 'similarity'; id: number; source: string; sentences: string[] }
  | { type: 'dedupe'; id: number; rows: string[]; threshold: number };
export type Duplicate = { row: number; of: number; score: number; exact: boolean };
export type Response =
  | { type: 'progress'; done: number; total: number }
  | { type: 'similarity'; id: number; scores: number[]; dim: number; ms: number }
  | { type: 'dedupe'; id: number; selected: number[]; duplicates: Duplicate[]; embedMs: number; dedupeMs: number }
  | { type: 'error'; id: number; message: string };

const post = (message: Response) => postMessage(message);
let model: Promise<Model> | undefined;
// The last dedupe input and its embeddings, so moving the threshold does not re-embed.
let cached: { key: string; vectors: Float32Array; dim: number; ms: number } | undefined;

function getModel() {
  model ??= loadModel(MODEL_NAME, (done, total) => post({ type: 'progress', done, total })).catch((error) => {
    model = undefined;
    throw error;
  });
  return model;
}

async function embed(texts: string[]) {
  const start = performance.now();
  const vectors = (await getModel()).encode(texts);
  return { vectors, dim: texts.length ? vectors.length / texts.length : 0, ms: performance.now() - start };
}

// Vectors are normalized, so the dot product is the cosine similarity.
function dot(vectors: Float32Array, dim: number, a: number, b: number) {
  let sum = 0;
  for (let i = 0; i < dim; i++) sum += vectors[a * dim + i] * vectors[b * dim + i];
  return sum;
}

async function similarity({ id, source, sentences }: Extract<Request, { type: 'similarity' }>) {
  const { vectors, dim, ms } = await embed([source, ...sentences]);
  const scores = sentences.map((_, i) => dot(vectors, dim, 0, i + 1));
  post({ type: 'similarity', id, scores, dim, ms });
}

// Same rule as semhash's self_deduplicate: rows are kept in input order, and a row is a duplicate
// of the most similar kept row when that similarity reaches the threshold. Exact copies go first.
async function dedupe({ id, rows, threshold }: Extract<Request, { type: 'dedupe' }>) {
  const firstIndex = new Map<string, number>();
  const exact: Duplicate[] = [];
  rows.forEach((row, i) => {
    const first = firstIndex.get(row);
    if (first === undefined) firstIndex.set(row, i);
    else exact.push({ row: i, of: first, score: 1, exact: true });
  });
  const unique = [...firstIndex.keys()];
  const key = unique.join('\n');
  if (cached?.key !== key) cached = { key, ...(await embed(unique)) };

  const start = performance.now();
  const { vectors, dim } = cached;
  const selected: number[] = [];
  const duplicates: Duplicate[] = [];
  unique.forEach((row, i) => {
    let best = -1;
    let bestScore = -Infinity;
    for (const j of selected) {
      const score = dot(vectors, dim, i, j);
      if (score > bestScore) [best, bestScore] = [j, score];
    }
    if (best >= 0 && bestScore >= threshold) {
      duplicates.push({ row: firstIndex.get(row)!, of: firstIndex.get(unique[best])!, score: bestScore, exact: false });
    } else selected.push(i);
  });
  // Exact copies of a row that turned out to be a near duplicate point at the row it duplicates.
  const nearOf = new Map(duplicates.map((d) => [d.row, d]));
  const resolved = exact.map((d) => {
    const near = nearOf.get(d.of);
    return near ? { ...d, of: near.of, score: near.score, exact: false } : d;
  });
  post({
    type: 'dedupe',
    id,
    selected: selected.map((i) => firstIndex.get(unique[i])!),
    duplicates: [...resolved, ...duplicates],
    embedMs: cached.ms,
    dedupeMs: performance.now() - start,
  });
}

onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    if (data.type === 'similarity') await similarity(data);
    else await dedupe(data);
  } catch (error) {
    post({ type: 'error', id: data.id, message: error instanceof Error ? error.message : String(error) });
  }
};

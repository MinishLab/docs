import init, { Model } from './wasm/m2v_wasm.js';
import { chunkFile, Index, type Chunk } from './semble';

const MODEL_URL = 'https://huggingface.co/minishlab/potion-code-16M-v2/resolve/main';
const MAX_FILES = 3000;
const MAX_FILE_BYTES = 1_000_000;
const FETCH_CONCURRENCY = 64;
const EMBED_BATCH = 256;

// semble's code extensions (semble.index.files.get_extensions([ContentType.CODE])).
const CODE_EXTENSIONS = new Set(
  '4th ada adb ads agda al as asm astro awk axi axs bash bat bb bbappend bbclass bicep blade bq brs bsl bzl c c3 c3i c3t caddyfile cairo cbl cc cedar cel cfc chatito circom cjs ck cl clar clj cljc cljs cls cmake cmd cob cobol conf corn cpp cr cs cshtml css cst cts cu cuda cue cxx cylc d dart dhall dl dockerfile dot dsp eds eex el elm elv enforce eps erb erl ex exs f f03 f08 f90 f95 fc fidl filter fir fish fnl fs fsd fsi fsx fth fun g gd gdshader gi gleam glsl gn gni gnuplot go gotmpl gp gql gradle graphql gren groovy gv h hack hare hbs hcl heex hlsl hoon hpp hrl hs http hurl hx hxx idr inc ino ispc j2 jai janet java jinja2 jl jq js jsonnet jsx just k kt kts lc lds lean leex less libsonnet liquid lisp ll lua luau m magik makefile matlab meson mjs mk ml mli mlir mll mojo move mts nasm ncl nginx nim nims ninja nix nqc nu nut odin p pas php pkl pl plt pm pony pp prisma pro promql prql ps ps1 psd1 psm1 pug purs py pyi pyw ql qml r rasi razor rb rbs re rego res resi rkt robot roc rs s scad scala scm scss sh shtml sig slang smali smk sml sol sp sparql sql squirrel st stan star sv svelte svh sw swift tact tal tape tcl td templ tera tf tfvars tl tla trigger ts tsconfig tsx twig typoscript typst v vb verilog vhd vhdl vim vrl vue w wast wat wgsl wl yuck zig ziggy zsh'.split(' '),
);

export type Request = { type: 'index'; repo: string } | { type: 'search'; query: string };
export type Response =
  | { type: 'progress'; stage: 'model' | 'files' | 'embed'; done: number; total: number }
  | { type: 'indexed'; repo: string; files: number; chunks: number; requested: number; codeFiles: number; fetchMs: number; indexMs: number }
  | { type: 'results'; query: string; results: { path: string; startLine: number; endLine: number; content: string; score: number }[]; ms: number }
  | { type: 'error'; message: string };

const extension = (path: string) => {
  const name = path.slice(path.lastIndexOf('/') + 1);
  return name.includes('.') ? name.slice(name.lastIndexOf('.') + 1).toLowerCase() : '';
};
const post = (message: Response) => postMessage(message);
let model: Promise<Model> | undefined;
let index: Index | undefined;

async function cachedBytes(url: string, onBytes: (n: number) => void) {
  const cache = await caches.open('minish-playground').catch(() => undefined);
  let response = await cache?.match(url);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) throw new Error(`Could not download ${url.split('/').pop()} (${response.status})`);
    await cache?.put(url, response.clone()).catch(() => {});
  }
  const reader = response.body!.getReader();
  const parts: Uint8Array[] = [];
  for (let r = await reader.read(); !r.done; r = await reader.read()) {
    parts.push(r.value);
    onBytes(r.value.length);
  }
  const bytes = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((offset, p) => (bytes.set(p, offset), offset + p.length), 0);
  return bytes;
}

async function loadModel() {
  await init();
  const total = 32_490_072 + 1_024_340;
  let done = 0;
  const onBytes = (n: number) => post({ type: 'progress', stage: 'model', done: (done += n), total });
  const [tokenizer, weights, config] = await Promise.all(
    ['tokenizer.json', 'model.safetensors', 'config.json'].map((f) => cachedBytes(`${MODEL_URL}/${f}`, onBytes)),
  );
  return new Model(tokenizer, weights, config);
}

async function fetchRepo(repo: string) {
  const treeResponse = await fetch(`https://api.github.com/repos/${repo}/git/trees/HEAD?recursive=1`);
  if (treeResponse.status === 404) throw new Error(`Could not find ${repo} on GitHub (is it public?)`);
  if (treeResponse.status === 403) throw new Error('GitHub rate limit reached, try again in a few minutes');
  if (!treeResponse.ok) throw new Error(`GitHub returned ${treeResponse.status}`);
  const tree: { tree: { path: string; type: string; size?: number }[] } = await treeResponse.json();

  const code = tree.tree.filter(
    (e) =>
      e.type === 'blob' &&
      (e.size ?? 0) > 0 &&
      (e.size ?? 0) <= MAX_FILE_BYTES &&
      CODE_EXTENSIONS.has(extension(e.path)),
  );
  const paths = code.slice(0, MAX_FILES).map((e) => e.path);
  const files: { path: string; text: string }[] = [];
  let next = 0;
  const worker = async () => {
    while (next < paths.length) {
      const path = paths[next++];
      const response = await fetch(`https://raw.githubusercontent.com/${repo}/HEAD/${path.split('/').map(encodeURIComponent).join('/')}`);
      if (response.ok) files.push({ path, text: await response.text() });
      post({ type: 'progress', stage: 'files', done: files.length, total: paths.length });
    }
  };
  await Promise.all(Array.from({ length: FETCH_CONCURRENCY }, worker));
  return { files, requested: paths.length, codeFiles: code.length };
}

async function buildIndex(repo: string) {
  const m = await (model ??= loadModel().catch((error) => {
    model = undefined;
    throw error;
  }));
  let start = performance.now();
  const { files, requested, codeFiles } = await fetchRepo(repo);
  const fetchMs = performance.now() - start;

  start = performance.now();
  const chunks: Chunk[] = files.flatMap((f) => chunkFile(f.path, f.text));
  const vectors: Float32Array[] = [];
  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    vectors.push(m.encode(chunks.slice(i, i + EMBED_BATCH).map((c) => c.content)));
    post({ type: 'progress', stage: 'embed', done: Math.min(i + EMBED_BATCH, chunks.length), total: chunks.length });
  }
  const embeddings = new Float32Array(vectors.reduce((n, v) => n + v.length, 0));
  vectors.reduce((offset, v) => (embeddings.set(v, offset), offset + v.length), 0);
  const dim = chunks.length ? embeddings.length / chunks.length : 0;
  index = new Index(chunks, embeddings, dim);
  const indexMs = performance.now() - start;

  post({ type: 'indexed', repo, files: files.length, chunks: chunks.length, requested, codeFiles, fetchMs, indexMs });
}

async function search(query: string) {
  if (!index || !query.trim()) return post({ type: 'results', query, results: [], ms: 0 });
  const start = performance.now();
  const embedding = (await model!).encode([query]);
  const results = index.search(query, embedding).map(({ chunk, score }) => ({ ...chunk, score }));
  post({ type: 'results', query, results, ms: performance.now() - start });
}

onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    if (data.type === 'index') await buildIndex(data.repo);
    else await search(data.query);
  } catch (error) {
    post({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};

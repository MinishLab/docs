// A browser port of semble's chunking and hybrid ranking (github.com/MinishLab/semble), minus
// tree-sitter: files are chunked by line. Constants and boosts mirror semble's Python source.

export type Chunk = { path: string; content: string; startLine: number; endLine: number };
export type Result = { chunk: Chunk; score: number };

const CHUNK_CHARS = 750;
const K1 = 1.5;
const B = 0.75;
const RRF_K = 60;
const ALPHA_SYMBOL = 0.3;
const ALPHA_NL = 0.5;
const DEFINITION_BOOST = 3.0;
const EMBEDDED_SYMBOL_SCALE = 0.5;
const EMBEDDED_STEM_MIN_LEN = 4;
const STEM_BOOST = 1.0;
const FILE_COHERENCE_FRAC = 0.2;
const SATURATION_DECAY = 0.5;

const STOPWORDS = new Set(
  'a an and are as at be by do does for from has have how if in is it not of on or the to was what when where which who why with'.split(' '),
);
const DEFINITION_KEYWORDS = [
  'class', 'module', 'defmodule', 'def', 'interface', 'struct', 'enum', 'trait', 'type', 'func', 'function',
  'object', 'abstract class', 'data class', 'fn', 'fun', 'package', 'namespace', 'protocol', 'record', 'typedef',
];
const SQL_DEFINITION_KEYWORDS = ['CREATE TABLE', 'CREATE VIEW', 'CREATE PROCEDURE', 'CREATE FUNCTION'];

const SYMBOL_QUERY_RE = new RegExp(
  '^(?:' +
    '[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\\\|->|\\.)[A-Za-z_][A-Za-z0-9_]*)+' +
    '|_[A-Za-z0-9_]*' +
    '|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*' +
    '|[A-Z][A-Za-z0-9]*' +
    ')$',
);
const EMBEDDED_SYMBOL_RE = /\b(?:[A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*|[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+)\b/g;
const CAMEL_RE = /[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+/g;
const TOKEN_RE = /[a-zA-Z_][a-zA-Z0-9_]*/g;

const TEST_FILE_RE = new RegExp(
  '(?:^|/)(?:' +
    [
      'test_[^/]*\\.py', '[^/]*_test\\.py', '[^/]*_test\\.go', '[^/]*Tests?\\.java', '[^/]*Test\\.php',
      '[^/]*_spec\\.rb', '[^/]*_test\\.rb', '[^/]*\\.test\\.[jt]sx?', '[^/]*\\.spec\\.[jt]sx?', '[^/]*Tests?\\.kt',
      '[^/]*Spec\\.kt', '[^/]*Tests?\\.swift', '[^/]*Spec\\.swift', '[^/]*Tests?\\.cs', 'test_[^/]*\\.cpp',
      '[^/]*_test\\.cpp', 'test_[^/]*\\.c', '[^/]*_test\\.c', '[^/]*Spec\\.scala', '[^/]*Suite\\.scala',
      '[^/]*Test\\.scala', '[^/]*_test\\.dart', 'test_[^/]*\\.dart', '[^/]*_spec\\.lua', '[^/]*_test\\.lua',
      'test_[^/]*\\.lua', 'test_helpers?[^/]*\\.\\w+',
    ].join('|') +
    ')$',
);
const TEST_DIR_RE = /(?:^|\/)(?:tests?|__tests__|spec|testing)(?:\/|$)/;
const COMPAT_DIR_RE = /(?:^|\/)(?:compat|_compat|legacy)(?:\/|$)/;
const EXAMPLES_DIR_RE = /(?:^|\/)(?:_?examples?|docs?_src)(?:\/|$)/;
const REEXPORT_FILENAMES = new Set(['__init__.py', 'package-info.java']);

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const fileName = (path: string) => path.slice(path.lastIndexOf('/') + 1);
const stemOf = (path: string) => fileName(path).replace(/\.[^.]*$/, '');
const dirParts = (path: string) => path.split('/').slice(0, -1);

export function chunkFile(path: string, source: string): Chunk[] {
  if (!source.trim()) return [];
  const lines = source.split(/(?<=\n)/);
  const chunks: Chunk[] = [];
  let content = '';
  let startLine = 1;
  lines.forEach((line, i) => {
    if (content && content.length + line.length > CHUNK_CHARS) {
      chunks.push({ path, content, startLine, endLine: i });
      content = '';
      startLine = i + 1;
    }
    content += line;
  });
  chunks.push({ path, content, startLine, endLine: lines.length });
  return chunks;
}

function splitIdentifier(token: string): string[] {
  const lower = token.toLowerCase();
  const parts = token.includes('_')
    ? lower.split('_').filter(Boolean)
    : (token.match(CAMEL_RE) ?? []).map((p) => p.toLowerCase());
  return parts.length >= 2 ? [lower, ...parts] : [lower];
}

const tokenize = (text: string) => (text.match(TOKEN_RE) ?? []).flatMap(splitIdentifier);

// Path components are appended so path-based queries match, with the stem twice to up-weight it.
const bm25Text = (chunk: Chunk) => {
  const stem = stemOf(chunk.path);
  return `${chunk.content} ${stem} ${stem} ${dirParts(chunk.path).slice(-3).join(' ')}`;
};

export class Index {
  private postings = new Map<string, { docs: number[]; tfs: number[] }>();
  private lengths: Int32Array;
  private avgLength: number;

  constructor(
    readonly chunks: Chunk[],
    private embeddings: Float32Array,
    private dim: number,
  ) {
    this.lengths = new Int32Array(chunks.length);
    chunks.forEach((chunk, doc) => {
      const counts = new Map<string, number>();
      for (const token of tokenize(bm25Text(chunk))) counts.set(token, (counts.get(token) ?? 0) + 1);
      for (const [term, tf] of counts) {
        let posting = this.postings.get(term);
        if (!posting) this.postings.set(term, (posting = { docs: [], tfs: [] }));
        posting.docs.push(doc);
        posting.tfs.push(tf);
        this.lengths[doc] += tf;
      }
    });
    this.avgLength = this.lengths.reduce((a, b) => a + b, 0) / Math.max(chunks.length, 1);
  }

  search(query: string, queryEmbedding: Float32Array, topK = 10): Result[] {
    const symbol = SYMBOL_QUERY_RE.test(query.trim());
    const alpha = symbol ? ALPHA_SYMBOL : ALPHA_NL;
    const candidates = topK * 5;

    const semantic = rrf(topIndices(this.cosine(queryEmbedding), candidates));
    const bm25 = rrf(topIndices(this.bm25(tokenize(query)), candidates, 0));
    const combined = new Map<number, number>();
    for (const doc of new Set([...semantic.keys(), ...bm25.keys()])) {
      const score = alpha * (semantic.get(doc) ?? 0) + (1 - alpha) * (bm25.get(doc) ?? 0);
      if (score) combined.set(doc, score);
    }
    this.boostMultiChunkFiles(combined);
    this.applyQueryBoost(combined, query, symbol);
    return this.rerank(combined, topK);
  }

  private cosine(query: Float32Array) {
    const scores = new Float32Array(this.chunks.length);
    for (let doc = 0; doc < scores.length; doc++) {
      let dot = 0;
      for (let i = 0, offset = doc * this.dim; i < this.dim; i++) dot += query[i] * this.embeddings[offset + i];
      scores[doc] = dot;
    }
    return scores;
  }

  private bm25(tokens: string[]) {
    const scores = new Float32Array(this.chunks.length);
    const n = this.chunks.length;
    const queryCounts = new Map<string, number>();
    for (const token of tokens) queryCounts.set(token, (queryCounts.get(token) ?? 0) + 1);
    for (const [term, queryTf] of queryCounts) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      const df = posting.docs.length;
      const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
      posting.docs.forEach((doc, i) => {
        const tf = posting.tfs[i];
        scores[doc] += (queryTf * idf * tf) / (K1 * (1 - B + (B * this.lengths[doc]) / this.avgLength) + tf);
      });
    }
    return scores;
  }

  // Promote files with several relevant chunks by boosting their best chunk.
  private boostMultiChunkFiles(scores: Map<number, number>) {
    const max = Math.max(0, ...scores.values());
    if (max === 0) return;
    const fileSum = new Map<string, number>();
    const best = new Map<string, number>();
    for (const [doc, score] of scores) {
      const path = this.chunks[doc].path;
      fileSum.set(path, (fileSum.get(path) ?? 0) + score);
      if (!best.has(path) || score > scores.get(best.get(path)!)!) best.set(path, doc);
    }
    const maxFileSum = Math.max(...fileSum.values());
    for (const [path, doc] of best) {
      scores.set(doc, scores.get(doc)! + (max * FILE_COHERENCE_FRAC * fileSum.get(path)!) / maxFileSum);
    }
  }

  private applyQueryBoost(scores: Map<number, number>, query: string, symbol: boolean) {
    if (!scores.size) return;
    const max = Math.max(...scores.values());
    if (symbol) {
      const trimmed = query.trim();
      const name = extractSymbolName(trimmed);
      const names = new Set([name, trimmed]);
      this.boostDefinitions(scores, names, max * DEFINITION_BOOST, (stem) => stemMatches(stem, name.toLowerCase()));
      return;
    }
    this.boostStemMatches(scores, query, max);
    const embedded = new Set(query.match(EMBEDDED_SYMBOL_RE) ?? []);
    if (!embedded.size) return;
    const lowered = [...embedded].map((s) => s.toLowerCase());
    this.boostDefinitions(scores, embedded, max * DEFINITION_BOOST * EMBEDDED_SYMBOL_SCALE, (stem) => {
      const norm = stem.replaceAll('_', '');
      return lowered.some(
        (s) =>
          stem === s ||
          norm === s ||
          (stem.length >= EMBEDDED_STEM_MIN_LEN && s.startsWith(stem)) ||
          (norm.length >= EMBEDDED_STEM_MIN_LEN && s.startsWith(norm)),
      );
    });
  }

  // Boost candidates that define one of names, then pull in non-candidates whose file stem fits.
  private boostDefinitions(scores: Map<number, number>, names: Set<string>, unit: number, stemOk: (stem: string) => boolean) {
    const patterns = [...names].flatMap(definitionPatterns);
    const tier = (doc: number) => {
      const chunk = this.chunks[doc];
      if (!patterns.some((re) => re.test(chunk.content))) return 0;
      const stem = stemOf(chunk.path).toLowerCase();
      return unit * ([...names].some((n) => stemMatches(stem, n.toLowerCase())) ? 1.5 : 1);
    };
    for (const doc of [...scores.keys()]) {
      const boost = tier(doc);
      if (boost) scores.set(doc, scores.get(doc)! + boost);
    }
    this.chunks.forEach((chunk, doc) => {
      if (scores.has(doc) || !stemOk(stemOf(chunk.path).toLowerCase())) return;
      const boost = tier(doc);
      if (boost) scores.set(doc, boost);
    });
  }

  // Boost chunks whose file stem or parent directory matches words in a natural-language query.
  private boostStemMatches(scores: Map<number, number>, query: string, max: number) {
    const keywords = new Set(
      (query.match(TOKEN_RE) ?? []).map((w) => w.toLowerCase()).filter((w) => w.length > 2 && !STOPWORDS.has(w)),
    );
    if (!keywords.size) return;
    for (const doc of [...scores.keys()]) {
      const path = this.chunks[doc].path;
      const parts = new Set(splitIdentifier(stemOf(path)));
      const parent = dirParts(path).at(-1);
      if (parent) splitIdentifier(parent).forEach((p) => parts.add(p));
      let matches = 0;
      for (const keyword of keywords) {
        if (parts.has(keyword)) matches++;
        else if ([...parts].some((part) => prefixOverlap(keyword, part))) matches++;
      }
      const ratio = matches / keywords.size;
      if (ratio >= 0.1) scores.set(doc, scores.get(doc)! + max * STEM_BOOST * ratio);
    }
  }

  // Path penalties (tests, compat, examples, re-exports), then halve each extra chunk from the same file.
  private rerank(scores: Map<number, number>, topK: number): Result[] {
    const penalised = [...scores].map(([doc, score]) => [doc, score * pathPenalty(this.chunks[doc].path)] as const);
    penalised.sort((a, b) => b[1] - a[1]);
    const perFile = new Map<string, number>();
    const selected: Result[] = [];
    let minSelected = Infinity;
    for (const [doc, score] of penalised) {
      if (selected.length >= topK && score <= minSelected) break;
      const chunk = this.chunks[doc];
      const seen = perFile.get(chunk.path) ?? 0;
      selected.push({ chunk, score: score * SATURATION_DECAY ** seen });
      perFile.set(chunk.path, seen + 1);
      if (selected.length >= topK) minSelected = Math.min(...selected.map((r) => r.score));
    }
    return selected.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}

function topIndices(scores: Float32Array, k: number, min = -Infinity) {
  const docs = [];
  for (let doc = 0; doc < scores.length; doc++) if (scores[doc] > min) docs.push(doc);
  return docs.sort((a, b) => scores[b] - scores[a]).slice(0, k);
}

const rrf = (ranked: number[]) => new Map(ranked.map((doc, rank) => [doc, 1 / (RRF_K + rank + 1)]));

function extractSymbolName(query: string) {
  for (const separator of ['::', '\\', '->', '.']) {
    if (query.includes(separator)) return query.slice(query.lastIndexOf(separator) + separator.length);
  }
  return query;
}

function definitionPatterns(name: string) {
  const suffix = `)\\s+(?:[A-Za-z_][A-Za-z0-9_]*(?:\\.|::))*${escapeRe(name)}(?:\\s|[<({:\\[;]|$)`;
  return [
    new RegExp(`(?:^|(?<=\\s))(?:${DEFINITION_KEYWORDS.map(escapeRe).join('|')}${suffix}`, 'm'),
    new RegExp(`(?:^|(?<=\\s))(?:${SQL_DEFINITION_KEYWORDS.map(escapeRe).join('|')}${suffix}`, 'mi'),
  ];
}

function stemMatches(stem: string, name: string) {
  const norm = stem.replaceAll('_', '');
  const trim = (s: string) => s.replace(/s+$/, '');
  return stem === name || norm === name || trim(stem) === name || trim(norm) === name;
}

function prefixOverlap(a: string, b: string) {
  const [shorter, longer] = a.length <= b.length ? [a, b] : [b, a];
  return shorter.length >= 3 && longer.startsWith(shorter);
}

function pathPenalty(path: string) {
  let penalty = 1;
  if (TEST_FILE_RE.test(path) || TEST_DIR_RE.test(path)) penalty *= 0.3;
  if (REEXPORT_FILENAMES.has(fileName(path))) penalty *= 0.5;
  if (COMPAT_DIR_RE.test(path)) penalty *= 0.3;
  if (EXAMPLES_DIR_RE.test(path)) penalty *= 0.3;
  if (path.endsWith('.d.ts')) penalty *= 0.7;
  return penalty;
}

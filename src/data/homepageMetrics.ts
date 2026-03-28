export type DownloadSource =
  | { type: 'pypi'; name: string }
  | { type: 'crates'; name: string };

export type HomepagePackageDefinition = {
  name: string;
  description: string;
  docsHref: string;
  logoSrc: string;
  logoAlt: string;
  githubHref: string;
  githubRepo: string;
  downloadsHref: string;
  downloadSource: DownloadSource;
  fallbackStars: number;
  fallbackDownloads: number;
};

export type HomepagePackageMetrics = HomepagePackageDefinition & {
  stars: number;
  downloads: number;
};

export const homepagePackages: HomepagePackageDefinition[] = [
  {
    name: 'Model2Vec',
    description: 'State-of-the-art static embeddings distilled from any sentence transformer.',
    docsHref: '/packages/model2vec/introduction/',
    logoSrc: '/images/logos/model2vec_logo.webp',
    logoAlt: 'Model2Vec',
    githubHref: 'https://github.com/minishlab/model2vec',
    githubRepo: 'minishlab/model2vec',
    downloadsHref: 'https://pypi.org/project/model2vec/',
    downloadSource: { type: 'pypi', name: 'model2vec' },
    fallbackStars: 2000,
    fallbackDownloads: 2200000,
  },
  {
    name: 'SemHash',
    description: 'Multimodal semantic deduplication, outlier filtering, and representative sampling.',
    docsHref: '/packages/semhash/introduction/',
    logoSrc: '/images/logos/semhash_logo.webp',
    logoAlt: 'SemHash',
    githubHref: 'https://github.com/minishlab/semhash',
    githubRepo: 'minishlab/semhash',
    downloadsHref: 'https://pypi.org/project/semhash/',
    downloadSource: { type: 'pypi', name: 'semhash' },
    fallbackStars: 909,
    fallbackDownloads: 264000,
  },
  {
    name: 'Vicinity',
    description: 'Flexible nearest neighbor search with multiple backends and evaluation.',
    docsHref: '/packages/vicinity/introduction/',
    logoSrc: '/images/logos/vicinity_logo.webp',
    logoAlt: 'Vicinity',
    githubHref: 'https://github.com/minishlab/vicinity',
    githubRepo: 'minishlab/vicinity',
    downloadsHref: 'https://pypi.org/project/vicinity/',
    downloadSource: { type: 'pypi', name: 'vicinity' },
    fallbackStars: 336,
    fallbackDownloads: 275000,
  },
  {
    name: 'Tokenlearn',
    description: 'Pre-train Model2Vec models on large corpora with efficient embedding distillation.',
    docsHref: '/packages/tokenlearn/usage/',
    logoSrc: '/images/logos/tokenlearn_logo.webp',
    logoAlt: 'Tokenlearn',
    githubHref: 'https://github.com/minishlab/tokenlearn',
    githubRepo: 'minishlab/tokenlearn',
    downloadsHref: 'https://pypi.org/project/tokenlearn/',
    downloadSource: { type: 'pypi', name: 'tokenlearn' },
    fallbackStars: 95,
    fallbackDownloads: 6400,
  },
  {
    name: 'Model2Vec-rs',
    description: 'High-performance Rust inference for Model2Vec models with low-overhead deployment.',
    docsHref: '/packages/model2vec-rs/usage/',
    logoSrc: '/images/logos/model2vec_rs_logo.webp',
    logoAlt: 'Model2Vec-rs',
    githubHref: 'https://github.com/minishlab/model2vec-rs',
    githubRepo: 'minishlab/model2vec-rs',
    downloadsHref: 'https://crates.io/crates/model2vec-rs',
    downloadSource: { type: 'crates', name: 'model2vec-rs' },
    fallbackStars: 164,
    fallbackDownloads: 21000,
  },
];

type GithubRepoResponse = {
  stargazers_count?: number;
};

type CratesIoResponse = {
  crate?: {
    downloads?: number;
  };
};

type PepyResponse = {
  total_downloads?: number;
};

const PEPY_API_KEY = import.meta.env.PEPY_API_KEY;
const GITHUB_TOKEN = import.meta.env.MINISH_GITHUB_TOKEN;
const REQUEST_HEADERS = {
  Accept: 'application/json',
  'User-Agent': 'minish-docs-build',
};

let homepageMetricsPromise: Promise<HomepagePackageMetrics[]> | undefined;

async function fetchJson<T>(url: string, extraHeaders: Record<string, string> = {}) {
  const response = await fetch(url, {
    headers: {
      ...REQUEST_HEADERS,
      ...extraHeaders,
    },
  });

  if (!response.ok) {
    throw new Error(`Request failed for ${url}: ${response.status}`);
  }

  return (await response.json()) as T;
}

async function fetchGithubStars(githubRepo: string) {
  const data = await fetchJson<GithubRepoResponse>(`https://api.github.com/repos/${githubRepo}`, {
    ...(GITHUB_TOKEN ? { Authorization: `Bearer ${GITHUB_TOKEN}` } : {}),
  });
  return typeof data.stargazers_count === 'number' ? data.stargazers_count : undefined;
}

async function fetchDownloads(source: DownloadSource) {
  if (source.type === 'crates') {
    const data = await fetchJson<CratesIoResponse>(`https://crates.io/api/v1/crates/${source.name}`);
    return typeof data.crate?.downloads === 'number' ? data.crate.downloads : undefined;
  }

  // PyPI does not expose all-time download counts, so Python package totals use Pepy when available.
  if (!PEPY_API_KEY) {
    return undefined;
  }

  const data = await fetchJson<PepyResponse>(
    `https://api.pepy.tech/api/v2/projects/${source.name}`,
    { 'X-API-Key': PEPY_API_KEY },
  );

  return typeof data.total_downloads === 'number' ? data.total_downloads : undefined;
}

async function resolvePackageMetrics(pkg: HomepagePackageDefinition): Promise<HomepagePackageMetrics> {
  const [stars, downloads] = await Promise.all([
    fetchGithubStars(pkg.githubRepo).catch(() => undefined),
    fetchDownloads(pkg.downloadSource).catch(() => undefined),
  ]);

  return {
    ...pkg,
    stars: stars ?? pkg.fallbackStars,
    downloads: downloads ?? pkg.fallbackDownloads,
  };
}

export async function getHomepagePackageMetrics() {
  homepageMetricsPromise ??= Promise.all(homepagePackages.map(resolvePackageMetrics));
  return homepageMetricsPromise;
}

export async function getHomepageTotals() {
  const packages = await getHomepagePackageMetrics();
  return packages.reduce(
    (totals, pkg) => ({
      stars: totals.stars + pkg.stars,
      downloads: totals.downloads + pkg.downloads,
    }),
    { stars: 0, downloads: 0 },
  );
}

export function formatCompactNumber(value: number) {
  if (value < 1000) {
    return `${value}`;
  }

  return new Intl.NumberFormat('en', {
    notation: 'compact',
    maximumFractionDigits: 1,
  })
    .format(value)
    .replace('K', 'k');
}

export function formatExactNumber(value: number) {
  return new Intl.NumberFormat('en').format(value);
}

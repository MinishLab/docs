import type { Request, Response } from './text-worker';

type Reply<T extends Request['type']> = Extract<Response, { type: T }>;

// One worker and one model shared by the Embed and Dedupe tabs, created on first use.
let worker: Worker | undefined;
let nextId = 0;
const pending = new Map<number, { resolve: (r: Response) => void; reject: (e: Error) => void }>();
const progressListeners = new Set<(done: number, total: number) => void>();

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./text-worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }: MessageEvent<Response>) => {
    if (data.type === 'progress') return progressListeners.forEach((listener) => listener(data.done, data.total));
    const request = pending.get(data.id);
    pending.delete(data.id);
    if (data.type === 'error') request?.reject(new Error(data.message));
    else request?.resolve(data);
  };
  return worker;
}

export function request<T extends Request['type']>(message: Omit<Extract<Request, { type: T }>, 'id'> & { type: T }) {
  const id = nextId++;
  return new Promise<Reply<T>>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (r: Response) => void, reject });
    getWorker().postMessage({ ...message, id });
  });
}

// Wires a tab to the shared model: download progress in its status line, a debounced update on
// every input, and a first update when the tab is first shown (which is when the model loads).
export function bindTab(root: HTMLElement, inputs: HTMLElement[], update: () => void) {
  const fill = root.querySelector<HTMLElement>('[data-fill]')!;
  const statusText = root.querySelector<HTMLElement>('[data-status-text]')!;
  progressListeners.add((done, total) => {
    fill.style.width = `${(done / total) * 100}%`;
    statusText.textContent = `Downloading potion-base-8M (once) ${Math.round(done / 1e6)}/${Math.round(total / 1e6)} MB`;
  });
  let timer: number | undefined;
  const schedule = () => {
    clearTimeout(timer);
    timer = window.setTimeout(update, 60);
  };
  inputs.forEach((input) => input.addEventListener('input', schedule));
  new IntersectionObserver((entries, observer) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    observer.disconnect();
    update();
  }).observe(root);
}

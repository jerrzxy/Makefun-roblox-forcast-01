import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// ---- Constants ----
const Q_MID_DAYS = 5.5;
const Q_SLOPE_DAYS = 0.5;
const Q_CAP = 0.85;
const Q_FLOOR = 0.02;

const SIGMA_TOP = 0.45;
const SIGMA_MID = 0.7;
const SIGMA_LOW = 1.3;

const UNNAMED_TOP = 0.08;
const UNNAMED_LOW = 0.12;
const MIN_P = 0.003;
const DROPOUT_RANK = 45;

const VEL_HOURS = 14;
const VEL_CAP = 8;
const VEL_W = 0.2;

const TAIL_MID = 0.15;
const TAIL_LOW = 0.225;
const TAIL_MAX_RANK = 45;
const TAIL_MIN_VEL = 3;

const MAX_LAST_RANK = 60;
const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

// Kernel width by place (index 0 unused; places 1..10)
const SIGMA_BY_PLACE: number[] = [
  0,
  SIGMA_TOP, // 1
  SIGMA_TOP, // 2
  0.5,       // 3
  0.55,      // 4
  SIGMA_MID, // 5
  0.8,       // 6
  0.9,       // 7
  SIGMA_LOW, // 8
  1.4,       // 9
  1.5,       // 10
];

function sigmaFor(place: number): number {
  const i = Math.min(Math.max(Math.round(place), 1), 10);
  return SIGMA_BY_PLACE[i];
}

function unnamedFor(place: number): number {
  return place <= 4 ? UNNAMED_TOP : UNNAMED_LOW;
}

// Share of each place's named mass given to the velocity tail.
function tailShare(place: number): number {
  const i = Math.round(place);
  if (i <= 4) return 0;
  if (i === 5) return TAIL_MID;
  if (i === 6) return 0.2;
  if (i === 7) return 0.25;
  if (i === 8) return TAIL_LOW;
  return 0.275; // places 9, 10
}

// ---- Types ----
interface TrendRow {
  rank: number;
  universe_id: number | string;
  [k: string]: unknown;
}

interface Snapshot {
  at: string;
  t: number;
  top_trending: TrendRow[];
}

interface Game {
  id: number | string;
  key: string;
  rank: number;
}

// ---- IO ----
const inputDir = process.env.MAKEFUN_INPUT_DIR || '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';

function readTask(): { places: number[]; output: string } {
  const raw = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8')) ?? {};

  let places: number[] = [];
  if (typeof raw.places === 'number') {
    // A count N means places 1..N.
    const n = Math.floor(raw.places);
    for (let i = 1; i <= n; i++) places.push(i);
  } else if (Array.isArray(raw.places)) {
    for (const p of raw.places) {
      let v: unknown = p;
      if (p && typeof p === 'object') {
        const o = p as Record<string, unknown>;
        v = o.place ?? o.rank ?? o.id;
      }
      const n = Number(v);
      if (Number.isFinite(n) && n >= 1) places.push(Math.round(n));
    }
  }
  if (places.length === 0) places = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  places = Array.from(new Set(places)).sort((a, b) => a - b);

  let output = 'forecast.json';
  const o = raw.output;
  if (typeof o === 'string' && o.length > 0) {
    output = o;
  } else if (o && typeof o === 'object') {
    const r = o as Record<string, unknown>;
    const cand = r.file ?? r.path ?? r.filename ?? r.name;
    if (typeof cand === 'string' && cand.length > 0) output = cand;
  }
  return { places, output };
}

function readHistory(): Snapshot[] {
  let text = '';
  try {
    text = readFileSync(join(inputDir, 'history.ndjson'), 'utf8');
  } catch {
    return [];
  }
  const snaps: Snapshot[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    let obj: any;
    try {
      obj = JSON.parse(s);
    } catch {
      continue;
    }
    if (!obj || typeof obj.at !== 'string' || !Array.isArray(obj.top_trending)) continue;
    const t = Date.parse(obj.at);
    if (!Number.isFinite(t)) continue;
    snaps.push({ at: obj.at, t, top_trending: obj.top_trending });
  }
  const indexed = snaps.map((s, i) => ({ s, i }));
  indexed.sort((a, b) => a.s.t - b.s.t || a.i - b.i);
  if (snaps.length === 0) throw new Error('empty history');
  return indexed.map((x) => x.s);
}

// ---- Helpers ----
function rankOne(snap: Snapshot): number | string | null {
  let best: TrendRow | null = null;
  for (const r of snap.top_trending) {
    if (r && r.rank === 1 && r.universe_id !== undefined && r.universe_id !== null) {
      best = r;
      break;
    }
  }
  return best ? best.universe_id : null;
}

function rankMap(snap: Snapshot): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of snap.top_trending) {
    if (!r || r.universe_id === undefined || r.universe_id === null) continue;
    const rank = Number(r.rank);
    if (!Number.isFinite(rank)) continue;
    const key = String(r.universe_id);
    const prev = m.get(key);
    if (prev === undefined || rank < prev) m.set(key, rank);
  }
  return m;
}

function round6(x: number): number {
  return Math.round(x * 1e6) / 1e6;
}

function gauss(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z);
}

// ---- Main ----
function main(): void {
  const task = readTask();
  const history = readHistory();

  mkdirSync(outputDir, { recursive: true });
  const outPath = join(outputDir, task.output);

  if (history.length === 0) {
    writeFileSync(outPath, JSON.stringify({ probabilities: [], top10: [] }));
    return;
  }

  const last = history[history.length - 1];

  // Games from the last read (rank <= 60), best rank per universe.
  const byKey = new Map<string, Game>();
  for (const r of last.top_trending) {
    if (!r || r.universe_id === undefined || r.universe_id === null) continue;
    const rank = Number(r.rank);
    if (!Number.isFinite(rank) || rank < 1 || rank > MAX_LAST_RANK) continue;
    const key = String(r.universe_id);
    const prev = byKey.get(key);
    if (!prev || rank < prev.rank) byKey.set(key, { id: r.universe_id, key, rank });
  }
  const games = Array.from(byKey.values());

  // Current #1 and its unbroken tenure at #1.
  const leaderId = rankOne(last);
  let q = 0;
  let leaderKey: string | null = null;
  if (leaderId !== null && byKey.has(String(leaderId))) {
    leaderKey = String(leaderId);
    let firstT = last.t;
    for (let i = history.length - 1; i >= 0; i--) {
      const id = rankOne(history[i]);
      if (id !== null && String(id) === leaderKey) firstT = history[i].t;
      else break;
    }
    const tenureDays = (last.t - firstT) / DAY_MS;
    const raw = 1 / (1 + Math.exp(-(tenureDays - Q_MID_DAYS) / Q_SLOPE_DAYS));
    q = Math.min(Q_CAP, Math.max(Q_FLOOR, raw));
  }

  // Reference read: latest snapshot at or before (last read time - VEL_HOURS)
  // in wall-clock time; if none, the first snapshot.
  const refCutoff = last.t - VEL_HOURS * HOUR_MS;
  let refSnap: Snapshot = history[0];
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].t <= refCutoff) {
      refSnap = history[i];
      break;
    }
  }
  const refRank = rankMap(refSnap);

  const feats = (g: Game): { r: number; v: number } => {
    const r = g.rank;
    const rr = refRank.get(g.key);
    const v = rr === undefined ? 0 : rr - r;
    return { r, v };
  };

  // Velocity-adjusted expected ranks.
  const stayMu = (g: Game): number => {
    const { r, v } = feats(g);
    const vel = Math.min(Math.max(v, 0), VEL_CAP);
    return Math.max(1, r - VEL_W * vel);
  };
  const dropMu = (g: Game): number | null => {
    if (g.key === leaderKey) return null;
    const { r, v } = feats(g);
    const vel = Math.min(Math.max(v, 0), VEL_CAP);
    return Math.max(1, r - 1 - VEL_W * vel);
  };

  const probabilities: { universe_id: number | string; place: number; p: number }[] = [];

  for (const place of task.places) {
    const sigma = sigmaFor(place);
    const unnamed = unnamedFor(place);

    // Kernel weights: mixture of the stay and dropout scenarios.
    const kernel = new Map<string, number>();
    const addK = (key: string, v: number) => kernel.set(key, (kernel.get(key) ?? 0) + v);

    // Stay scenario
    {
      const w = leaderKey === null ? 1 : 1 - q;
      const raws = games.map((g) => gauss(place, stayMu(g), sigma));
      const sum = raws.reduce((a, b) => a + b, 0);
      if (sum > 0 && w > 0) {
        games.forEach((g, i) => addK(g.key, w * (raws[i] / sum)));
      }
    }

    // Dropout scenario
    if (leaderKey !== null && q > 0) {
      const raws = games.map((g) => {
        const mu = dropMu(g);
        return mu === null ? 0 : gauss(place, mu, sigma);
      });
      const sum = raws.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        games.forEach((g, i) => {
          if (raws[i] > 0) addK(g.key, q * (raws[i] / sum));
        });
      }
      // The old #1 has no shifted rank; it sometimes only slips to #2 or #3.
      if (place === 2) addK(leaderKey, q * 0.35);
      else if (place === 3) addK(leaderKey, q * 0.12);
    }

    // Normalise the kernel map to sum to 1.
    let kTotal = 0;
    for (const v of kernel.values()) kTotal += v;
    if (kTotal > 0) {
      for (const [k, v] of kernel) kernel.set(k, v / kTotal);
    }

    // Velocity tail: climbers sitting between rank 11 and TAIL_MAX_RANK.
    const tail = new Map<string, number>();
    let tTotal = 0;
    for (const g of games) {
      if (g.rank < 11 || g.rank > TAIL_MAX_RANK) continue;
      const { r, v } = feats(g);
      if (v < TAIL_MIN_VEL) continue;
      const w = (1 + v / 8) * Math.exp(-(r - 11) / 18);
      tail.set(g.key, w);
      tTotal += w;
    }
    if (tTotal > 0) {
      for (const [k, v] of tail) tail.set(k, v / tTotal);
    }

    // If no game qualifies for the tail, the kernel keeps the whole share.
    const tf = tTotal > 0 ? tailShare(place) : 0;

    for (const g of games) {
      const kernelPart = kernel.get(g.key) ?? 0;
      const tailPart = tail.get(g.key) ?? 0;
      const p = ((1 - tf) * kernelPart + tf * tailPart) * (1 - unnamed);
      if (p < MIN_P) continue;
      const pr = round6(p);
      if (pr < MIN_P) continue;
      probabilities.push({ universe_id: g.id, place, p: pr });
    }
  }

  probabilities.sort(
    (a, b) => a.place - b.place || b.p - a.p || String(a.universe_id).localeCompare(String(b.universe_id)),
  );

  // top10 by blended, velocity-adjusted expected rank.
  const blended = games.map((g) => {
    const rp = stayMu(g);
    let e: number;
    if (leaderKey === null) {
      e = rp;
    } else if (g.key === leaderKey) {
      e = (1 - q) * rp + q * DROPOUT_RANK;
    } else {
      e = (1 - q) * rp + q * (dropMu(g) as number);
    }
    return { g, e };
  });
  blended.sort((a, b) => a.e - b.e || a.g.rank - b.g.rank || a.g.key.localeCompare(b.g.key));
  const top10 = blended.slice(0, 10).map((x) => x.g.id);

  writeFileSync(outPath, JSON.stringify({ probabilities, top10 }));
}

main();

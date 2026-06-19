import { CLASS_COLORS, CLASS_NAMES } from "../lib/palette";
import type { PointCloud, SceneMeta } from "../types";

// Binary layout (little-endian, concatenated). Float32 sections first to keep their
// byte offsets 4-byte aligned for any N. A "shade" Float32[N] section is present only
// when meta.hasShade (baked hillshade for terrain depth):
//   position F32[N*3] | scalar F32[N] | [shade F32[N]] | rgb U8[N*3] | pred U8[N] | gt U8[N]
export async function loadScene(name: string): Promise<PointCloud> {
  const base = `${import.meta.env.BASE_URL}data/${name}`;
  const meta: SceneMeta = await (await fetch(`${base}.meta.json`)).json();
  const buf = await (await fetch(`${base}.bin`)).arrayBuffer();

  const n = meta.numPoints;
  let off = 0;
  const position = new Float32Array(buf, off, n * 3);
  off += n * 3 * 4;
  const entropy = new Float32Array(buf, off, n);
  off += n * 4;
  let shade: Float32Array;
  if (meta.hasShade) {
    shade = new Float32Array(buf, off, n);
    off += n * 4;
  } else {
    shade = new Float32Array(n).fill(1);
  }
  const rgb = new Uint8Array(buf, off, n * 3);
  off += n * 3;
  const pred = new Uint8Array(buf, off, n);
  off += n;
  const gt = new Uint8Array(buf, off, n);

  let entropyMax = 1e-6;
  for (let i = 0; i < n; i++) if (entropy[i] > entropyMax) entropyMax = entropy[i];
  return { meta, position, rgb, pred, gt, entropy, entropyMax, shade };
}

// Synthetic terrain so `npm run dev` shows something before real data is exported.
export function makeSampleScene(n = 40000): PointCloud {
  const position = new Float32Array(n * 3);
  const rgb = new Uint8Array(n * 3);
  const pred = new Uint8Array(n);
  const gt = new Uint8Array(n);
  const entropy = new Float32Array(n);
  const rng = mulberry32(42);

  for (let i = 0; i < n; i++) {
    const x = (rng() - 0.5) * 50;
    const y = (rng() - 0.5) * 50;
    const ground = 2 * Math.sin(x / 8) + 1.5 * Math.cos(y / 6);
    // a "building" block and some "vegetation" bumps
    const inBldg = Math.abs(x - 12) < 5 && Math.abs(y + 8) < 5;
    const z = inBldg ? ground + 6 : ground + (rng() < 0.2 ? rng() * 3 : 0);
    position[i * 3] = x;
    position[i * 3 + 1] = z; // y-up for three.js
    position[i * 3 + 2] = y;

    let cls = 1; // ground
    if (inBldg) cls = 3; // building
    else if (z - ground > 0.8) cls = 2; // vegetation
    gt[i] = cls;
    // prediction: mostly correct, some noise near boundaries
    pred[i] = rng() < 0.92 ? cls : (cls + 1) % 4;
    entropy[i] = pred[i] === gt[i] ? rng() * 0.2 : 0.6 + rng() * 0.4;
    const c = CLASS_COLORS[cls];
    rgb[i * 3] = parseInt(c.slice(1, 3), 16);
    rgb[i * 3 + 1] = parseInt(c.slice(3, 5), 16);
    rgb[i * 3 + 2] = parseInt(c.slice(5, 7), 16);
  }

  const meta: SceneMeta = {
    numPoints: n,
    classNames: CLASS_NAMES,
    classColors: CLASS_COLORS,
    hasColor: true,
    hasGt: true,
  };
  return { meta, position, rgb, pred, gt, entropy, entropyMax: 1, shade: new Float32Array(n).fill(1) };
}

function mulberry32(seed: number) {
  let a = seed;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

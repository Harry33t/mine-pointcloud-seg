// FRACTAL 7-class palette (must match mpcseg viz / training class ids).
export const CLASS_NAMES = [
  "other",
  "ground",
  "vegetation",
  "building",
  "water",
  "bridge",
  "permanent",
];

export const CLASS_COLORS = [
  "#9e9e9e", // 0 other
  "#8d6e63", // 1 ground
  "#43a047", // 2 vegetation
  "#e53935", // 3 building
  "#1e88e5", // 4 water
  "#fb8c00", // 5 bridge
  "#8e24aa", // 6 permanent
];

const UNKNOWN_COLOR: [number, number, number] = [0.1, 0.1, 0.1];

function hexToRgb01(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

export const CLASS_RGB01 = CLASS_COLORS.map(hexToRgb01);

// Compact viridis-like ramp for continuous fields (entropy).
const VIRIDIS: [number, number, number][] = [
  [0.267, 0.005, 0.329],
  [0.283, 0.141, 0.458],
  [0.254, 0.265, 0.53],
  [0.207, 0.372, 0.553],
  [0.164, 0.471, 0.558],
  [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518],
  [0.267, 0.749, 0.441],
  [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.15],
  [0.993, 0.906, 0.144],
];

export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function classRgb(id: number): [number, number, number] {
  return id >= 0 && id < CLASS_RGB01.length ? CLASS_RGB01[id] : UNKNOWN_COLOR;
}

export { UNKNOWN_COLOR };

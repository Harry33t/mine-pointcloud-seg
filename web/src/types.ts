export type ColorMode = "rgb" | "pred" | "gt" | "entropy";

export interface SceneMeta {
  numPoints: number;
  classNames: string[];
  classColors: string[]; // hex per class id
  hasColor: boolean;
  hasGt: boolean;
  classLabel?: string; // label for the class/prediction mode (default "Prediction")
  scalarLabel?: string; // legend title for the continuous field
  scalarShort?: string; // short button label for the continuous field (default "Uncertainty")
  hasShade?: boolean; // baked hillshade present (terrain depth)
  stats?: Record<string, string>; // headline numbers for the HUD overlay
  annotations?: { label: string; pos: [number, number, number] }[];
  bounds?: { min: [number, number, number]; max: [number, number, number] };
}

export interface PointCloud {
  meta: SceneMeta;
  position: Float32Array; // [N*3]
  rgb: Uint8Array; // [N*3], 0-255
  pred: Uint8Array; // [N]
  gt: Uint8Array; // [N], 255 = unknown
  entropy: Float32Array; // [N]
  entropyMax: number;
  shade: Float32Array; // [N], 0-1 baked hillshade (1 = unshaded)
}

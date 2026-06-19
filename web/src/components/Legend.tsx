import type { ColorMode, SceneMeta } from "../types";
import { viridis } from "../lib/palette";

function rgb01ToCss([r, g, b]: [number, number, number]) {
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
}

export default function Legend({
  mode,
  meta,
  entropyMax,
}: {
  mode: ColorMode;
  meta: SceneMeta;
  entropyMax: number;
}) {
  if (mode === "entropy") {
    const stops = Array.from({ length: 10 }, (_, i) => rgb01ToCss(viridis(i / 9)));
    return (
      <div className="legend">
        <div className="legend-title">{meta.scalarLabel ?? "predictive entropy"}</div>
        <div className="ramp" style={{ background: `linear-gradient(to right, ${stops.join(",")})` }} />
        <div className="ramp-labels">
          <span>0</span>
          <span>{entropyMax.toFixed(2)}</span>
        </div>
      </div>
    );
  }
  if (mode === "rgb") {
    return <div className="legend"><div className="legend-title">aerial RGB</div></div>;
  }
  return (
    <div className="legend">
      <div className="legend-title">
        {mode === "pred" ? (meta.classLabel ?? "predicted class") : "ground-truth class"}
      </div>
      <ul className="classes">
        {meta.classNames.map((name, i) => (
          <li key={name}>
            <span className="swatch" style={{ background: meta.classColors[i] }} />
            {name}
          </li>
        ))}
      </ul>
    </div>
  );
}

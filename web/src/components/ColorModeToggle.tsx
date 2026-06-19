import type { ColorMode, SceneMeta } from "../types";

// Buttons adapt to the scene: RGB only if the cloud has colour, ground-truth only
// if labels exist, and the class/scalar labels come from the scene meta.
export default function ColorModeToggle({
  mode,
  onChange,
  meta,
}: {
  mode: ColorMode;
  onChange: (m: ColorMode) => void;
  meta: SceneMeta;
}) {
  const modes: { id: ColorMode; label: string }[] = [];
  if (meta.hasColor) modes.push({ id: "rgb", label: "RGB / aerial" });
  modes.push({ id: "pred", label: meta.classLabel ?? "Prediction" });
  if (meta.hasGt) modes.push({ id: "gt", label: "Ground truth" });
  modes.push({ id: "entropy", label: meta.scalarShort ?? "Uncertainty" });

  return (
    <div className="toggle">
      {modes.map((m) => (
        <button
          key={m.id}
          className={mode === m.id ? "active" : ""}
          onClick={() => onChange(m.id)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

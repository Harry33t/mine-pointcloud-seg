import { useEffect, useState } from "react";
import PointCloudViewer from "./components/PointCloudViewer";
import ColorModeToggle from "./components/ColorModeToggle";
import Legend from "./components/Legend";
import ResultsPanel from "./components/ResultsPanel";
import { loadScene, makeSampleScene } from "./data/loadPointCloud";
import type { ColorMode, PointCloud } from "./types";

interface SceneDef {
  id: string;
  label: string;
  defaultMode: ColorMode;
  title: string;
  description: string;
  shows: string;
  source: string;
  sourceUrl: string;
}

// Scenes are exported to web/public/data/<id>.bin + .meta.json.
const SCENES: SceneDef[] = [
  {
    id: "mine",
    label: "McKinley open-pit mine · 0.3 m LiDAR + aerial",
    defaultMode: "rgb",
    title: "McKinley open-pit coal mine, New Mexico",
    description:
      "~400 m AOI · 0.3 m airborne-LiDAR bare-earth DEM with 0.1 m orthophoto draped · 81 m relief · 400k points (vertical ×1.5).",
    shows:
      "Geometry-only weak labels (cloth-simulation ground filter + slope) split the pit into floor/bench, slope/spoil and steep highwall with ZERO manual annotation — the weak-label method applied to real, unlabelled mine terrain.",
    source:
      "OpenTopography · U.S. OSMRE, collected by Surdex (2023-10-17) · DOI 10.5069/G9BZ6486 · CC BY 4.0",
    sourceUrl: "https://doi.org/10.5069/G9BZ6486",
  },
  {
    id: "scene",
    label: "FRACTAL aerial benchmark · model predictions",
    defaultMode: "pred",
    title: "FRACTAL aerial-LiDAR benchmark (IGN France)",
    description:
      "50 m tile · PTv3 semantic segmentation + per-point uncertainty · 119k points.",
    shows:
      "Switch prediction vs ground truth to see where the model is right/wrong; uncertainty (softmax entropy) peaks at class boundaries. Calibrated by temperature scaling — see Results below.",
    source: "IGN · FRACTAL dataset · Hugging Face IGNF/FRACTAL · open licence",
    sourceUrl: "https://huggingface.co/datasets/IGNF/FRACTAL",
  },
];

function InfoCard({ s }: { s: SceneDef }) {
  return (
    <div className="info">
      <div className="info-title">{s.title}</div>
      <p className="info-desc">{s.description}</p>
      <p className="info-desc">{s.shows}</p>
      <p className="info-src">
        <a href={s.sourceUrl} target="_blank" rel="noreferrer">
          {s.source}
        </a>
      </p>
    </div>
  );
}

export default function App() {
  const [sceneId, setSceneId] = useState(SCENES[0].id);
  const [pc, setPc] = useState<PointCloud | null>(null);
  const [mode, setMode] = useState<ColorMode>(SCENES[0].defaultMode);
  const [source, setSource] = useState<"real" | "sample">("real");
  const sceneDef = SCENES.find((s) => s.id === sceneId) ?? SCENES[0];

  useEffect(() => {
    let alive = true;
    setPc(null);
    loadScene(sceneId)
      .then((scene) => {
        if (!alive) return;
        setPc(scene);
        setSource("real");
        setMode(scene.meta.hasColor ? sceneDef.defaultMode : "pred");
      })
      .catch(() => {
        if (!alive) return;
        setPc(makeSampleScene());
        setSource("sample");
        setMode("pred");
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneId]);

  return (
    <div className="app">
      <header>
        <h1>Label-Efficient Semantic Segmentation of Post-Mining LiDAR</h1>
        <p className="sub">
          Self-supervised pre-training + weak labels · cross-site transfer · calibrated per-point uncertainty
        </p>
      </header>

      <main className="stage">
        <div className="viewer">
          {pc ? (
            <PointCloudViewer pc={pc} mode={mode} />
          ) : (
            <div className="loading">loading point cloud…</div>
          )}
          {pc?.meta.stats && (
            <div className="hud">
              <div className="hud-title">{sceneDef.title}</div>
              {Object.entries(pc.meta.stats).map(([k, v]) => (
                <div className="hud-row" key={k}>
                  <span className="hud-k">{k}</span>
                  <span className="hud-v">{v}</span>
                </div>
              ))}
            </div>
          )}
          {source === "sample" && (
            <div className="badge">sample data — export a real scene to /public/data</div>
          )}
        </div>
        <aside className="panel">
          <select
            className="scene-select"
            value={sceneId}
            onChange={(e) => setSceneId(e.target.value)}
          >
            {SCENES.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          {pc && <ColorModeToggle mode={mode} onChange={setMode} meta={pc.meta} />}
          {pc && <Legend mode={mode} meta={pc.meta} entropyMax={pc.entropyMax} />}
          <InfoCard s={sceneDef} />
        </aside>
      </main>

      <ResultsPanel />

      <footer>
        <span>drag to orbit · scroll to zoom · switch scene & colouring above</span>
        <span className="attrib">
          Data: OpenTopography (McKinley Mine, CC BY 4.0) · IGN FRACTAL · processed with Pointcept / PTv3.
        </span>
      </footer>
    </div>
  );
}

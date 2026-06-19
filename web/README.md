# web — interactive viewer (React + TypeScript)

Signature 3D viewer for the project: orbit a LiDAR scene and switch the point
colouring between **RGB ↔ prediction ↔ ground-truth ↔ uncertainty**, alongside the
label-efficiency curve, calibration diagram, and cross-site (LOCO) matrix.

Stack: Vite · React 18 · TypeScript · three.js / react-three-fiber.

## Run

```bash
cd web
npm install
npm run dev      # http://localhost:5173  (shows synthetic sample data out of the box)
npm run build    # static bundle in dist/
```

## Feeding real data

The app loads `public/data/<scene>.bin` + `<scene>.meta.json` (default scene name
`scene`). Generate them from a LAS produced by inference:

```bash
python -m mpcseg.viz.export_web \
  --in outputs/scene_viz.las \
  --out web/public/data/scene --max-points 300000
```

Drop the result figures into `public/figs/` (`label_efficiency.png`,
`reliability.png`, `loco_matrix.png`) — the Results panel picks them up automatically.

If no real data is present, the viewer falls back to a synthetic terrain so the UI
always runs.

## Layout
- `src/components/PointCloudViewer.tsx` — r3f canvas, per-point colour modes
- `src/components/ColorModeToggle.tsx` · `Legend.tsx` · `ResultsPanel.tsx`
- `src/data/loadPointCloud.ts` — binary loader + synthetic fallback
- `src/lib/palette.ts` — class palette + viridis ramp (matches the Python side)

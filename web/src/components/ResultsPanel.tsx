// Result figures produced by the Python side, shown as an editorial figure gallery.
import { useState } from "react";

const BASE = import.meta.env.BASE_URL;

interface Item {
  src: string;
  title: string;
  text: string;
}

const ITEMS: Item[] = [
  {
    src: "label_efficiency.png",
    title: "Label efficiency",
    text: "Per-class IoU against label budget — dominant classes saturate from 1% of labels, while rare classes (bridge, permanent structure) only emerge near full supervision.",
  },
  {
    src: "ssl_curve.png",
    title: "Self-supervised pre-training",
    text: "In-domain MSC pre-training on unlabelled aerial point clouds lifts mIoU by ~4–5 points over random initialisation at 1, 10 and 100% labels.",
  },
  {
    src: "reliability.png",
    title: "Calibrated uncertainty",
    text: "A single temperature fitted on held-out data cuts expected calibration error from 0.021 to 0.007; the per-point entropy drives the viewer's uncertainty mode.",
  },
  {
    src: "loco_matrix.png",
    title: "Cross-site transfer",
    text: "Leave-one-site-out across four geographic regions: a ~6 mIoU drop off-diagonal, smallest when training on the data-richest site.",
  },
  {
    src: "mine_hillshade.png",
    title: "Real mine",
    text: "Geometric weak labels segment the McKinley open-pit (0.3 m airborne LiDAR) into floor, slope and highwall with no manual annotation — the method on real, unlabelled terrain.",
  },
];

function Figure({ item }: { item: Item }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <figure className="result-fig">
      <img src={`${BASE}figs/${item.src}`} alt={item.title} onError={() => setOk(false)} />
      <figcaption>
        <strong>{item.title}.</strong> {item.text}
      </figcaption>
    </figure>
  );
}

export default function ResultsPanel() {
  return (
    <section className="results">
      <h2>Results</h2>
      <p className="results-lead">
        Validated on open aerial-LiDAR benchmarks, then carried over to a real open-pit mine.
      </p>
      <div className="result-grid">
        {ITEMS.map((it) => (
          <Figure key={it.src} item={it} />
        ))}
      </div>
    </section>
  );
}

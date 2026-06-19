import { Canvas } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import { useMemo, useState } from "react";
import * as THREE from "three";
import type { ColorMode, PointCloud } from "../types";
import { viridis } from "../lib/palette";

function hexToRgb01(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

function buildColors(pc: PointCloud, mode: ColorMode): Float32Array {
  const n = pc.meta.numPoints;
  const col = new Float32Array(n * 3);
  const pal = pc.meta.classColors.map(hexToRgb01);
  const unknown: [number, number, number] = [0.1, 0.1, 0.1];
  const cls = (id: number) => (id >= 0 && id < pal.length ? pal[id] : unknown);
  for (let i = 0; i < n; i++) {
    let r = 0,
      g = 0,
      b = 0;
    if (mode === "rgb") {
      r = pc.rgb[i * 3] / 255;
      g = pc.rgb[i * 3 + 1] / 255;
      b = pc.rgb[i * 3 + 2] / 255;
    } else if (mode === "pred") {
      [r, g, b] = cls(pc.pred[i]);
    } else if (mode === "gt") {
      [r, g, b] = cls(pc.gt[i] === 255 ? -1 : pc.gt[i]);
    } else {
      [r, g, b] = viridis(pc.entropy[i] / pc.entropyMax);
    }
    // bake hillshade so the terrain reads as solid 3D (shadow floor keeps detail)
    const m = 0.32 + 0.68 * pc.shade[i];
    col[i * 3] = r * m;
    col[i * 3 + 1] = g * m;
    col[i * 3 + 2] = b * m;
  }
  return col;
}

function centered(pc: PointCloud) {
  const n = pc.meta.numPoints;
  const p = pc.position;
  let cx = 0,
    cy = 0,
    cz = 0;
  for (let i = 0; i < n; i++) {
    cx += p[i * 3];
    cy += p[i * 3 + 1];
    cz += p[i * 3 + 2];
  }
  cx /= n;
  cy /= n;
  cz /= n;
  const out = new Float32Array(n * 3);
  let radius = 1;
  for (let i = 0; i < n; i++) {
    const x = p[i * 3] - cx,
      y = p[i * 3 + 1] - cy,
      z = p[i * 3 + 2] - cz;
    out[i * 3] = x;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = z;
    radius = Math.max(radius, Math.hypot(x, y, z));
  }
  return { position: out, radius, center: [cx, cy, cz] as [number, number, number] };
}

function Scene({ pc, mode, sizeFactor }: { pc: PointCloud; mode: ColorMode; sizeFactor: number }) {
  const { position, radius, center } = useMemo(() => centered(pc), [pc]);
  const colors = useMemo(() => buildColors(pc, mode), [pc, mode]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(position, 3));
    return g;
  }, [position]);

  useMemo(() => {
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.attributes.color.needsUpdate = true;
  }, [geometry, colors]);

  return (
    <>
      <points geometry={geometry}>
        <pointsMaterial size={Math.max(0.03, radius * sizeFactor)} vertexColors sizeAttenuation />
      </points>
      {pc.meta.annotations?.map((a, i) => (
        <Html
          key={i}
          position={[a.pos[0] - center[0], a.pos[1] - center[1], a.pos[2] - center[2]]}
          center
          zIndexRange={[10, 0]}
        >
          <div className="annot">{a.label}</div>
        </Html>
      ))}
    </>
  );
}

export default function PointCloudViewer({
  pc,
  mode,
  sizeFactor = 0.0035,
}: {
  pc: PointCloud;
  mode: ColorMode;
  sizeFactor?: number;
}) {
  const [spin, setSpin] = useState(true);
  const b = pc.meta.bounds;
  const ext = b
    ? Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2])
    : 60;
  const d = ext * 0.85;
  return (
    <Canvas
      key={`${pc.meta.numPoints}-${ext.toFixed(0)}`}
      camera={{ position: [d, d * 0.65, d], near: ext * 0.002, far: ext * 30 }}
      dpr={[1, 2]}
      onPointerDown={() => setSpin(false)}
    >
      <color attach="background" args={["#120c08"]} />
      <Scene pc={pc} mode={mode} sizeFactor={sizeFactor} />
      <OrbitControls
        makeDefault
        autoRotate={spin}
        autoRotateSpeed={0.45}
        maxDistance={ext * 6}
      />
    </Canvas>
  );
}

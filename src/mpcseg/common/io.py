"""Point-cloud I/O helpers built on laspy + Open3D + numpy.

Read/write LAS/LAZ, crop by bounding box, voxel downsample, estimate normals.
All functions work on plain numpy arrays so they stay framework-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PointCloud:
    """A minimal in-memory point cloud.

    coord:    (N, 3) float64 XYZ
    intensity:(N,)   float32 or None
    color:    (N, 3) float32 in [0, 1] or None
    label:    (N,)   int32 semantic/classification or None
    extra:    dict of name -> (N,) array for any extra per-point scalar field
    """

    coord: np.ndarray
    intensity: np.ndarray | None = None
    color: np.ndarray | None = None
    label: np.ndarray | None = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:  # number of points
        return int(self.coord.shape[0])

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """Return a new cloud keeping points where ``mask`` is True."""
        return PointCloud(
            coord=self.coord[mask],
            intensity=None if self.intensity is None else self.intensity[mask],
            color=None if self.color is None else self.color[mask],
            label=None if self.label is None else self.label[mask],
            extra={k: v[mask] for k, v in self.extra.items()},
        )


def read_las(path: str | Path) -> PointCloud:
    """Read a LAS/LAZ file into a :class:`PointCloud`.

    Requires ``laspy[lazrs,laszip]`` for LAZ. RGB (16-bit) is normalised to [0, 1].
    The ASPRS ``classification`` field is loaded as ``label`` (note: for raw mine
    data this is only ground/unclassified, not semantic mine classes).
    """
    import laspy

    las = laspy.read(str(path))
    coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)

    intensity = None
    if "intensity" in las.point_format.dimension_names:
        intensity = np.asarray(las.intensity, dtype=np.float32)

    color = None
    if {"red", "green", "blue"} <= set(las.point_format.dimension_names):
        rgb = np.stack([las.red, las.green, las.blue], axis=1).astype(np.float32)
        # LAS stores 16-bit colour; normalise. Fall back gracefully for 8-bit.
        scale = 65535.0 if rgb.max() > 255 else 255.0
        color = rgb / scale

    label = None
    if "classification" in las.point_format.dimension_names:
        label = np.asarray(las.classification, dtype=np.int32)

    return PointCloud(coord=coord, intensity=intensity, color=color, label=label)


def write_las(path: str | Path, pc: PointCloud, point_format: int = 7) -> None:
    """Write a :class:`PointCloud` to LAS 1.4.

    ``pc.extra`` scalar fields are written as ExtraBytes dimensions (float32) so
    downstream tools (e.g. PotreeConverter) expose them as switchable attributes.
    point_format 7 carries RGB + GPS time; use 6 if no colour is needed.
    """
    import laspy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=point_format, version="1.4")
    # robust offset/scale so coordinates round-trip without precision loss
    header.offsets = pc.coord.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    for name in pc.extra:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32))

    las = laspy.LasData(header)
    las.x, las.y, las.z = pc.coord[:, 0], pc.coord[:, 1], pc.coord[:, 2]
    if pc.intensity is not None:
        las.intensity = pc.intensity.astype(np.uint16)
    if pc.color is not None:
        rgb16 = np.clip(pc.color * 65535.0, 0, 65535).astype(np.uint16)
        las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    if pc.label is not None:
        las.classification = np.clip(pc.label, 0, 255).astype(np.uint8)
    for name, values in pc.extra.items():
        las[name] = values.astype(np.float32)

    las.write(str(path))


def crop_bbox(pc: PointCloud, bbox: tuple[float, float, float, float]) -> PointCloud:
    """Crop to a 2D bounding box ``(xmin, ymin, xmax, ymax)`` (CRS units of the data)."""
    xmin, ymin, xmax, ymax = bbox
    m = (
        (pc.coord[:, 0] >= xmin)
        & (pc.coord[:, 0] <= xmax)
        & (pc.coord[:, 1] >= ymin)
        & (pc.coord[:, 1] <= ymax)
    )
    return pc.subset(m)


def to_open3d(pc: PointCloud):
    """Convert to an Open3D point cloud (for voxel downsample / normals / viewing)."""
    import open3d as o3d

    o = o3d.geometry.PointCloud()
    o.points = o3d.utility.Vector3dVector(pc.coord)
    if pc.color is not None:
        o.colors = o3d.utility.Vector3dVector(pc.color)
    return o


def estimate_normals(pc: PointCloud, radius: float = 1.0, max_nn: int = 30) -> np.ndarray:
    """Estimate per-point normals via Open3D. Returns (N, 3) float32."""
    import open3d as o3d

    o = to_open3d(pc)
    o.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(o.normals, dtype=np.float32)

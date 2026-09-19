"""深度圆形上表面检测中各模块共用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


ColorRange = Tuple[np.ndarray, np.ndarray]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    """与彩色图对齐后的深度相机内参。"""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class CameraFrame:
    """一帧彩色图，以及可选的对齐深度和内参。"""

    color: np.ndarray
    timestamp: float
    depth_m: Optional[np.ndarray] = None
    intrinsics: Optional[CameraIntrinsics] = None
    # D435i加速度计测得的重力方向，已经转换到彩色相机坐标系。
    gravity_camera: Optional[Point3D] = None


@dataclass(frozen=True)
class SurfacePlaneObservation:
    """从颜色候选区域中拟合出的三维上表面平面。"""

    normal: Point3D
    offset: float
    center: Point3D
    inlier_count: int
    candidate_count: int
    # abs(dot(平面单位法向量, 重力单位向量))，越接近1越像水平上表面。
    gravity_alignment: Optional[float] = None

    @property
    def inlier_ratio(self) -> float:
        if self.candidate_count <= 0:
            return 0.0
        return self.inlier_count / self.candidate_count

    @property
    def normal_to_gravity_deg(self) -> Optional[float]:
        """平面法向量与重力轴的夹角；0度表示水平面。"""
        if self.gravity_alignment is None:
            return None
        alignment = min(1.0, max(0.0, self.gravity_alignment))
        return float(np.degrees(np.arccos(alignment)))


@dataclass(frozen=True)
class CircleObservation:
    """一帧中通过几何检查的近圆形上表面。"""

    center: Point2D
    radius: float
    contour_area: float
    circularity: float
    fill_ratio: float


@dataclass(frozen=True)
class RingObservation:
    """A detected RGB ring, represented by integer pixel geometry."""

    center: Tuple[int, int]
    radius: int


@dataclass(frozen=True)
class EllipseObservation:
    """轮廓拟合出的椭圆，角度表示长轴相对图像x轴的旋转角。"""

    center: Point2D
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float
    fit_error: float


@dataclass(frozen=True)
class TrackEstimate:
    """alpha-beta滤波后的二维圆心和像素速度。"""

    center: Point2D
    velocity: Point2D
    measured: bool
    missing_frames: int


@dataclass(frozen=True)
class PipelineResult:
    """处理一帧得到的颜色、深度表面、圆形和跟踪结果。"""

    blurred_frame: np.ndarray
    color_mask: np.ndarray
    surface_mask: np.ndarray
    depth_applied: bool
    surface_planes: List[SurfacePlaneObservation]
    observations: List[CircleObservation]
    selected: Optional[CircleObservation]
    estimate: Optional[TrackEstimate]

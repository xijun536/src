"""组合颜色、深度上表面、圆形检测和二维跟踪。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..association import choose_observation
from ..models import CameraIntrinsics, ColorRange, PipelineResult, Point3D
from .alpha_beta_tracker import AlphaBetaTracker
from .circle_detector import CircleDetector
from .depth_surface_filter import DepthSurfaceFilter
from .dilation_filter import DilationFilter
from .erosion_filter import ErosionFilter
from .gaussian_filter import GaussianFilter
from .hsv_binary_filter import HSVBinaryFilter


@dataclass(frozen=True)
class PipelineConfig:
    gaussian_size: int = 5
    morphology_size: int = 3
    open_iterations: int = 1
    close_iterations: int = 1

    depth_enabled: bool = True
    depth_required: bool = True
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    plane_distance_m: float = 0.005
    ransac_iterations: int = 180
    min_plane_points: int = 120
    max_sample_points: int = 4000
    min_plane_inlier_ratio: float = 0.25
    use_gravity: bool = True
    gravity_required: bool = True
    max_normal_gravity_angle_deg: float = 30.0
    surface_close_iterations: int = 0

    min_area: float = 150.0
    min_contour_points: int = 20
    min_radius_px: float = 10.0
    max_radius_px: float = 300.0
    min_circularity: float = 0.65
    min_fill_ratio: float = 0.65

    alpha: float = 0.75
    beta: float = 0.15
    max_missing: int = 5
    association_distance: float = 160.0


class CircleSurfaceTrackingPipeline:
    """先用深度去除侧面，再检测并跟踪近圆形上表面。"""

    def __init__(
        self,
        color_ranges: Sequence[ColorRange],
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if min(
            self.config.open_iterations,
            self.config.close_iterations,
            self.config.surface_close_iterations,
        ) < 0:
            raise ValueError("形态学处理次数不能小于0")
        if self.config.association_distance < 0.0:
            raise ValueError("association_distance不能小于0")

        self.gaussian = GaussianFilter(self.config.gaussian_size)
        self.hsv_binary = HSVBinaryFilter(color_ranges)
        self.erosion = ErosionFilter(self.config.morphology_size)
        self.dilation = DilationFilter(self.config.morphology_size)
        self.depth_surface = DepthSurfaceFilter(
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
            plane_distance_m=self.config.plane_distance_m,
            ransac_iterations=self.config.ransac_iterations,
            min_points=self.config.min_plane_points,
            max_sample_points=self.config.max_sample_points,
            min_inlier_ratio=self.config.min_plane_inlier_ratio,
            use_gravity=self.config.use_gravity,
            gravity_required=self.config.gravity_required,
            max_normal_gravity_angle_deg=(
                self.config.max_normal_gravity_angle_deg
            ),
        )
        self.circle_detector = CircleDetector(
            min_area=self.config.min_area,
            min_contour_points=self.config.min_contour_points,
            min_radius_px=self.config.min_radius_px,
            max_radius_px=self.config.max_radius_px,
            min_circularity=self.config.min_circularity,
            min_fill_ratio=self.config.min_fill_ratio,
        )
        self.tracker = AlphaBetaTracker(
            alpha=self.config.alpha,
            beta=self.config.beta,
            max_missing=self.config.max_missing,
        )

    def set_color_ranges(self, color_ranges: Sequence[ColorRange]) -> None:
        """Replace the HSV target color and discard tracking from the old color."""

        new_filter = HSVBinaryFilter(color_ranges)
        self.hsv_binary = new_filter
        self.reset()

    def process(
        self,
        frame: np.ndarray,
        timestamp: float,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        gravity_camera: Optional[Point3D] = None,
    ) -> PipelineResult:
        # 1. 彩色图像去噪并生成目标颜色掩膜。
        blurred = self.gaussian.apply(frame)
        color_mask = self.hsv_binary.apply(blurred)

        #两种方法：先腐蚀后膨胀或者先膨胀后腐蚀
        if self.config.open_iterations > 0:
            color_mask = self.erosion.apply(color_mask, self.config.open_iterations)
            color_mask = self.dilation.apply(color_mask, self.config.open_iterations)
        if self.config.close_iterations > 0:
            color_mask = self.dilation.apply(color_mask, self.config.close_iterations)
            color_mask = self.erosion.apply(color_mask, self.config.close_iterations)

        # 2. 在颜色区域中使用对齐深度，只保留主要平面上表面。
        depth_applied = False
        surface_planes = []
        surface_mask = color_mask.copy()
        if self.config.depth_enabled:
            if depth_m is not None and intrinsics is not None:
                surface_mask, surface_planes = self.depth_surface.apply(
                    color_mask,
                    depth_m,
                    intrinsics,
                    gravity_camera,
                )
                depth_applied = True
                if self.config.surface_close_iterations > 0:
                    surface_mask = self.dilation.apply(
                        surface_mask,
                        self.config.surface_close_iterations,
                    )
                    surface_mask = self.erosion.apply(
                        surface_mask,
                        self.config.surface_close_iterations,
                    )
            elif self.config.depth_required:
                surface_mask = np.zeros_like(color_mask)

        # 3. 对深度上表面做圆形度检查和最小外接圆计算。
        observations = self.circle_detector.detect(surface_mask)
        predicted = self.tracker.predict(timestamp)
        selected = choose_observation(
            observations,
            predicted,
            self.config.association_distance,
        )
        estimate = self.tracker.update(
            selected.center if selected is not None else None,
            timestamp,
        )

        return PipelineResult(
            blurred_frame=blurred,
            color_mask=color_mask,
            surface_mask=surface_mask,
            depth_applied=depth_applied,
            surface_planes=surface_planes,
            observations=observations,
            selected=selected,
            estimate=estimate,
        )

    def reset(self) -> None:
        self.tracker.reset()

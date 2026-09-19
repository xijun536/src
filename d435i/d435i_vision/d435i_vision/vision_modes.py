"""Pure result schemas and per-frame dispatch for configurable vision modes."""

#定义了核心模式调度器，以及用于传输的信息

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .models import (
    CameraFrame,
    ColorRange,
    EllipseObservation,
    PipelineResult,
    RingObservation,
)
from .ring_geometry import (
    analyze_ellipse_geometry,
    calculate_ring_calibration,
    filter_ellipses_by_major_axis_cm,
    filter_ellipses_near_rings,
    measure_ellipse_distances_cm,
)
from .ros_control import VisionMode
from .visualization import draw_debug, draw_idle, draw_rings
from .vision_status import (
    RingCalibrationConfig,
    RingCalibrationStatus,
    SurfaceTrackingStatus,
    surface_status_from_pipeline,
)


@dataclass(frozen=True)
class ModeFrameResult:
    """Message and optional display images produced for one camera frame."""

    message: Optional[Dict[str, Any]]
    main_image: Optional[np.ndarray]
    auxiliary_images: Dict[str, np.ndarray]
    surface_status: Optional[SurfaceTrackingStatus] = None
    ring_status: Optional[RingCalibrationStatus] = None


def surface_message(
    wire_value: int,
    timestamp: float,
    color: str,
    result: PipelineResult,
) -> Optional[Dict[str, Any]]:
    """构建一个表面追踪的信息，用于终端/JSON传递，或者窗口显示"""

    if result.estimate is None:
        return None

    message: Dict[str, Any] = {
        "mode": wire_value,
        "mode_name": VisionMode.SURFACE_TRACKING.value,
        "timestamp_monotonic_s": round(float(timestamp), 6),
        "color": color,
        "measured": result.estimate.measured,
        "center_px": [round(value, 2) for value in result.estimate.center],
        "velocity_px_s": [round(value, 2) for value in result.estimate.velocity],
        "speed_px_s": round(math.hypot(*result.estimate.velocity), 2),
        "missing_frames": result.estimate.missing_frames,
        "depth_applied": result.depth_applied,
        "surface_plane_count": len(result.surface_planes),
    }
    if result.selected is not None:
        message.update(
            {
                "radius_px": round(result.selected.radius, 2),
                "circularity": round(result.selected.circularity, 3),
                "circle_fill_ratio": round(result.selected.fill_ratio, 3),
            }
        )
    if result.surface_planes:
        plane = result.surface_planes[0]
        message.update(
            {
                "surface_center_camera_m": [round(value, 4) for value in plane.center],
                "surface_normal_camera": [round(value, 4) for value in plane.normal],
                "surface_inlier_ratio": round(plane.inlier_ratio, 3),
            }
        )
        if plane.normal_to_gravity_deg is not None:
            message["surface_normal_to_gravity_deg"] = round(
                plane.normal_to_gravity_deg,
                2,
            )
    return message


def ring_message(
    wire_value: int,
    rings: Sequence[RingObservation],
    collinearity_tolerance_px: float = 5.0,
    distances_from_first_cm: Optional[Sequence[Optional[float]]] = None,
    ellipses: Sequence[EllipseObservation] = (),
    ellipse_candidate_count: Optional[int] = None,
) -> Dict[str, Any]:
    """生成只公开椭圆结果的色环识别消息；Hough圆仅供内部筛选。"""

    geometry = analyze_ellipse_geometry(ellipses, collinearity_tolerance_px)
    centimeter_distances = (
        tuple(distances_from_first_cm)
        if distances_from_first_cm is not None
        else (None,) * len(ellipses)
    )
    if len(centimeter_distances) != len(ellipses):
        raise ValueError("厘米距离数量必须与椭圆数量一致")

    message: Dict[str, Any] = {
        "mode": wire_value,
        "mode_name": VisionMode.RING_DETECTION.value,
        "success": bool(ellipses),
        "ellipse_count": len(ellipses),
        "ellipse_candidate_count": (
            len(ellipses)
            if ellipse_candidate_count is None
            else int(ellipse_candidate_count)
        ),
        "ellipses": [
            {
                "center_px": [round(value, 2) for value in ellipse.center],
                "major_axis_px": round(ellipse.major_axis_px, 2),
                "minor_axis_px": round(ellipse.minor_axis_px, 2),
                "angle_deg": round(ellipse.angle_deg, 2),
                "fit_error": round(ellipse.fit_error, 3),
            }
            for ellipse in ellipses
        ],
        "origin_center_px": (
            [round(float(value), 2) for value in ellipses[0].center]
            if ellipses
            else None
        ),
        "relative_centers_px": (
            [[round(x, 2), round(y, 2)] for x, y in geometry.relative_centers]
            if geometry is not None
            else []
        ),
        "distances_from_first_px": (
            [round(value, 2) for value in geometry.distances_from_first]
            if geometry is not None
            else []
        ),
        "distances_from_first_cm": [
            round(float(value), 2) if value is not None else None
            for value in centimeter_distances
        ],
        "collinear": geometry.collinear if geometry is not None else None,
        "collinearity_error_px": (
            round(geometry.collinearity_error_px, 2)
            if geometry is not None and geometry.collinearity_error_px is not None
            else None
        ),
        "collinearity_tolerance_px": round(float(collinearity_tolerance_px), 2),
    }
    if len(ellipses) == 3:
        message["middle_center_px"] = [
            round(float(value), 2) for value in ellipses[1].center
        ]
        message["left_right_y_delta_px"] = round(
            float(ellipses[0].center[1] - ellipses[2].center[1]),
            2,
        )
    return message


def mode_change_message(
    previous: VisionMode,
    current: VisionMode,
    wire_value: int,
) -> Dict[str, Any]:
    """Build the status record emitted for an actual semantic mode change."""

    return {
        "event": "mode_changed",
        "previous_mode_name": previous.value,
        "mode": wire_value,
        "mode_name": current.value,
    }


def color_change_message(previous: str, current: str) -> Dict[str, Any]:
    """Build the status record emitted after changing the surface target color."""

    return {
        "event": "surface_color_changed",
        "previous_color": previous,
        "color": current,
    }


def serialize_message(message: Mapping[str, Any]) -> str:
    """Serialize one result identically for stdout and optional ROS publishing."""

    return json.dumps(dict(message), ensure_ascii=False, separators=(",", ":"))


class VisionModeDispatcher:
    """Run exactly one vision branch for each shared camera frame."""

    def __init__(
        self,
        surface_pipeline: Any,
        ring_detector: Any,
        color: str,
        color_ranges_by_name: Optional[Mapping[str, Sequence[ColorRange]]] = None,
        ring_calibration: Optional[RingCalibrationConfig] = None,
    ) -> None:
        self.surface_pipeline = surface_pipeline
        self.ring_detector = ring_detector
        self.color = color
        self.ring_calibration = ring_calibration or RingCalibrationConfig()
        self._color_ranges_by_name = dict(color_ranges_by_name or {})
        if self._color_ranges_by_name and color not in self._color_ranges_by_name:
            raise ValueError(f"unconfigured initial surface color: {color}")
        self._mode: Optional[VisionMode] = None

    def switch_color(self, color: str) -> bool:
        """Apply one configured HSV target and reset tracking from the old color."""

        if color == self.color:
            return False
        ranges = self._color_ranges_by_name.get(color)
        if ranges is None:
            raise ValueError(f"unconfigured surface color: {color}")
        self.surface_pipeline.set_color_ranges(ranges)
        self.color = color
        return True

    def switch(self, mode: VisionMode) -> bool:
        """Record a mode and reset surface state after semantic transitions."""

        if mode == self._mode:
            return False
        if self._mode is VisionMode.RING_DETECTION:
            self.ring_detector.reset()
        if self._mode is not None:
            self.surface_pipeline.reset()
        self._mode = mode
        return True

    def process(
        self,
        mode: VisionMode,
        wire_value: int,
        camera_frame: CameraFrame,
        fps: float,
        render: bool,
    ) -> ModeFrameResult:
        """Process one camera frame in the requested mode."""

        if mode is VisionMode.IDLE:  #待击状态下，不进行处理，直接返回图像
            return ModeFrameResult(
                message=None,
                main_image=draw_idle(camera_frame.color) if render else None,
                auxiliary_images={},  #无附属窗口
            )

        if mode is VisionMode.SURFACE_TRACKING:  #表面圆追踪，显示附属窗口
            result = self.surface_pipeline.process(
                camera_frame.color,
                camera_frame.timestamp,
                camera_frame.depth_m,
                camera_frame.intrinsics,
                camera_frame.gravity_camera,
            )
            message = surface_message(
                wire_value,
                camera_frame.timestamp,
                self.color,
                result,
            )
            surface_status = surface_status_from_pipeline(self.color, result)
            if not render:
                return ModeFrameResult(
                    message,
                    None,
                    {},
                    surface_status=surface_status,
                )
            return ModeFrameResult(
                message,
                draw_debug(camera_frame.color, result, self.color, fps),
                {
                    "hsv-color-mask": result.color_mask,  #hsv色域切割
                    "depth-top-surface-mask": result.surface_mask,  #深度判断表面判断
                },
                surface_status=surface_status,
            )

        if mode is VisionMode.RING_DETECTION:  #色环识别模式，显示附属窗口
            rings = self.ring_detector.detect(camera_frame.color)  #检测出色环
            ellipse_candidates = self.ring_detector.ellipse_observations
            tolerance = self.ring_detector.config.collinearity_tolerance_px
            size_filtered_ellipses = filter_ellipses_by_major_axis_cm(
                ellipse_candidates,
                camera_frame.depth_m,
                camera_frame.intrinsics,
                self.ring_detector.config.ellipse_min_major_axis_cm,
                self.ring_detector.config.ellipse_max_major_axis_cm,
            )
            ellipses = (
                filter_ellipses_near_rings(
                    size_filtered_ellipses,
                    rings,
                    camera_frame.depth_m,
                    camera_frame.intrinsics,
                    self.ring_detector.config.ellipse_max_center_distance_cm,
                )
                if self.ring_detector.config.ellipse_require_hough_proximity
                else size_filtered_ellipses
            )
            distances_cm = measure_ellipse_distances_cm(
                ellipses,
                camera_frame.depth_m,
                camera_frame.intrinsics,
            )
            message = ring_message(
                wire_value,
                rings,
                tolerance,
                distances_cm,
                ellipses,
                len(ellipse_candidates),
            )  #和色环相关信息
            ring_status = calculate_ring_calibration(
                ellipses,
                camera_frame.depth_m,
                camera_frame.intrinsics,
                self.ring_calibration,
                tolerance,
                len(ellipse_candidates),
            )
            if not render:
                return ModeFrameResult(
                    message,
                    None,
                    {},
                    ring_status=ring_status,
                )
            intermediates = self.ring_detector.intermediates
            return ModeFrameResult(
                message,
                draw_rings(
                    camera_frame.color,
                    rings,
                    tolerance,
                    distances_cm,
                    ellipses,
                ),
                {
                    "ring-gray": intermediates["gray"],  #灰度图
                    "ring-gradient": intermediates["gradient"],  #梯度学处理图
                    "ring-threshold": intermediates["threshold"],  #阈值图
                },
                ring_status=ring_status,
            )

        raise ValueError(f"unsupported vision mode: {mode!r}")

"""ROS-independent typed measurements published by the vision runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Tuple

from .models import PipelineResult, Point2D, Point3D


@dataclass(frozen=True)
class RingCalibrationConfig:
    """Acceptance thresholds for eye-in-hand three-ring calibration."""

    target_distance_m: float = 0.30
    distance_tolerance_m: float = 0.01
    middle_center_tolerance_px: float = 10.0
    max_x_axis_angle_deg: float = 3.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RingCalibrationConfig":
        if not isinstance(data, Mapping):
            raise ValueError("ring_calibration configuration must be a mapping")
        raw = {
            "target_distance_m": data.get("target_distance_m", 0.30),
            "distance_tolerance_m": data.get("distance_tolerance_m", 0.01),
            "middle_center_tolerance_px": data.get(
                "middle_center_tolerance_px", 10.0
            ),
            "max_x_axis_angle_deg": data.get("max_x_axis_angle_deg", 3.0),
        }
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw.values()
        ):
            raise ValueError("ring_calibration values must be numeric")
        values = {name: float(value) for name, value in raw.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("ring_calibration values must be finite")
        if values["target_distance_m"] <= 0.0:
            raise ValueError("ring_calibration.target_distance_m must be positive")
        if values["distance_tolerance_m"] <= 0.0:
            raise ValueError("ring_calibration.distance_tolerance_m must be positive")
        if values["middle_center_tolerance_px"] < 0.0:
            raise ValueError(
                "ring_calibration.middle_center_tolerance_px must not be negative"
            )
        if not 0.0 <= values["max_x_axis_angle_deg"] <= 90.0:
            raise ValueError(
                "ring_calibration.max_x_axis_angle_deg must be in the range 0..90"
            )
        return cls(**values)


@dataclass(frozen=True)
class RingCenterStatus:
    """One final ellipse center and its optional camera-frame position."""

    order: int
    center_px: Point2D
    position_camera_m: Optional[Point3D]
    major_axis_px: float
    minor_axis_px: float
    ellipse_angle_deg: float
    fit_error: float


@dataclass(frozen=True)
class RingCalibrationStatus:
    """Complete measurement and readiness state for three-ring calibration."""

    active: bool
    ellipse_candidate_count: int
    rings: Tuple[RingCenterStatus, ...]
    collinear: bool
    collinearity_error_px: Optional[float]
    middle_error_px: Optional[Point2D]
    middle_center_camera_m: Optional[Point3D]
    middle_error_camera_m: Optional[Point3D]
    target_distance_m: float
    distance_error_m: Optional[float]
    line_direction_camera: Optional[Point3D]
    line_to_x_axis_angle_deg: Optional[float]
    center_aligned: bool
    distance_aligned: bool
    angle_aligned: bool
    calibration_ready: bool
    failure_reason: str

    @property
    def ring_count(self) -> int:
        return len(self.rings)

    @property
    def valid_3d_count(self) -> int:
        return sum(ring.position_camera_m is not None for ring in self.rings)

    @property
    def exactly_three(self) -> bool:
        return self.ring_count == 3

    @property
    def all_depth_valid(self) -> bool:
        return self.exactly_three and self.valid_3d_count == 3

    @property
    def middle_center_error_px(self) -> Optional[float]:
        if self.middle_error_px is None:
            return None
        return float(math.hypot(*self.middle_error_px))

    @classmethod
    def inactive(cls, target_distance_m: float) -> "RingCalibrationStatus":
        return cls(
            active=False,
            ellipse_candidate_count=0,
            rings=(),
            collinear=False,
            collinearity_error_px=None,
            middle_error_px=None,
            middle_center_camera_m=None,
            middle_error_camera_m=None,
            target_distance_m=float(target_distance_m),
            distance_error_m=None,
            line_direction_camera=None,
            line_to_x_axis_angle_deg=None,
            center_aligned=False,
            distance_aligned=False,
            angle_aligned=False,
            calibration_ready=False,
            failure_reason="ring_mode_inactive",
        )


@dataclass(frozen=True)
class SurfaceTrackingStatus:
    """Typed measurements from the colored top-surface tracker."""

    active: bool
    tracking_valid: bool
    measured: bool
    color: str
    center_px: Optional[Point2D]
    velocity_px_s: Optional[Point2D]
    speed_px_s: float
    radius_px: float
    missing_frames: int
    depth_applied: bool
    surface_center_camera_m: Optional[Point3D]
    surface_normal_camera: Optional[Point3D]
    surface_inlier_ratio: float
    normal_to_gravity_deg: Optional[float]

    @classmethod
    def inactive(cls, color: str) -> "SurfaceTrackingStatus":
        return cls(
            active=False,
            tracking_valid=False,
            measured=False,
            color=color,
            center_px=None,
            velocity_px_s=None,
            speed_px_s=0.0,
            radius_px=0.0,
            missing_frames=0,
            depth_applied=False,
            surface_center_camera_m=None,
            surface_normal_camera=None,
            surface_inlier_ratio=0.0,
            normal_to_gravity_deg=None,
        )


@dataclass(frozen=True)
class VisionStatusSnapshot:
    """One complete frame result before conversion to a ROS message."""

    frame_sequence: int
    current_mode_value: int
    current_mode_name: str
    selected_color: str
    camera_ok: bool
    result_available: bool
    processing_fps: float
    surface: SurfaceTrackingStatus
    ring: RingCalibrationStatus


def build_vision_snapshot(
    frame_sequence: int,
    mode_value: int,
    mode_name: str,
    selected_color: str,
    processing_fps: float,
    surface_status: Optional[SurfaceTrackingStatus],
    ring_status: Optional[RingCalibrationStatus],
    target_distance_m: float,
) -> VisionStatusSnapshot:
    """Build one complete status while clearing results from inactive modes."""

    if mode_name == "surface_tracking" and surface_status is None:
        raise ValueError("surface mode requires a surface status")
    if mode_name == "ring_detection" and ring_status is None:
        raise ValueError("ring mode requires a ring status")
    if mode_name not in {"idle", "surface_tracking", "ring_detection"}:
        raise ValueError(f"unsupported vision mode name: {mode_name}")
    return VisionStatusSnapshot(
        frame_sequence=int(frame_sequence),
        current_mode_value=int(mode_value),
        current_mode_name=mode_name,
        selected_color=selected_color,
        camera_ok=True,
        result_available=True,
        processing_fps=float(processing_fps),
        surface=(
            surface_status
            if mode_name == "surface_tracking"
            else SurfaceTrackingStatus.inactive(selected_color)
        ),
        ring=(
            ring_status
            if mode_name == "ring_detection"
            else RingCalibrationStatus.inactive(target_distance_m)
        ),
    )


def surface_status_from_pipeline(
    color: str,
    result: PipelineResult,
) -> SurfaceTrackingStatus:
    """Convert one surface-pipeline result without introducing ROS types."""

    estimate = result.estimate
    plane = result.surface_planes[0] if result.surface_planes else None
    selected = result.selected
    return SurfaceTrackingStatus(
        active=True,
        tracking_valid=estimate is not None,
        measured=False if estimate is None else estimate.measured,
        color=color,
        center_px=None if estimate is None else estimate.center,
        velocity_px_s=None if estimate is None else estimate.velocity,
        speed_px_s=(
            0.0 if estimate is None else float(math.hypot(*estimate.velocity))
        ),
        radius_px=0.0 if selected is None else float(selected.radius),
        missing_frames=0 if estimate is None else estimate.missing_frames,
        depth_applied=result.depth_applied,
        surface_center_camera_m=None if plane is None else plane.center,
        surface_normal_camera=None if plane is None else plane.normal,
        surface_inlier_ratio=0.0 if plane is None else plane.inlier_ratio,
        normal_to_gravity_deg=(
            None if plane is None else plane.normal_to_gravity_deg
        ),
    )

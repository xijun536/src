"""色环圆心的相对坐标、距离和共线关系。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .models import (
    CameraIntrinsics,
    EllipseObservation,
    Point2D,
    Point3D,
    RingObservation,
)
from .vision_status import (
    RingCalibrationConfig,
    RingCalibrationStatus,
    RingCenterStatus,
)


@dataclass(frozen=True)
class RingGeometry:
    """以第一个色环圆心为原点的二维几何结果。"""

    relative_centers: Tuple[Point2D, ...]
    distances_from_first: Tuple[float, ...]
    collinear: Optional[bool]
    collinearity_error_px: Optional[float]


def measure_ring_distances_cm(
    rings: Sequence[RingObservation],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    sample_radius_px: int = 2,
) -> Tuple[Optional[float], ...]:
    """利用对齐深度和相机内参计算各圆心到第一个圆心的三维距离。"""

    return _measure_center_distances_cm(
        tuple(ring.center for ring in rings),
        depth_m,
        intrinsics,
        sample_radius_px,
    )


def measure_ellipse_distances_cm(
    ellipses: Sequence[EllipseObservation],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    sample_radius_px: int = 2,
) -> Tuple[Optional[float], ...]:
    """计算各椭圆中心到第一个椭圆中心的三维距离。"""

    return _measure_center_distances_cm(
        tuple(ellipse.center for ellipse in ellipses),
        depth_m,
        intrinsics,
        sample_radius_px,
    )


def _measure_center_distances_cm(
    centers: Sequence[Point2D],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    sample_radius_px: int,
) -> Tuple[Optional[float], ...]:
    unavailable = (None,) * len(centers)
    if not centers:
        return ()
    points = [
        deproject_center(center, depth_m, intrinsics, sample_radius_px)
        for center in centers
    ]

    origin = points[0]
    if origin is None:
        return unavailable
    return tuple(
        None if point is None else math.dist(origin, point) * 100.0
        for point in points
    )


def filter_ellipses_near_rings(
    ellipses: Sequence[EllipseObservation],
    rings: Sequence[RingObservation],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    max_distance_cm: float,
    sample_radius_px: int = 2,
) -> Tuple[EllipseObservation, ...]:
    """只保留三维圆心到任意Hough圆心不超过指定距离的椭圆。"""

    maximum = float(max_distance_cm)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("椭圆与Hough圆心的最大距离必须是正有限数")
    if not ellipses or not rings:
        return ()

    ring_points = []
    for ring in rings:
        point = deproject_center(
            ring.center,
            depth_m,
            intrinsics,
            sample_radius_px,
        )
        if point is not None:
            ring_points.append(point)
    if not ring_points:
        return ()

    accepted = []
    for ellipse in ellipses:
        point = deproject_center(
            ellipse.center,
            depth_m,
            intrinsics,
            sample_radius_px,
        )
        if point is not None and any(
            math.dist(point, ring_point) * 100.0 <= maximum
            for ring_point in ring_points
        ):
            accepted.append(ellipse)
    return tuple(accepted)


def filter_ellipses_by_major_axis_cm(
    ellipses: Sequence[EllipseObservation],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    min_major_axis_cm: float,
    max_major_axis_cm: float,
    sample_radius_px: int = 2,
) -> Tuple[EllipseObservation, ...]:
    """只保留三维实体长轴位于指定厘米范围内的椭圆。"""

    minimum = float(min_major_axis_cm)
    maximum = float(max_major_axis_cm)
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError("椭圆长轴范围必须满足 0 < min <= max 且数值有限")

    accepted = []
    for ellipse in ellipses:
        angle = math.radians(ellipse.angle_deg)
        half_axis = ellipse.major_axis_px / 2.0
        column_offset = math.cos(angle) * half_axis
        row_offset = math.sin(angle) * half_axis
        first = deproject_center(
            (
                ellipse.center[0] - column_offset,
                ellipse.center[1] - row_offset,
            ),
            depth_m,
            intrinsics,
            sample_radius_px,
        )
        second = deproject_center(
            (
                ellipse.center[0] + column_offset,
                ellipse.center[1] + row_offset,
            ),
            depth_m,
            intrinsics,
            sample_radius_px,
        )
        if first is None or second is None:
            continue
        major_axis_cm = math.dist(first, second) * 100.0
        meets_minimum = major_axis_cm >= minimum or math.isclose(
            major_axis_cm,
            minimum,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        meets_maximum = major_axis_cm <= maximum or math.isclose(
            major_axis_cm,
            maximum,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        if meets_minimum and meets_maximum:
            accepted.append(ellipse)
    return tuple(accepted)


def deproject_center(
    center: Point2D,
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    sample_radius_px: int,
) -> Optional[Point3D]:
    """用圆心邻域的有效深度中位数将一个图像点反投影到相机坐标系。"""

    if (
        depth_m is None
        or intrinsics is None
        or not isinstance(depth_m, np.ndarray)
        or depth_m.ndim != 2
        or sample_radius_px < 0
        or not math.isfinite(intrinsics.fx)
        or not math.isfinite(intrinsics.fy)
        or not math.isfinite(intrinsics.ppx)
        or not math.isfinite(intrinsics.ppy)
        or intrinsics.fx <= 0.0
        or intrinsics.fy <= 0.0
    ):
        return None

    column, row = center
    height, width = depth_m.shape
    if (
        not math.isfinite(float(column))
        or not math.isfinite(float(row))
        or not 0.0 <= column < width
        or not 0.0 <= row < height
    ):
        return None
    sample_column = int(round(column))
    sample_row = int(round(row))
    left = max(0, sample_column - sample_radius_px)
    right = min(width, sample_column + sample_radius_px + 1)
    top = max(0, sample_row - sample_radius_px)
    bottom = min(height, sample_row + sample_radius_px + 1)
    sample = depth_m[top:bottom, left:right]
    valid = sample[np.isfinite(sample) & (sample > 0.0)]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    x = (float(column) - intrinsics.ppx) * z / intrinsics.fx
    y = (float(row) - intrinsics.ppy) * z / intrinsics.fy
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    return (x, y, z)


def calculate_ring_calibration(
    ellipses: Sequence[EllipseObservation],
    depth_m: Optional[np.ndarray],
    intrinsics: Optional[CameraIntrinsics],
    config: RingCalibrationConfig,
    collinearity_tolerance_px: float,
    ellipse_candidate_count: int,
) -> RingCalibrationStatus:
    """Calculate camera-frame errors and readiness for exactly three ellipses."""

    ordered = tuple(sorted(ellipses, key=lambda item: item.center[0]))
    geometry = analyze_ellipse_geometry(ordered, collinearity_tolerance_px)
    rings = tuple(
        RingCenterStatus(
            order=index,
            center_px=(float(ellipse.center[0]), float(ellipse.center[1])),
            position_camera_m=deproject_center(
                ellipse.center,
                depth_m,
                intrinsics,
                sample_radius_px=2,
            ),
            major_axis_px=float(ellipse.major_axis_px),
            minor_axis_px=float(ellipse.minor_axis_px),
            ellipse_angle_deg=float(ellipse.angle_deg),
            fit_error=float(ellipse.fit_error),
        )
        for index, ellipse in enumerate(ordered, start=1)
    )

    exactly_three = len(rings) == 3
    all_depth_valid = exactly_three and all(
        ring.position_camera_m is not None for ring in rings
    )
    collinear = bool(geometry is not None and geometry.collinear)
    middle_error_px: Optional[Point2D] = None
    middle_point: Optional[Point3D] = None
    middle_error: Optional[Point3D] = None
    distance_error: Optional[float] = None
    line_direction: Optional[Point3D] = None
    line_angle_deg: Optional[float] = None

    if exactly_three and intrinsics is not None:
        middle_error_px = (
            float(rings[1].center_px[0] - intrinsics.ppx),
            float(rings[1].center_px[1] - intrinsics.ppy),
        )

    if all_depth_valid:
        points = [
            np.asarray(ring.position_camera_m, dtype=np.float64)
            for ring in rings
        ]
        middle_point = tuple(float(value) for value in points[1])
        middle_error_array = points[1] - np.asarray(
            (0.0, 0.0, config.target_distance_m),
            dtype=np.float64,
        )
        middle_error = tuple(float(value) for value in middle_error_array)
        distance_error = float(middle_error_array[2])

        direction = points[2] - points[0]
        direction_norm = float(np.linalg.norm(direction))
        if math.isfinite(direction_norm) and direction_norm >= 1e-9:
            direction /= direction_norm
            line_direction = tuple(float(value) for value in direction)
            line_angle_deg = float(
                np.degrees(
                    np.arccos(np.clip(float(direction[0]), -1.0, 1.0))
                )
            )

    center_aligned = (
        middle_error_px is not None
        and math.hypot(*middle_error_px) <= config.middle_center_tolerance_px
    )
    distance_aligned = (
        distance_error is not None
        and (
            abs(distance_error) <= config.distance_tolerance_m
            or math.isclose(
                abs(distance_error),
                config.distance_tolerance_m,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
    )
    angle_aligned = (
        line_angle_deg is not None
        and (
            line_angle_deg <= config.max_x_axis_angle_deg
            or math.isclose(
                line_angle_deg,
                config.max_x_axis_angle_deg,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
    )
    calibration_ready = (
        exactly_three
        and all_depth_valid
        and collinear
        and center_aligned
        and distance_aligned
        and angle_aligned
    )
    failure_reason = (
        "waiting_for_three_rings"
        if not exactly_three
        else "invalid_ring_depth"
        if not all_depth_valid
        else "rings_not_collinear"
        if not collinear
        else "middle_ring_off_center"
        if not center_aligned
        else "distance_out_of_range"
        if not distance_aligned
        else "line_not_parallel"
        if not angle_aligned
        else "ready"
    )

    return RingCalibrationStatus(
        active=True,
        ellipse_candidate_count=int(ellipse_candidate_count),
        rings=rings,
        collinear=collinear,
        collinearity_error_px=(
            None if geometry is None else geometry.collinearity_error_px
        ),
        middle_error_px=middle_error_px,
        middle_center_camera_m=middle_point,
        middle_error_camera_m=middle_error,
        target_distance_m=config.target_distance_m,
        distance_error_m=distance_error,
        line_direction_camera=line_direction,
        line_to_x_axis_angle_deg=line_angle_deg,
        center_aligned=center_aligned,
        distance_aligned=distance_aligned,
        angle_aligned=angle_aligned,
        calibration_ready=calibration_ready,
        failure_reason=failure_reason,
    )


def analyze_ring_geometry(
    rings: Sequence[RingObservation],
    collinearity_tolerance_px: float,
) -> Optional[RingGeometry]:
    """计算相对圆心；恰好三个圆时计算第二点到首尾直线的距离。"""

    return _analyze_centers(
        tuple(ring.center for ring in rings),
        collinearity_tolerance_px,
    )


def analyze_ellipse_geometry(
    ellipses: Sequence[EllipseObservation],
    collinearity_tolerance_px: float,
) -> Optional[RingGeometry]:
    """计算椭圆中心的相对坐标、距离和三点共线关系。"""

    return _analyze_centers(
        tuple(ellipse.center for ellipse in ellipses),
        collinearity_tolerance_px,
    )


def _analyze_centers(
    centers: Sequence[Point2D],
    collinearity_tolerance_px: float,
) -> Optional[RingGeometry]:
    """计算一组二维中心点相对第一个点的几何关系。"""

    tolerance = float(collinearity_tolerance_px)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("共线容差必须是非负有限数")
    if not centers:
        return None

    origin_x, origin_y = centers[0]
    relative = tuple(
        (float(center[0] - origin_x), float(center[1] - origin_y))
        for center in centers
    )
    distances = tuple(math.hypot(x, y) for x, y in relative)

    collinear: Optional[bool] = None
    error: Optional[float] = None
    if len(relative) == 3:
        second_x, second_y = relative[1]
        third_x, third_y = relative[2]
        baseline_length = math.hypot(third_x, third_y)
        if baseline_length > 1e-9:
            error = abs(third_x * second_y - third_y * second_x) / baseline_length
        else:
            error = math.hypot(second_x, second_y)
        collinear = error <= tolerance

    return RingGeometry(
        relative_centers=relative,
        distances_from_first=distances,
        collinear=collinear,
        collinearity_error_px=error,
    )

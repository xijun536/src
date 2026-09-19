from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from d435i_vision.models import (
    CameraIntrinsics,
    CircleObservation,
    EllipseObservation,
    PipelineResult,
    SurfacePlaneObservation,
    TrackEstimate,
)
from d435i_vision.ring_geometry import calculate_ring_calibration
from d435i_vision.vision_status import (
    build_vision_snapshot,
    RingCalibrationConfig,
    RingCalibrationStatus,
    SurfaceTrackingStatus,
    surface_status_from_pipeline,
)


INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=600.0,
    fy=600.0,
    ppx=320.0,
    ppy=240.0,
)


def ellipses_at(*centers: tuple[float, float]) -> tuple[EllipseObservation, ...]:
    return tuple(
        EllipseObservation(center, 60.0, 45.0, 0.0, 0.02)
        for center in centers
    )


def calibrate(
    centers: tuple[tuple[float, float], ...] = (
        (220.0, 240.0),
        (320.0, 240.0),
        (420.0, 240.0),
    ),
    depth_value: float = 0.30,
) -> RingCalibrationStatus:
    depth = np.full((480, 640), depth_value, dtype=np.float64)
    return calculate_ring_calibration(
        ellipses_at(*centers),
        depth,
        INTRINSICS,
        RingCalibrationConfig(),
        collinearity_tolerance_px=5.0,
        ellipse_candidate_count=len(centers),
    )


def test_ring_calibration_config_uses_approved_defaults() -> None:
    config = RingCalibrationConfig.from_mapping({})

    assert config.target_distance_m == 0.30
    assert config.distance_tolerance_m == 0.01
    assert config.middle_center_tolerance_px == 10.0
    assert config.max_x_axis_angle_deg == 3.0


@pytest.mark.parametrize(
    "data",
    [
        {"target_distance_m": True},
        {"target_distance_m": 0.0},
        {"distance_tolerance_m": 0.0},
        {"middle_center_tolerance_px": -1.0},
        {"max_x_axis_angle_deg": 91.0},
        {"max_x_axis_angle_deg": float("nan")},
    ],
)
def test_ring_calibration_config_rejects_invalid_values(data) -> None:
    with pytest.raises(ValueError):
        RingCalibrationConfig.from_mapping(data)


def test_three_centered_rings_at_target_are_ready() -> None:
    result = calibrate()

    assert result.ring_count == 3
    assert result.valid_3d_count == 3
    assert result.exactly_three is True
    assert result.all_depth_valid is True
    assert result.collinear is True
    assert result.middle_error_px == pytest.approx((0.0, 0.0))
    assert result.middle_center_error_px == pytest.approx(0.0)
    assert result.middle_center_camera_m == pytest.approx((0.0, 0.0, 0.30))
    assert result.middle_error_camera_m == pytest.approx((0.0, 0.0, 0.0))
    assert result.line_direction_camera == pytest.approx((1.0, 0.0, 0.0))
    assert result.line_to_x_axis_angle_deg == pytest.approx(0.0)
    assert result.calibration_ready is True
    assert result.failure_reason == "ready"


@pytest.mark.parametrize("depth", [0.29, 0.30, 0.31])
def test_distance_tolerance_boundaries_are_inclusive(depth: float) -> None:
    result = calibrate(depth_value=depth)

    assert result.distance_aligned is True


@pytest.mark.parametrize("depth", [0.289, 0.311])
def test_distance_outside_one_centimeter_fails(depth: float) -> None:
    result = calibrate(depth_value=depth)

    assert result.distance_aligned is False
    assert result.calibration_ready is False
    assert result.failure_reason == "distance_out_of_range"


def test_middle_center_above_ten_pixels_fails() -> None:
    result = calibrate(((220.0, 240.0), (331.0, 240.0), (420.0, 240.0)))

    assert result.center_aligned is False
    assert result.middle_center_error_px == pytest.approx(11.0)
    assert result.failure_reason == "middle_ring_off_center"


def test_line_above_three_degrees_fails() -> None:
    result = calibrate(((220.0, 229.0), (320.0, 240.0), (420.0, 251.0)))

    assert result.collinear is True
    assert result.line_to_x_axis_angle_deg > 3.0
    assert result.angle_aligned is False
    assert result.failure_reason == "line_not_parallel"


def test_non_collinear_centers_fail_before_center_error() -> None:
    result = calibrate(((220.0, 240.0), (320.0, 250.0), (420.0, 240.0)))

    assert result.collinear is False
    assert result.center_aligned is True
    assert result.failure_reason == "rings_not_collinear"


def test_invalid_depth_preserves_2d_centers_but_blocks_calibration() -> None:
    result = calibrate(depth_value=float("nan"))

    assert result.ring_count == 3
    assert result.valid_3d_count == 0
    assert result.all_depth_valid is False
    assert all(ring.position_camera_m is None for ring in result.rings)
    assert result.calibration_ready is False
    assert result.failure_reason == "invalid_ring_depth"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fx", float("nan")),
        ("fx", float("inf")),
        ("fy", float("nan")),
        ("fy", float("inf")),
        ("ppx", float("nan")),
        ("ppx", float("inf")),
        ("ppy", float("nan")),
        ("ppy", float("inf")),
    ],
)
def test_nonfinite_intrinsics_never_produce_valid_3d_positions(
    field: str,
    value: float,
) -> None:
    intrinsics = replace(INTRINSICS, **{field: value})
    depth = np.full((480, 640), 0.30, dtype=np.float64)

    result = calculate_ring_calibration(
        ellipses_at((220.0, 240.0), (320.0, 240.0), (420.0, 240.0)),
        depth,
        intrinsics,
        RingCalibrationConfig(),
        collinearity_tolerance_px=5.0,
        ellipse_candidate_count=3,
    )

    assert result.valid_3d_count == 0
    assert result.all_depth_valid is False
    assert result.failure_reason == "invalid_ring_depth"


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_any_count_other_than_three_waits_for_three_rings(count: int) -> None:
    centers = tuple((100.0 + index * 80.0, 240.0) for index in range(count))
    result = calibrate(centers)

    assert result.ring_count == count
    assert result.exactly_three is False
    assert result.calibration_ready is False
    assert result.failure_reason == "waiting_for_three_rings"


def test_inactive_ring_status_never_reuses_measurement() -> None:
    result = RingCalibrationStatus.inactive(0.30)

    assert result.active is False
    assert result.rings == ()
    assert result.target_distance_m == 0.30
    assert result.calibration_ready is False
    assert result.failure_reason == "ring_mode_inactive"


def pipeline_result(with_estimate: bool) -> PipelineResult:
    mask = np.zeros((4, 5), dtype=np.uint8)
    return PipelineResult(
        blurred_frame=np.zeros((4, 5, 3), dtype=np.uint8),
        color_mask=mask,
        surface_mask=mask,
        depth_applied=True,
        surface_planes=[
            SurfacePlaneObservation(
                normal=(0.0, 0.0, -1.0),
                offset=0.30,
                center=(0.01, -0.02, 0.30),
                inlier_count=90,
                candidate_count=100,
                gravity_alignment=1.0,
            )
        ],
        observations=[],
        selected=CircleObservation((100.0, 200.0), 25.0, 1000.0, 0.9, 0.8),
        estimate=(
            TrackEstimate((100.0, 200.0), (3.0, 4.0), False, 2)
            if with_estimate
            else None
        ),
    )


def test_surface_status_preserves_tracking_and_plane_measurements() -> None:
    status = surface_status_from_pipeline("green", pipeline_result(True))

    assert status.active is True
    assert status.tracking_valid is True
    assert status.measured is False
    assert status.center_px == (100.0, 200.0)
    assert status.velocity_px_s == (3.0, 4.0)
    assert status.speed_px_s == 5.0
    assert status.radius_px == 25.0
    assert status.surface_center_camera_m == (0.01, -0.02, 0.30)
    assert status.surface_normal_camera == (0.0, 0.0, -1.0)
    assert status.surface_inlier_ratio == 0.9
    assert status.normal_to_gravity_deg == pytest.approx(0.0)


def test_surface_status_remains_active_without_tracking_measurement() -> None:
    status = surface_status_from_pipeline("yellow", pipeline_result(False))

    assert status.active is True
    assert status.tracking_valid is False
    assert status.measured is False
    assert status.center_px is None
    assert status.velocity_px_s is None
    assert status.speed_px_s == 0.0
    assert math.isfinite(status.surface_inlier_ratio)


def test_inactive_surface_status_contains_no_old_measurement() -> None:
    status = SurfaceTrackingStatus.inactive("blue")

    assert status.active is False
    assert status.tracking_valid is False
    assert status.color == "blue"
    assert status.center_px is None
    assert status.surface_center_camera_m is None


def test_snapshot_builder_activates_only_the_current_mode_result() -> None:
    surface = surface_status_from_pipeline("green", pipeline_result(True))
    ring = calibrate()

    surface_snapshot = build_vision_snapshot(
        1, 1, "surface_tracking", "green", 30.0, surface, None, 0.30
    )
    ring_snapshot = build_vision_snapshot(
        2, 2, "ring_detection", "green", 29.0, None, ring, 0.30
    )
    idle_snapshot = build_vision_snapshot(
        3, 0, "idle", "green", 28.0, None, None, 0.30
    )

    assert surface_snapshot.surface is surface
    assert surface_snapshot.ring.active is False
    assert ring_snapshot.surface.active is False
    assert ring_snapshot.ring is ring
    assert idle_snapshot.result_available is True
    assert idle_snapshot.surface.active is False
    assert idle_snapshot.ring.active is False


@pytest.mark.parametrize(
    ("mode_name", "surface", "ring"),
    [
        ("surface_tracking", None, None),
        ("ring_detection", None, None),
        ("unknown", None, None),
    ],
)
def test_snapshot_builder_rejects_missing_or_unknown_active_results(
    mode_name,
    surface,
    ring,
) -> None:
    with pytest.raises(ValueError):
        build_vision_snapshot(
            1, 1, mode_name, "green", 30.0, surface, ring, 0.30
        )

from __future__ import annotations

import math

import numpy as np
import pytest

from d435i_vision import ring_geometry
from d435i_vision.models import CameraIntrinsics, EllipseObservation, RingObservation
from d435i_vision.ring_geometry import analyze_ring_geometry


def test_three_collinear_rings_use_first_center_as_origin():
    rings = (
        RingObservation((100, 100), 20),
        RingObservation((130, 104), 20),
        RingObservation((160, 108), 20),
    )

    geometry = analyze_ring_geometry(rings, collinearity_tolerance_px=5.0)

    assert geometry is not None
    assert geometry.relative_centers == ((0, 0), (30, 4), (60, 8))
    assert geometry.distances_from_first[0] == 0.0
    assert math.isclose(geometry.distances_from_first[1], math.hypot(30, 4))
    assert math.isclose(geometry.distances_from_first[2], math.hypot(60, 8))
    assert geometry.collinearity_error_px == 0.0
    assert geometry.collinear is True


def test_second_center_distance_to_first_third_line_controls_collinearity():
    rings = (
        RingObservation((100, 100), 20),
        RingObservation((130, 112), 20),
        RingObservation((160, 100), 20),
    )

    geometry = analyze_ring_geometry(rings, collinearity_tolerance_px=5.0)

    assert geometry is not None
    assert geometry.collinearity_error_px == 12.0
    assert geometry.collinear is False


def test_fewer_than_three_rings_have_relative_geometry_but_no_collinearity_result():
    rings = (
        RingObservation((20, 30), 10),
        RingObservation((23, 34), 10),
    )

    geometry = analyze_ring_geometry(rings, collinearity_tolerance_px=5.0)

    assert geometry is not None
    assert geometry.relative_centers == ((0, 0), (3, 4))
    assert geometry.distances_from_first == (0.0, 5.0)
    assert geometry.collinear is None
    assert geometry.collinearity_error_px is None


def test_depth_and_intrinsics_convert_pixel_centers_to_3d_centimeters():
    rings = (
        RingObservation((10, 10), 4),
        RingObservation((13, 14), 4),
    )
    depth_m = np.ones((30, 30), dtype=np.float32)
    intrinsics = CameraIntrinsics(
        width=30,
        height=30,
        fx=100.0,
        fy=100.0,
        ppx=0.0,
        ppy=0.0,
    )

    distances = ring_geometry.measure_ring_distances_cm(
        rings,
        depth_m,
        intrinsics,
    )

    assert distances == (0.0, 5.0)


def test_depth_sampling_uses_valid_median_and_ignores_invalid_center_depth():
    rings = (
        RingObservation((5, 5), 3),
        RingObservation((15, 5), 3),
    )
    depth_m = np.zeros((20, 25), dtype=np.float32)
    depth_m[3:8, 3:8] = 1.0
    depth_m[3:8, 13:18] = 1.0
    depth_m[5, 5] = 0.0
    depth_m[5, 15] = np.nan
    depth_m[3, 13] = np.inf
    intrinsics = CameraIntrinsics(25, 20, 100.0, 100.0, 0.0, 0.0)

    distances = ring_geometry.measure_ring_distances_cm(
        rings,
        depth_m,
        intrinsics,
    )

    assert distances == (0.0, 10.0)


def test_missing_or_invalid_depth_returns_unavailable_centimeter_distances():
    rings = (
        RingObservation((5, 5), 3),
        RingObservation((15, 5), 3),
    )
    intrinsics = CameraIntrinsics(25, 20, 100.0, 100.0, 0.0, 0.0)

    assert ring_geometry.measure_ring_distances_cm(rings, None, intrinsics) == (
        None,
        None,
    )
    assert ring_geometry.measure_ring_distances_cm(
        rings,
        np.zeros((20, 25), dtype=np.float32),
        intrinsics,
    ) == (None, None)


def test_only_ellipses_within_two_3d_centimeters_of_any_hough_center_are_kept():
    rings = (RingObservation((10, 10), 4),)
    near = EllipseObservation((11.0, 10.0), 20.0, 12.0, 0.0, 0.02)
    far = EllipseObservation((13.0, 10.0), 20.0, 12.0, 0.0, 0.02)
    depth_m = np.ones((30, 30), dtype=np.float32)
    intrinsics = CameraIntrinsics(30, 30, 100.0, 100.0, 0.0, 0.0)

    result = ring_geometry.filter_ellipses_near_rings(
        (near, far),
        rings,
        depth_m,
        intrinsics,
        max_distance_cm=2.0,
    )

    assert result == (near,)


def test_ellipse_without_valid_depth_is_rejected_by_3d_proximity_filter():
    rings = (RingObservation((5, 5), 4),)
    ellipse = EllipseObservation((15.0, 5.0), 20.0, 12.0, 0.0, 0.02)
    depth_m = np.zeros((20, 25), dtype=np.float32)
    depth_m[3:8, 3:8] = 1.0
    intrinsics = CameraIntrinsics(25, 20, 100.0, 100.0, 0.0, 0.0)

    result = ring_geometry.filter_ellipses_near_rings(
        (ellipse,),
        rings,
        depth_m,
        intrinsics,
        max_distance_cm=2.0,
    )

    assert result == ()


def test_major_axis_filter_keeps_only_inclusive_five_to_eleven_cm_range():
    too_short = EllipseObservation((30.0, 20.0), 4.9, 3.0, 0.0, 0.02)
    minimum = EllipseObservation((60.0, 20.0), 5.0, 3.0, 0.0, 0.02)
    middle = EllipseObservation((90.0, 20.0), 8.0, 4.0, 0.0, 0.02)
    maximum = EllipseObservation((120.0, 20.0), 11.0, 5.0, 0.0, 0.02)
    too_long = EllipseObservation((150.0, 20.0), 11.1, 5.0, 0.0, 0.02)
    depth_m = np.ones((50, 180), dtype=np.float32)
    intrinsics = CameraIntrinsics(180, 50, 100.0, 100.0, 90.0, 25.0)

    result = ring_geometry.filter_ellipses_by_major_axis_cm(
        (too_short, minimum, middle, maximum, too_long),
        depth_m,
        intrinsics,
        min_major_axis_cm=5.0,
        max_major_axis_cm=11.0,
    )

    assert result == (minimum, middle, maximum)


def test_major_axis_filter_rejects_ellipse_with_missing_endpoint_depth():
    ellipse = EllipseObservation((50.0, 20.0), 10.0, 5.0, 0.0, 0.02)
    depth_m = np.ones((50, 100), dtype=np.float32)
    depth_m[18:23, 53:58] = 0.0
    intrinsics = CameraIntrinsics(100, 50, 100.0, 100.0, 50.0, 25.0)

    result = ring_geometry.filter_ellipses_by_major_axis_cm(
        (ellipse,),
        depth_m,
        intrinsics,
        min_major_axis_cm=5.0,
        max_major_axis_cm=11.0,
    )

    assert result == ()


def test_major_axis_filter_rejects_endpoint_outside_depth_image():
    ellipse = EllipseObservation((-50.0, 50.0), 10.0, 5.0, 0.0, 0.02)
    depth_m = np.ones((100, 100), dtype=np.float32)
    intrinsics = CameraIntrinsics(100, 100, 100.0, 100.0, 50.0, 50.0)

    result = ring_geometry.filter_ellipses_by_major_axis_cm(
        (ellipse,),
        depth_m,
        intrinsics,
        min_major_axis_cm=5.0,
        max_major_axis_cm=11.0,
    )

    assert result == ()


def test_major_axis_filter_uses_ellipse_angle_and_vertical_focal_length():
    ellipse = EllipseObservation((50.0, 50.0), 20.0, 8.0, 90.0, 0.02)
    depth_m = np.ones((100, 100), dtype=np.float32)
    intrinsics = CameraIntrinsics(100, 100, 100.0, 200.0, 50.0, 50.0)

    result = ring_geometry.filter_ellipses_by_major_axis_cm(
        (ellipse,),
        depth_m,
        intrinsics,
        min_major_axis_cm=9.0,
        max_major_axis_cm=11.0,
    )

    assert result == (ellipse,)


@pytest.mark.parametrize(
    "minimum, maximum",
    [(-1.0, 11.0), (0.0, 11.0), (12.0, 11.0), (float("nan"), 11.0)],
)
def test_major_axis_filter_rejects_invalid_centimeter_range(minimum, maximum):
    with pytest.raises(ValueError):
        ring_geometry.filter_ellipses_by_major_axis_cm(
            (),
            None,
            None,
            min_major_axis_cm=minimum,
            max_major_axis_cm=maximum,
        )

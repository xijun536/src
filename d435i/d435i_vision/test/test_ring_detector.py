from __future__ import annotations

import numpy as np
import pytest

from d435i_vision.algorithm.ring_detector import RingConfig
from d435i_vision.algorithm.ring_tracker import RingTracker
from d435i_vision.models import RingObservation


def test_ring_config_accepts_valid_mapping():
    config = RingConfig.from_mapping({"dp": 1.5, "min_distance": 50,
                                      "param1": 100, "param2": 0.95,
                                      "min_radius": 15, "max_radius": 50,
                                      "collinearity_tolerance_px": 3.5,
                                      "ellipse_enabled": True,
                                      "ellipse_min_contour_points": 20,
                                      "ellipse_min_axis_ratio": 0.35,
                                      "ellipse_max_fit_error": 0.2,
                                      "ellipse_max_center_distance_cm": 2.0,
                                      "ellipse_require_hough_proximity": False,
                                      "tracking_enabled": True,
                                      "tracking_smoothing": 0.45,
                                      "tracking_max_match_distance_px": 40.0,
                                      "tracking_max_missing_frames": 5,
                                      "ellipse_binary_threshold": 127,
                                      "ellipse_edge_sample_count": 72,
                                      "ellipse_edge_search_radius_px": 3,
                                      "ellipse_min_edge_support": 0.50,
                                      "ellipse_coverage_bin_count": 24,
                                      "ellipse_min_covered_bins": 14,
                                      "ellipse_min_major_axis_cm": 5.0,
                                      "ellipse_max_major_axis_cm": 11.0})
    assert config.min_radius == 15
    assert config.collinearity_tolerance_px == 3.5
    assert config.ellipse_enabled is True
    assert config.ellipse_min_contour_points == 20
    assert config.ellipse_min_axis_ratio == 0.35
    assert config.ellipse_max_fit_error == 0.2
    assert config.ellipse_max_center_distance_cm == 2.0
    assert config.ellipse_require_hough_proximity is False
    assert config.tracking_enabled is True
    assert config.tracking_smoothing == 0.45
    assert config.tracking_max_match_distance_px == 40.0
    assert config.tracking_max_missing_frames == 5
    assert config.ellipse_binary_threshold == 127
    assert config.ellipse_edge_sample_count == 72
    assert config.ellipse_edge_search_radius_px == 3
    assert config.ellipse_min_edge_support == 0.50
    assert config.ellipse_coverage_bin_count == 24
    assert config.ellipse_min_covered_bins == 14
    assert config.ellipse_min_major_axis_cm == 5.0
    assert config.ellipse_max_major_axis_cm == 11.0


def test_ring_config_preserves_existing_positional_tracking_arguments():
    config = RingConfig(
        1.5,
        50.0,
        100.0,
        0.95,
        15,
        50,
        5.0,
        True,
        20,
        0.35,
        0.20,
        2.0,
        False,
        0.30,
        55.0,
        7,
    )

    assert config.tracking_enabled is False
    assert config.tracking_smoothing == 0.30
    assert config.tracking_max_match_distance_px == 55.0
    assert config.tracking_max_missing_frames == 7
    assert config.ellipse_require_hough_proximity is False


@pytest.mark.parametrize("override", [
    {"dp": 0}, {"min_distance": 0}, {"param1": 0}, {"param2": 0},
    {"param2": 1.1}, {"min_radius": -1},
    {"min_radius": 50, "max_radius": 50},
    {"ellipse_min_contour_points": 4},
    {"ellipse_min_axis_ratio": 0.0}, {"ellipse_min_axis_ratio": 1.1},
    {"ellipse_max_fit_error": 0.0}, {"ellipse_max_fit_error": 1.1},
    {"ellipse_max_center_distance_cm": 0.0},
    {"tracking_smoothing": 0.0}, {"tracking_smoothing": 1.1},
    {"tracking_max_match_distance_px": 0.0},
    {"tracking_max_missing_frames": -1},
    {"ellipse_binary_threshold": -1}, {"ellipse_binary_threshold": 256},
    {"ellipse_edge_sample_count": 7},
    {"ellipse_edge_sample_count": 4097},
    {"ellipse_edge_search_radius_px": -1},
    {"ellipse_min_edge_support": 0.0},
    {"ellipse_min_edge_support": 1.1},
    {"ellipse_coverage_bin_count": 3},
    {"ellipse_coverage_bin_count": 73},
    {"ellipse_min_covered_bins": 0},
    {"ellipse_min_covered_bins": 25},
    {"ellipse_min_major_axis_cm": 0.0},
    {"ellipse_max_major_axis_cm": 0.0},
    {"ellipse_min_major_axis_cm": 12.0, "ellipse_max_major_axis_cm": 11.0},
])
def test_ring_config_rejects_invalid_values(override):
    values = {"dp": 1.5, "min_distance": 50, "param1": 100,
              "param2": 0.95, "min_radius": 15, "max_radius": 50}
    values.update(override)
    with pytest.raises(ValueError):
        RingConfig.from_mapping(values)


@pytest.mark.parametrize("field, value", [
    ("dp", True), ("dp", "1.5"), ("dp", float("nan")), ("dp", float("inf")),
    ("min_distance", True), ("param1", "100"), ("param1", float("inf")),
    ("param2", "0.95"), ("param2", float("nan")),
    ("collinearity_tolerance_px", True),
    ("collinearity_tolerance_px", "5"),
    ("collinearity_tolerance_px", float("inf")),
    ("ellipse_enabled", 1),
    ("ellipse_require_hough_proximity", 1),
    ("ellipse_min_contour_points", 20.0),
    ("ellipse_min_axis_ratio", True),
    ("ellipse_min_axis_ratio", "0.35"),
    ("ellipse_max_fit_error", float("nan")),
    ("ellipse_max_center_distance_cm", True),
    ("ellipse_max_center_distance_cm", "2.0"),
    ("ellipse_max_center_distance_cm", float("inf")),
    ("tracking_enabled", 1),
    ("tracking_smoothing", True),
    ("tracking_smoothing", "0.45"),
    ("tracking_max_match_distance_px", float("inf")),
    ("tracking_max_missing_frames", 5.0),
    ("tracking_max_missing_frames", True),
    ("ellipse_binary_threshold", 127.0),
    ("ellipse_binary_threshold", True),
    ("ellipse_edge_sample_count", 72.0),
    ("ellipse_edge_search_radius_px", 3.0),
    ("ellipse_min_edge_support", True),
    ("ellipse_min_edge_support", "0.5"),
    ("ellipse_min_edge_support", float("nan")),
    ("ellipse_coverage_bin_count", 24.0),
    ("ellipse_min_covered_bins", 14.0),
    ("ellipse_min_major_axis_cm", True),
    ("ellipse_min_major_axis_cm", "5.0"),
    ("ellipse_max_major_axis_cm", float("inf")),
    ("min_radius", 15.9), ("min_radius", "15"),
    ("max_radius", 50.1), ("max_radius", True),
])
def test_ring_config_rejects_non_strict_scalar_types(field, value):
    values = {"dp": 1.5, "min_distance": 50.0, "param1": 100.0,
              "param2": 0.95, "min_radius": 15, "max_radius": 50}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        RingConfig.from_mapping(values)


def test_ring_config_rejects_negative_collinearity_tolerance():
    with pytest.raises(ValueError):
        RingConfig.from_mapping({"collinearity_tolerance_px": -0.1})


@pytest.fixture
def detector():
    from d435i_vision.algorithm.ring_detector import RingDetector
    return RingDetector(RingConfig.from_mapping({"min_radius": 10, "max_radius": 60}))


@pytest.fixture
def frame():
    return np.zeros((200, 500, 3), dtype=np.uint8)


def test_returns_any_positive_number_sorted_left_to_right(monkeypatch, detector, frame):
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: np.array([[[300, 100, 20], [100, 110, 22],
                                           [200, 120, 21], [400, 130, 19]]], dtype=np.float32),
    )
    result = detector.detect(frame)
    assert [ring.center[0] for ring in result] == [100, 200, 300, 400]
    assert set(detector.intermediates) == {"gray", "gradient", "threshold"}


def test_returns_empty_tuple_when_hough_finds_nothing(monkeypatch, detector, frame):
    monkeypatch.setattr("d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
                        lambda *args, **kwargs: None)
    assert detector.detect(frame) == ()
    assert set(detector.intermediates) == {"gray", "gradient", "threshold"}


def test_smooths_matched_circle_center_and_radius(monkeypatch, detector, frame):
    hough_results = iter(
        (
            np.array([[[100, 100, 20]]], dtype=np.float32),
            np.array([[[120, 100, 40]]], dtype=np.float32),
        )
    )
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: next(hough_results),
    )

    assert detector.detect(frame) == (RingObservation((100, 100), 20),)
    assert detector.detect(frame) == (RingObservation((109, 100), 29),)


def test_keeps_circle_for_five_missing_frames_then_removes_it(
    monkeypatch, detector, frame
):
    hough_results = iter(
        (np.array([[[100, 100, 20]]], dtype=np.float32),) + (None,) * 6
    )
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: next(hough_results),
    )

    expected = (RingObservation((100, 100), 20),)
    assert detector.detect(frame) == expected
    for _ in range(5):
        assert detector.detect(frame) == expected
    assert detector.detect(frame) == ()


def test_tracking_can_be_disabled(monkeypatch, frame):
    from d435i_vision.algorithm.ring_detector import RingDetector

    detector = RingDetector(
        RingConfig.from_mapping(
            {
                "min_radius": 10,
                "max_radius": 60,
                "tracking_enabled": False,
            }
        )
    )
    hough_results = iter(
        (
            np.array([[[100, 100, 20]]], dtype=np.float32),
            np.array([[[120, 100, 40]]], dtype=np.float32),
        )
    )
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: next(hough_results),
    )

    assert detector.detect(frame) == (RingObservation((100, 100), 20),)
    assert detector.detect(frame) == (RingObservation((120, 100), 40),)


def test_reset_discards_retained_circle(monkeypatch, detector, frame):
    hough_results = iter(
        (np.array([[[100, 100, 20]]], dtype=np.float32), None)
    )
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: next(hough_results),
    )

    assert detector.detect(frame) == (RingObservation((100, 100), 20),)
    detector.reset()
    assert detector.detect(frame) == ()


def test_tracker_prefers_two_valid_matches_over_one_closest_pair():
    tracker = RingTracker(
        smoothing=1.0,
        max_match_distance_px=40.0,
        max_missing_frames=5,
    )
    tracker.update(
        (RingObservation((100, 100), 20), RingObservation((139, 100), 20))
    )

    result = tracker.update(
        (RingObservation((101, 100), 20), RingObservation((98, 100), 20))
    )

    assert [ring.center for ring in result] == [(98, 100), (101, 100)]


def test_deduplicates_concentric_candidates_and_rejects_clipped_circles(
    monkeypatch, detector, frame
):
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: np.array(
            [[[100, 100, 25], [105, 103, 22], [10, 10, 20], [250, 100, 20]]],
            dtype=np.float32,
        ),
    )
    assert [(item.center, item.radius) for item in detector.detect(frame)] == [
        ((100, 100), 25), ((250, 100), 20)
    ]
    assert set(detector.intermediates) == {"gray", "gradient", "threshold"}


@pytest.mark.parametrize("candidate_count", [1, 2, 3])
def test_accepts_one_two_three_candidates(monkeypatch, detector, frame, candidate_count):
    circles = np.array([[[100 + i * 100, 100, 20] for i in range(candidate_count)]], dtype=np.float32)
    monkeypatch.setattr("d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
                        lambda *args, **kwargs: circles)
    assert len(detector.detect(frame)) == candidate_count


def test_detects_a_real_slanted_ellipse_alongside_hough_results(
    monkeypatch, detector, frame
):
    import cv2

    cv2.ellipse(frame, (250, 100), (45, 20), 25, 0, 360, (255, 255, 255), -1)
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda *args, **kwargs: np.array([[[100, 100, 20]]], dtype=np.float32),
    )

    hough_rings = detector.detect(frame)

    assert [ring.center for ring in hough_rings] == [(100, 100)]
    assert len(detector.ellipse_observations) == 1
    ellipse = detector.ellipse_observations[0]
    assert abs(ellipse.center[0] - 250.0) < 3.0
    assert abs(ellipse.center[1] - 100.0) < 3.0
    assert ellipse.major_axis_px > ellipse.minor_axis_px
    assert 0.35 < ellipse.minor_axis_px / ellipse.major_axis_px < 0.7


def test_ellipse_stage_receives_rebinarized_gradient_mask(monkeypatch, frame):
    import cv2
    from d435i_vision.algorithm.ring_detector import RingDetector

    cv2.line(frame, (40, 100), (460, 100), (255, 255, 255), 5)
    detector = RingDetector(
        RingConfig.from_mapping(
            {
                "min_radius": 10,
                "max_radius": 60,
                "ellipse_binary_threshold": 230,
            }
        )
    )
    ellipse_inputs = []
    hough_inputs = []
    monkeypatch.setattr(
        detector,
        "_detect_ellipses",
        lambda image: ellipse_inputs.append(image.copy()) or (),
    )
    monkeypatch.setattr(
        "d435i_vision.algorithm.ring_detector.cv2.HoughCircles",
        lambda image, *args, **kwargs: hough_inputs.append(image.copy()) or None,
    )

    detector.detect(frame)

    assert ellipse_inputs and hough_inputs
    assert np.array_equal(hough_inputs[0], detector.intermediates["threshold"])
    assert np.any((hough_inputs[0] > 0) & (hough_inputs[0] < 255))
    assert set(np.unique(ellipse_inputs[0])).issubset({0, 255})
    assert np.array_equal(
        ellipse_inputs[0],
        np.where(hough_inputs[0] > 230, 255, 0).astype(np.uint8),
    )


def _ellipse_filter_detector(**overrides):
    from d435i_vision.algorithm.ring_detector import RingDetector

    values = {
        "min_radius": 10,
        "max_radius": 120,
        "ellipse_min_contour_points": 5,
        "ellipse_max_fit_error": 0.30,
        "ellipse_edge_sample_count": 72,
        "ellipse_edge_search_radius_px": 3,
        "ellipse_min_edge_support": 0.50,
        "ellipse_coverage_bin_count": 24,
        "ellipse_min_covered_bins": 14,
    }
    values.update(overrides)
    return RingDetector(RingConfig.from_mapping(values))


def test_ellipse_filter_rejects_partial_arc_with_low_perimeter_support():
    import cv2

    binary = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(binary, (160, 160), (80, 40), 20, 0, 120, 255, 3)
    detector = _ellipse_filter_detector(ellipse_min_covered_bins=1)

    assert detector._detect_ellipses(binary) == ()


def test_ellipse_filter_rejects_partial_arc_with_low_perimeter_coverage():
    import cv2

    binary = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(binary, (160, 160), (80, 40), 20, 0, 120, 255, 3)
    detector = _ellipse_filter_detector(ellipse_min_edge_support=0.20)

    assert detector._detect_ellipses(binary) == ()


def test_ellipse_filter_keeps_complete_ellipse():
    import cv2

    binary = np.zeros((320, 320), dtype=np.uint8)
    cv2.ellipse(binary, (160, 160), (80, 40), 20, 0, 360, 255, 3)
    detector = _ellipse_filter_detector()

    ellipses = detector._detect_ellipses(binary)

    assert len(ellipses) == 1
    assert abs(ellipses[0].center[0] - 160.0) < 2.0
    assert abs(ellipses[0].center[1] - 160.0) < 2.0


def test_edge_support_samples_eccentric_ellipse_by_physical_arc_length():
    """长轴端的参数角范围不应被误当成同等长度的椭圆周长。"""

    from d435i_vision.algorithm.ring_detector import RingDetector

    center = (200.0, 200.0)
    major_axis = 240.0
    minor_axis = 84.0
    dense_angles = np.linspace(0.0, 2.0 * np.pi, 20_000, endpoint=False)
    supported_angles = dense_angles[
        np.abs(np.cos(dense_angles)) >= np.sqrt(0.5)
    ]
    columns = np.rint(
        center[0] + (major_axis / 2.0) * np.cos(supported_angles)
    ).astype(int)
    rows = np.rint(
        center[1] + (minor_axis / 2.0) * np.sin(supported_angles)
    ).astype(int)
    distance_to_edge = np.full((400, 400), 100.0, dtype=np.float32)
    distance_to_edge[rows, columns] = 0.0

    support_ratio, _ = RingDetector._ellipse_edge_metrics(
        distance_to_edge,
        center,
        major_axis,
        minor_axis,
        angle_deg=0.0,
        sample_count=72,
        bin_count=24,
        search_radius_px=0,
    )

    assert 0.30 < support_ratio < 0.45

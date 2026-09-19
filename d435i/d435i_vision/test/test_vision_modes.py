from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from d435i_vision import models, visualization
from d435i_vision.models import (
    CameraFrame,
    CameraIntrinsics,
    CircleObservation,
    PipelineResult,
    RingObservation,
    SurfacePlaneObservation,
    TrackEstimate,
)
from d435i_vision.ros_control import VisionMode
from d435i_vision.vision_modes import (
    VisionModeDispatcher,
    mode_change_message,
    ring_message,
    serialize_message,
    surface_message,
)


def pipeline_result(*, with_estimate: bool = True) -> PipelineResult:
    mask = np.zeros((8, 10), dtype=np.uint8)
    return PipelineResult(
        blurred_frame=np.zeros((8, 10, 3), dtype=np.uint8),
        color_mask=mask,
        surface_mask=mask,
        depth_applied=True,
        surface_planes=[
            SurfacePlaneObservation(
                normal=(0.0, 0.6, 0.8),
                offset=-0.23456,
                center=(0.12345, -0.54321, 0.67891),
                inlier_count=3,
                candidate_count=4,
                gravity_alignment=0.8,
            )
        ],
        observations=[],
        selected=CircleObservation(
            center=(10.0, 20.0),
            radius=21.236,
            contour_area=800.0,
            circularity=0.87654,
            fill_ratio=0.76543,
        ),
        estimate=(
            TrackEstimate(
                center=(100.126, 200.234),
                velocity=(3.456, -4.567),
                measured=False,
                missing_frames=2,
            )
            if with_estimate
            else None
        ),
    )


def test_surface_message_preserves_every_existing_emit_state_field():
    message = surface_message(17, 12.3456789, "绿色", pipeline_result())

    assert message == {
        "mode": 17,
        "mode_name": "surface_tracking",
        "timestamp_monotonic_s": 12.345679,
        "color": "绿色",
        "measured": False,
        "center_px": [100.13, 200.23],
        "velocity_px_s": [3.46, -4.57],
        "speed_px_s": 5.73,
        "missing_frames": 2,
        "depth_applied": True,
        "surface_plane_count": 1,
        "radius_px": 21.24,
        "circularity": 0.877,
        "circle_fill_ratio": 0.765,
        "surface_center_camera_m": [0.1235, -0.5432, 0.6789],
        "surface_normal_camera": [0.0, 0.6, 0.8],
        "surface_inlier_ratio": 0.75,
        "surface_normal_to_gravity_deg": 36.87,
    }


def test_surface_message_returns_none_without_track_estimate():
    assert surface_message(1, 4.0, "green", pipeline_result(with_estimate=False)) is None


def test_ring_message_supports_two_ellipses_without_three_ellipse_fields():
    ellipses = (
        models.EllipseObservation((100.0, 110.0), 40.0, 20.0, 0.0, 0.02),
        models.EllipseObservation((200.0, 120.0), 42.0, 21.0, 5.0, 0.03),
    )
    message = ring_message(
        2,
        (),
        distances_from_first_cm=(0.0, 12.34),
        ellipses=ellipses,
    )

    assert message == {
        "mode": 2,
        "mode_name": "ring_detection",
        "success": True,
        "ellipse_count": 2,
        "ellipse_candidate_count": 2,
        "origin_center_px": [100.0, 110.0],
        "relative_centers_px": [[0.0, 0.0], [100.0, 10.0]],
        "distances_from_first_px": [0.0, 100.5],
        "distances_from_first_cm": [0.0, 12.34],
        "collinear": None,
        "collinearity_error_px": None,
        "collinearity_tolerance_px": 5.0,
        "ellipses": [
            {
                "center_px": [100.0, 110.0],
                "major_axis_px": 40.0,
                "minor_axis_px": 20.0,
                "angle_deg": 0.0,
                "fit_error": 0.02,
            },
            {
                "center_px": [200.0, 120.0],
                "major_axis_px": 42.0,
                "minor_axis_px": 21.0,
                "angle_deg": 5.0,
                "fit_error": 0.03,
            },
        ],
    }


def test_ring_message_adds_geometry_for_exactly_three_ellipses():
    ellipses = (
        models.EllipseObservation((100.0, 110.0), 40.0, 20.0, 0.0, 0.02),
        models.EllipseObservation((200.0, 120.0), 42.0, 21.0, 0.0, 0.02),
        models.EllipseObservation((300.0, 140.0), 44.0, 22.0, 0.0, 0.02),
    )

    message = ring_message(
        2,
        (),
        distances_from_first_cm=(0.0, 8.0, 16.0),
        ellipses=ellipses,
    )

    assert message["middle_center_px"] == [200.0, 120.0]
    assert message["left_right_y_delta_px"] == -30.0
    assert message["origin_center_px"] == [100.0, 110.0]
    assert message["relative_centers_px"] == [
        [0.0, 0.0],
        [100.0, 10.0],
        [200.0, 30.0],
    ]
    assert message["distances_from_first_px"] == [0.0, 100.5, 202.24]
    assert message["distances_from_first_cm"] == [0.0, 8.0, 16.0]
    assert message["collinear"] is True
    assert message["collinearity_error_px"] == 4.94
    assert message["collinearity_tolerance_px"] == 5.0


def test_ring_message_reports_empty_detection_as_failure():
    message = ring_message(23, ())

    assert message["success"] is False
    assert message["ellipse_count"] == 0
    assert message["ellipse_candidate_count"] == 0
    assert message["ellipses"] == []
    assert message["origin_center_px"] is None
    assert message["relative_centers_px"] == []
    assert message["distances_from_first_px"] == []
    assert message["distances_from_first_cm"] == []
    assert message["collinear"] is None


def test_mode_change_message_names_previous_and_current_modes():
    assert mode_change_message(VisionMode.IDLE, VisionMode.RING_DETECTION, 29) == {
        "event": "mode_changed",
        "previous_mode_name": "idle",
        "mode": 29,
        "mode_name": "ring_detection",
    }


def test_serialize_message_is_compact_unicode_json():
    serialized = serialize_message({"颜色": "绿色", "values": [1, 2]})

    assert serialized == '{"颜色":"绿色","values":[1,2]}'
    assert json.loads(serialized) == {"颜色": "绿色", "values": [1, 2]}


class SurfacePipelineSpy:
    def __init__(self, result: PipelineResult) -> None:
        self.result = result
        self.calls = []
        self.reset_calls = 0

    def process(self, frame, timestamp, depth_m, intrinsics, gravity_camera):
        self.calls.append((frame, timestamp, depth_m, intrinsics, gravity_camera))
        return self.result

    def reset(self):
        self.reset_calls += 1


class RingDetectorSpy:
    def __init__(
        self,
        rings=(),
        collinearity_tolerance_px=5.0,
        ellipse_observations=(),
        ellipse_require_hough_proximity=False,
    ) -> None:
        self.rings = rings
        self.calls = []
        self.reset_calls = 0
        self.ellipse_observations = ellipse_observations
        self.config = SimpleNamespace(
            collinearity_tolerance_px=collinearity_tolerance_px,
            ellipse_max_center_distance_cm=2.0,
            ellipse_require_hough_proximity=ellipse_require_hough_proximity,
            ellipse_min_major_axis_cm=5.0,
            ellipse_max_major_axis_cm=11.0,
        )
        self.intermediates = {
            "gray": np.full((8, 10), 10, dtype=np.uint8),
            "gradient": np.full((8, 10), 20, dtype=np.uint8),
            "threshold": np.full((8, 10), 30, dtype=np.uint8),
        }

    def detect(self, frame):
        self.calls.append(frame)
        return self.rings

    def reset(self):
        self.reset_calls += 1


def camera_frame() -> CameraFrame:
    return CameraFrame(
        color=np.zeros((80, 120, 3), dtype=np.uint8),
        timestamp=7.125,
        depth_m=np.full((80, 120), 0.5, dtype=np.float32),
        intrinsics=CameraIntrinsics(
            width=120,
            height=80,
            fx=100.0,
            fy=100.0,
            ppx=60.0,
            ppy=40.0,
        ),
        gravity_camera=(0.0, 1.0, 0.0),
    )


def test_dispatcher_isolates_modes_and_resets_only_after_initial_switch():
    surface = SurfacePipelineSpy(pipeline_result(with_estimate=False))
    rings = RingDetectorSpy((RingObservation((60, 40), 10),))
    dispatcher = VisionModeDispatcher(surface, rings, "green")
    frame = camera_frame()

    assert dispatcher.switch(VisionMode.IDLE) is True
    idle = dispatcher.process(VisionMode.IDLE, 0, frame, 30.0, render=False)
    assert surface.calls == []
    assert rings.calls == []
    assert idle.message is None

    assert dispatcher.switch(VisionMode.SURFACE_TRACKING) is True
    surface_result = dispatcher.process(
        VisionMode.SURFACE_TRACKING, 1, frame, 30.0, False
    )
    assert len(surface.calls) == 1
    assert rings.calls == []
    assert surface_result.message is None

    assert dispatcher.switch(VisionMode.RING_DETECTION) is True
    ring_result = dispatcher.process(VisionMode.RING_DETECTION, 2, frame, 30.0, False)
    assert len(surface.calls) == 1
    assert len(rings.calls) == 1
    assert ring_result.message["ellipse_count"] == 0
    assert ring_result.message["success"] is False
    assert surface.reset_calls == 2


def test_dispatcher_attaches_typed_surface_status() -> None:
    surface = SurfacePipelineSpy(pipeline_result(with_estimate=False))
    dispatcher = VisionModeDispatcher(surface, RingDetectorSpy(), "green")

    result = dispatcher.process(
        VisionMode.SURFACE_TRACKING,
        1,
        camera_frame(),
        30.0,
        render=False,
    )

    assert result.surface_status is not None
    assert result.surface_status.active is True
    assert result.surface_status.tracking_valid is False
    assert result.surface_status.color == "green"
    assert result.ring_status is None


def test_dispatcher_attaches_typed_ring_calibration_status() -> None:
    ellipses = tuple(
        models.EllipseObservation((x, 40.0), 20.0, 12.0, 0.0, 0.02)
        for x in (20.0, 60.0, 100.0)
    )
    detector = RingDetectorSpy(ellipse_observations=ellipses)
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result(with_estimate=False)),
        detector,
        "green",
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION,
        2,
        camera_frame(),
        30.0,
        render=False,
    )

    assert result.ring_status is not None
    assert result.ring_status.active is True
    assert result.ring_status.ring_count == 3
    assert result.ring_status.ellipse_candidate_count == 3
    assert result.surface_status is None
    assert result.message["ellipse_count"] == 3


def test_idle_dispatcher_has_no_typed_mode_result() -> None:
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()),
        RingDetectorSpy(),
        "green",
    )

    result = dispatcher.process(
        VisionMode.IDLE,
        0,
        camera_frame(),
        30.0,
        render=False,
    )

    assert result.surface_status is None
    assert result.ring_status is None
    assert result.message is None


def test_dispatcher_same_mode_switch_has_no_reset_and_returns_false():
    surface = SurfacePipelineSpy(pipeline_result())
    dispatcher = VisionModeDispatcher(surface, RingDetectorSpy(), "green")

    assert dispatcher.switch(VisionMode.SURFACE_TRACKING) is True
    assert dispatcher.switch(VisionMode.SURFACE_TRACKING) is False
    assert surface.reset_calls == 0


def test_dispatcher_resets_ring_tracking_when_leaving_ring_mode():
    surface = SurfacePipelineSpy(pipeline_result())
    rings = RingDetectorSpy()
    dispatcher = VisionModeDispatcher(surface, rings, "green")

    assert dispatcher.switch(VisionMode.RING_DETECTION) is True
    assert dispatcher.switch(VisionMode.IDLE) is True
    assert rings.reset_calls == 1


def test_surface_dispatch_uses_camera_timestamp_and_all_rgbd_fields():
    surface = SurfacePipelineSpy(pipeline_result())
    dispatcher = VisionModeDispatcher(surface, RingDetectorSpy(), "green")
    frame = camera_frame()

    result = dispatcher.process(VisionMode.SURFACE_TRACKING, 41, frame, 24.0, False)

    called_frame, timestamp, depth_m, intrinsics, gravity = surface.calls[0]
    assert called_frame is frame.color
    assert timestamp == 7.125
    assert depth_m is frame.depth_m
    assert intrinsics is frame.intrinsics
    assert gravity is frame.gravity_camera
    assert result.message["timestamp_monotonic_s"] == 7.125
    assert result.message["mode"] == 41


def test_render_false_never_builds_images():
    surface = SurfacePipelineSpy(pipeline_result())
    rings = RingDetectorSpy((RingObservation((60, 40), 10),))
    dispatcher = VisionModeDispatcher(surface, rings, "green")
    frame = camera_frame()

    for mode, wire_value in (
        (VisionMode.IDLE, 0),
        (VisionMode.SURFACE_TRACKING, 1),
        (VisionMode.RING_DETECTION, 2),
    ):
        result = dispatcher.process(mode, wire_value, frame, 30.0, False)
        assert result.main_image is None
        assert result.auxiliary_images == {}


def test_idle_render_draws_visible_idle_overlay():
    frame = camera_frame()
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), RingDetectorSpy(), "green"
    )

    result = dispatcher.process(VisionMode.IDLE, 0, frame, 30.0, True)

    assert result.message is None
    assert result.main_image is not frame.color
    assert np.any(result.main_image != frame.color)
    assert result.auxiliary_images == {}


def test_surface_render_uses_exact_existing_mask_names():
    pipeline_output = pipeline_result()
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_output), RingDetectorSpy(), "green"
    )

    result = dispatcher.process(
        VisionMode.SURFACE_TRACKING, 1, camera_frame(), 30.0, True
    )

    assert result.main_image is not None
    assert result.auxiliary_images == {
        "hsv-color-mask": pipeline_output.color_mask,
        "depth-top-surface-mask": pipeline_output.surface_mask,
    }


def test_ring_render_labels_every_returned_ellipse_and_exposes_intermediates():
    rings = (
        RingObservation((30, 40), 12),
        RingObservation((80, 40), 12),
    )
    ellipses = (
        models.EllipseObservation((30.0, 40.0), 20.0, 12.0, 0.0, 0.02),
        models.EllipseObservation((80.0, 40.0), 20.0, 12.0, 0.0, 0.02),
    )
    detector = RingDetectorSpy(rings, ellipse_observations=ellipses)
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, True)

    assert result.main_image is not None
    for ellipse in ellipses:
        x, y = (int(value) for value in ellipse.center)
        label_area = result.main_image[y + 5 : y + 24, x + 5 : x + 25]
        assert np.any(label_area != 0)
    assert result.auxiliary_images == {
        "ring-gray": detector.intermediates["gray"],
        "ring-gradient": detector.intermediates["gradient"],
        "ring-threshold": detector.intermediates["threshold"],
    }


def test_ring_dispatch_uses_detector_collinearity_tolerance():
    rings = (
        RingObservation((20, 40), 10),
        RingObservation((60, 46), 10),
        RingObservation((100, 40), 10),
    )
    ellipses = tuple(
        models.EllipseObservation(
            (float(ring.center[0]), float(ring.center[1])),
            20.0,
            12.0,
            0.0,
            0.02,
        )
        for ring in rings
    )
    detector = RingDetectorSpy(
        rings,
        collinearity_tolerance_px=7.0,
        ellipse_observations=ellipses,
    )
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["collinearity_error_px"] == 6.0
    assert result.message["collinearity_tolerance_px"] == 7.0
    assert result.message["collinear"] is True


def test_ring_render_draws_measurement_line_between_separated_ellipses():
    ellipses = (
        models.EllipseObservation((20.0, 60.0), 20.0, 12.0, 0.0, 0.02),
        models.EllipseObservation((80.0, 60.0), 20.0, 12.0, 0.0, 0.02),
    )
    detector = RingDetectorSpy((), ellipse_observations=ellipses)
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, True
    )

    # 两个椭圆中心之间的中点不属于椭圆边缘；非黑像素说明连线已画出。
    assert np.any(result.main_image[59:62, 48:53] != 0)


def test_ring_dispatch_adds_ellipse_center_distances_to_result_message():
    ellipses = (
        models.EllipseObservation((20.0, 40.0), 20.0, 12.0, 0.0, 0.02),
        models.EllipseObservation((30.0, 40.0), 20.0, 12.0, 0.0, 0.02),
    )
    detector = RingDetectorSpy((), ellipse_observations=ellipses)
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["distances_from_first_cm"] == [0.0, 5.0]


def test_ring_message_reports_ellipse_geometry_and_ellipse_success():
    ellipse = models.EllipseObservation(
        center=(123.456, 78.901),
        major_axis_px=80.123,
        minor_axis_px=42.567,
        angle_deg=27.891,
        fit_error=0.03456,
    )

    message = ring_message(2, (), ellipses=(ellipse,))

    assert message["success"] is True
    assert message["ellipse_count"] == 1
    assert message["ellipse_candidate_count"] == 1
    assert message["origin_center_px"] == [123.46, 78.9]
    assert message["ellipses"] == [
        {
            "center_px": [123.46, 78.9],
            "major_axis_px": 80.12,
            "minor_axis_px": 42.57,
            "angle_deg": 27.89,
            "fit_error": 0.035,
        }
    ]
    assert "ring_count" not in message
    assert "circles" not in message


def test_draw_rings_visibly_draws_an_ellipse_without_a_hough_circle():
    ellipse = models.EllipseObservation(
        center=(100.0, 100.0),
        major_axis_px=80.0,
        minor_axis_px=40.0,
        angle_deg=0.0,
        fit_error=0.02,
    )

    output = visualization.draw_rings(
        np.zeros((220, 220, 3), dtype=np.uint8),
        (),
        ellipses=(ellipse,),
    )

    assert np.any(output[98:103, 138:143] != 0)


def test_dispatcher_only_reports_ellipses_near_a_hough_center_in_3d():
    rings = (RingObservation((20, 40), 5),)
    near = models.EllipseObservation((21.0, 40.0), 20.0, 12.0, 0.0, 0.02)
    far = models.EllipseObservation((80.0, 40.0), 20.0, 12.0, 0.0, 0.02)
    detector = RingDetectorSpy(
        rings,
        ellipse_observations=(near, far),
        ellipse_require_hough_proximity=True,
    )
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["ellipse_candidate_count"] == 2
    assert result.message["ellipse_count"] == 1
    assert result.message["ellipses"][0]["center_px"] == [21.0, 40.0]


def test_dispatcher_keeps_ellipse_without_hough_when_proximity_filter_is_off():
    ellipse = models.EllipseObservation(
        (80.0, 40.0), 20.0, 12.0, 0.0, 0.02
    )
    detector = RingDetectorSpy((), ellipse_observations=(ellipse,))
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["success"] is True
    assert result.message["ellipse_candidate_count"] == 1
    assert result.message["ellipse_count"] == 1
    assert result.message["ellipses"][0]["center_px"] == [80.0, 40.0]


def test_dispatcher_filters_ellipses_by_physical_major_axis_length():
    accepted = models.EllipseObservation(
        (40.0, 40.0), 16.0, 8.0, 0.0, 0.02
    )
    too_long = models.EllipseObservation(
        (80.0, 40.0), 24.0, 12.0, 0.0, 0.02
    )
    detector = RingDetectorSpy(
        (),
        ellipse_observations=(accepted, too_long),
    )
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["ellipse_candidate_count"] == 2
    assert result.message["ellipse_count"] == 1
    assert result.message["ellipses"][0]["center_px"] == [40.0, 40.0]


def test_ring_message_hides_hough_and_uses_ellipse_centers_for_geometry():
    hough_rings = (
        RingObservation((20, 20), 8),
        RingObservation((50, 35), 9),
        RingObservation((80, 20), 10),
    )
    ellipses = (
        models.EllipseObservation((100.0, 100.0), 20.0, 12.0, 0.0, 0.02),
        models.EllipseObservation((140.0, 104.0), 22.0, 13.0, 5.0, 0.03),
        models.EllipseObservation((180.0, 108.0), 24.0, 14.0, 10.0, 0.04),
    )

    message = ring_message(
        2,
        hough_rings,
        collinearity_tolerance_px=5.0,
        distances_from_first_cm=(0.0, 4.2, 8.4),
        ellipses=ellipses,
    )

    assert "ring_count" not in message
    assert "circles" not in message
    assert message["success"] is True
    assert message["origin_center_px"] == [100.0, 100.0]
    assert message["relative_centers_px"] == [
        [0.0, 0.0],
        [40.0, 4.0],
        [80.0, 8.0],
    ]
    assert message["collinear"] is True
    assert message["middle_center_px"] == [140.0, 104.0]


def test_hough_detection_without_accepted_ellipse_is_not_success():
    message = ring_message(
        2,
        (RingObservation((50, 50), 20),),
        ellipses=(),
    )

    assert message["success"] is False
    assert message["ellipse_count"] == 0
    assert "ring_count" not in message
    assert "circles" not in message


def test_ring_render_draws_only_ellipses_and_not_hough_circles():
    frame = np.zeros((300, 640, 3), dtype=np.uint8)
    hough_ring = RingObservation((500, 200), 20)
    ellipse = models.EllipseObservation(
        (100.0, 200.0), 80.0, 40.0, 0.0, 0.02
    )

    output = visualization.draw_rings(
        frame,
        (hough_ring,),
        distances_from_first_cm=(0.0,),
        ellipses=(ellipse,),
    )

    assert np.all(output[178:183, 498:503] == 0)
    assert np.all(output[198:203, 498:503] == 0)
    assert np.any(output[198:203, 138:143] != 0)


def test_ring_render_lists_ellipse_centers_distances_and_axes(monkeypatch):
    ellipses = (
        models.EllipseObservation((100.0, 200.0), 20.0, 12.0, 0.0, 0.02),
        models.EllipseObservation((160.0, 200.0), 22.0, 14.0, 0.0, 0.02),
        models.EllipseObservation((220.0, 200.0), 24.0, 16.0, 0.0, 0.02),
    )
    text_calls = []
    original_put_text = visualization.cv2.putText

    def record_put_text(image, text, origin, *args, **kwargs):
        text_calls.append((text, origin))
        return original_put_text(image, text, origin, *args, **kwargs)

    monkeypatch.setattr(visualization.cv2, "putText", record_put_text)

    visualization.draw_rings(
        np.zeros((300, 640, 3), dtype=np.uint8),
        (
            RingObservation((300, 220), 10),
            RingObservation((360, 240), 11),
            RingObservation((420, 220), 12),
        ),
        distances_from_first_cm=(0.0, 6.0, 12.0),
        ellipses=ellipses,
    )

    assert ("ELLIPSE=3", (12, 24)) in text_calls
    assert ("E1 = (0, 0)", (12, 48)) in text_calls
    assert ("d1-1 = 0.0 cm", (220, 48)) in text_calls
    assert ("axes1 = 20x12 px", (430, 48)) in text_calls
    assert ("E2 = (60, 0)", (12, 72)) in text_calls
    assert ("E3 = (120, 0)", (12, 96)) in text_calls
    assert not any("HOUGH" in text or text.startswith("H") for text, _ in text_calls)


def test_dispatcher_keeps_hough_filter_but_reports_ellipse_geometry_only():
    hough_rings = (RingObservation((20, 40), 5),)
    near = models.EllipseObservation((21.0, 40.0), 20.0, 12.0, 0.0, 0.02)
    far = models.EllipseObservation((80.0, 40.0), 20.0, 12.0, 0.0, 0.02)
    detector = RingDetectorSpy(
        hough_rings,
        ellipse_observations=(near, far),
        ellipse_require_hough_proximity=True,
    )
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["ellipse_candidate_count"] == 2
    assert result.message["ellipse_count"] == 1
    assert result.message["origin_center_px"] == [21.0, 40.0]
    assert "ring_count" not in result.message
    assert "circles" not in result.message


def test_dispatcher_measures_distance_between_ellipse_centers_not_hough_centers():
    detector = RingDetectorSpy(
        (RingObservation((20, 40), 5), RingObservation((100, 40), 5)),
        ellipse_observations=(
            models.EllipseObservation((40.0, 40.0), 20.0, 12.0, 0.0, 0.02),
            models.EllipseObservation((60.0, 40.0), 20.0, 12.0, 0.0, 0.02),
        ),
        ellipse_require_hough_proximity=False,
    )
    dispatcher = VisionModeDispatcher(
        SurfacePipelineSpy(pipeline_result()), detector, "green"
    )

    result = dispatcher.process(
        VisionMode.RING_DETECTION, 2, camera_frame(), 30.0, False
    )

    assert result.message["origin_center_px"] == [40.0, 40.0]
    assert result.message["distances_from_first_cm"] == [0.0, 10.0]

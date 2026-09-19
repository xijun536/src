from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from d435i_vision.ros_status import (
    LatestStatusCache,
    RosMessageTypes,
    RosStatusBridge,
    vision_status_to_ros,
)
from d435i_vision.vision_status import (
    RingCalibrationStatus,
    RingCenterStatus,
    SurfaceTrackingStatus,
    VisionStatusSnapshot,
)


def fake_message_types() -> RosMessageTypes:
    return RosMessageTypes(
        vision_status=lambda: SimpleNamespace(header=SimpleNamespace()),
        surface_result=SimpleNamespace,
        ring_result=SimpleNamespace,
        ring_center=SimpleNamespace,
        point=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        vector3=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
    )


def ready_snapshot() -> VisionStatusSnapshot:
    centers = tuple(
        RingCenterStatus(
            order=index,
            center_px=(u, 240.0),
            position_camera_m=(x, 0.0, 0.30),
            major_axis_px=60.0,
            minor_axis_px=45.0,
            ellipse_angle_deg=0.0,
            fit_error=0.02,
        )
        for index, (u, x) in enumerate(
            ((220.0, -0.05), (320.0, 0.0), (420.0, 0.05)),
            start=1,
        )
    )
    ring = RingCalibrationStatus(
        active=True,
        ellipse_candidate_count=3,
        rings=centers,
        collinear=True,
        collinearity_error_px=0.0,
        middle_error_px=(0.0, 0.0),
        middle_center_camera_m=(0.0, 0.0, 0.30),
        middle_error_camera_m=(0.0, 0.0, 0.0),
        target_distance_m=0.30,
        distance_error_m=0.0,
        line_direction_camera=(1.0, 0.0, 0.0),
        line_to_x_axis_angle_deg=0.0,
        center_aligned=True,
        distance_aligned=True,
        angle_aligned=True,
        calibration_ready=True,
        failure_reason="ready",
    )
    return VisionStatusSnapshot(
        frame_sequence=7,
        current_mode_value=2,
        current_mode_name="ring_detection",
        selected_color="green",
        camera_ok=True,
        result_available=True,
        processing_fps=29.5,
        surface=SurfaceTrackingStatus.inactive("green"),
        ring=ring,
    )


class FakeMonotonic:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class FakeNode:
    def __init__(self) -> None:
        self.publisher = FakePublisher()
        self.service_callback = None
        self.publisher_args = None
        self.service_args = None
        self.destroyed_publishers = []
        self.destroyed_services = []
        self.destroy_node_calls = 0

    def create_publisher(self, message_type, topic, qos):
        self.publisher_args = (message_type, topic, qos)
        return self.publisher

    def create_service(self, service_type, name, callback):
        self.service_args = (service_type, name)
        self.service_callback = callback
        return SimpleNamespace(name=name)

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "ros-stamp")
        )

    def destroy_publisher(self, publisher) -> None:
        self.destroyed_publishers.append(publisher)

    def destroy_service(self, service) -> None:
        self.destroyed_services.append(service)

    def destroy_node(self) -> None:
        self.destroy_node_calls += 1


def response_message():
    return SimpleNamespace(cache_available=False, result_age_s=0.0, status=None)


def started_bridge(monotonic=None) -> tuple[RosStatusBridge, FakeNode]:
    node = FakeNode()
    bridge = RosStatusBridge(
        message_types=fake_message_types(),
        service_type=SimpleNamespace,
        status_topic="/vision/status",
        service_name="/vision/get_status",
        frame_id="camera_color_optical_frame",
        status_qos="best-effort-qos",
        target_distance_m=0.30,
        monotonic=monotonic or FakeMonotonic(1.0, 1.1, 1.2),
    )
    bridge.start(node)
    return bridge, node


def test_converter_maps_complete_ring_measurement() -> None:
    message = vision_status_to_ros(
        ready_snapshot(),
        fake_message_types(),
        stamp="stamp",
        frame_id="camera_frame",
    )

    assert message.header.stamp == "stamp"
    assert message.header.frame_id == "camera_frame"
    assert message.frame_sequence == 7
    assert message.current_mode == 2
    assert message.current_mode_name == "ring_detection"
    assert message.surface.active is False
    assert message.ring.ring_count == 3
    assert message.ring.valid_3d_count == 3
    assert message.ring.rings[1].position_3d_valid is True
    assert message.ring.rings[1].position_camera_m.z == pytest.approx(0.30)
    assert message.ring.middle_center_error_px == pytest.approx(0.0)
    assert message.ring.line_direction_camera.x == pytest.approx(1.0)
    assert message.ring.calibration_ready is True


def test_converter_uses_validity_flags_and_zero_for_missing_3d_values() -> None:
    snapshot = ready_snapshot()
    first = replace(snapshot.ring.rings[0], position_camera_m=None)
    invalid_ring = replace(
        snapshot.ring,
        rings=(first, *snapshot.ring.rings[1:]),
        calibration_ready=False,
        failure_reason="invalid_ring_depth",
    )

    message = vision_status_to_ros(
        replace(snapshot, ring=invalid_ring),
        fake_message_types(),
        stamp="stamp",
        frame_id="camera_frame",
    )

    assert message.ring.all_depth_valid is False
    assert message.ring.rings[0].position_3d_valid is False
    assert message.ring.rings[0].position_camera_m.x == 0.0
    assert message.ring.rings[0].position_camera_m.y == 0.0
    assert message.ring.rings[0].position_camera_m.z == 0.0


def test_latest_cache_returns_a_copy_and_monotonic_age() -> None:
    cache = LatestStatusCache()
    original = SimpleNamespace(value=[1])
    cache.update(original, updated_monotonic_s=10.0)

    first, age = cache.read(now_monotonic_s=10.25)
    first.value.append(2)
    second, _ = cache.read(now_monotonic_s=9.0)

    assert age == pytest.approx(0.25)
    assert second.value == [1]
    assert cache.read(now_monotonic_s=9.0)[1] == 0.0


def test_empty_cache_uses_negative_age() -> None:
    cache = LatestStatusCache()

    assert cache.read(10.0) == (None, -1.0)


def test_bridge_publishes_and_service_returns_same_cached_status() -> None:
    bridge, node = started_bridge(FakeMonotonic(10.0, 10.25))
    bridge.set_control_state(2, "ring_detection", "green")
    bridge.publish(ready_snapshot())

    response = bridge.handle_get_status(object(), response_message())

    assert len(node.publisher.messages) == 1
    assert response.cache_available is True
    assert response.result_age_s == pytest.approx(0.25)
    assert response.status.frame_sequence == 7
    assert response.status.ring.calibration_ready is True


def test_service_before_first_frame_reports_control_state_without_measurement() -> None:
    bridge, _ = started_bridge()
    bridge.set_control_state(0, "idle", "green")

    response = bridge.handle_get_status(object(), response_message())

    assert response.cache_available is False
    assert response.result_age_s == -1.0
    assert response.status.current_mode == 0
    assert response.status.current_mode_name == "idle"
    assert response.status.selected_color == "green"
    assert response.status.camera_ok is False
    assert response.status.result_available is False
    assert response.status.surface.active is False
    assert response.status.ring.active is False


def test_control_state_change_invalidates_old_frame_result() -> None:
    bridge, _ = started_bridge(FakeMonotonic(1.0, 2.0))
    bridge.set_control_state(2, "ring_detection", "green")
    bridge.publish(ready_snapshot())

    bridge.set_control_state(0, "idle", "green")
    response = bridge.handle_get_status(object(), response_message())

    assert response.cache_available is False
    assert response.status.current_mode_name == "idle"
    assert response.status.ring.active is False


def test_old_frame_cannot_repopulate_cache_after_control_state_change() -> None:
    bridge, _ = started_bridge(FakeMonotonic(1.0, 2.0))
    bridge.set_control_state(2, "ring_detection", "green")
    old_ring_frame = ready_snapshot()

    bridge.set_control_state(0, "idle", "green")
    bridge.publish(old_ring_frame)
    response = bridge.handle_get_status(object(), response_message())

    assert response.cache_available is False
    assert response.status.current_mode_name == "idle"
    assert response.status.ring.active is False


def test_old_color_frame_cannot_repopulate_cache_after_color_change() -> None:
    bridge, _ = started_bridge(FakeMonotonic(1.0, 2.0))
    bridge.set_control_state(2, "ring_detection", "green")

    bridge.set_control_state(2, "ring_detection", "yellow")
    bridge.publish(ready_snapshot())
    response = bridge.handle_get_status(object(), response_message())

    assert response.cache_available is False
    assert response.status.selected_color == "yellow"


def test_bridge_closes_its_entities_without_destroying_shared_node() -> None:
    bridge, node = started_bridge()

    bridge.close()

    assert node.destroyed_publishers == [node.publisher]
    assert len(node.destroyed_services) == 1
    assert node.destroy_node_calls == 0

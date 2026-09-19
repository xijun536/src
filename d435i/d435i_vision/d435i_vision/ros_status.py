"""Convert vision results to ROS messages and expose the latest status."""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional, Tuple

from .models import Point3D
from .vision_status import (
    RingCalibrationStatus,
    SurfaceTrackingStatus,
    VisionStatusSnapshot,
)


@dataclass(frozen=True)
class RosMessageTypes:
    """ROS message classes needed by the converter, injected for testability."""

    vision_status: Any
    surface_result: Any
    ring_result: Any
    ring_center: Any
    point: Any
    vector3: Any


def _optional_number(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def _point(message_types: RosMessageTypes, value: Optional[Point3D]) -> Any:
    message = message_types.point()
    if value is not None:
        message.x, message.y, message.z = map(float, value)
    return message


def _vector(message_types: RosMessageTypes, value: Optional[Point3D]) -> Any:
    message = message_types.vector3()
    if value is not None:
        message.x, message.y, message.z = map(float, value)
    return message


def vision_status_to_ros(
    snapshot: VisionStatusSnapshot,
    message_types: RosMessageTypes,
    *,
    stamp: Any,
    frame_id: str,
) -> Any:
    """Convert a ROS-independent snapshot to generated ROS message objects."""

    message = message_types.vision_status()
    message.header.stamp = stamp
    message.header.frame_id = str(frame_id)
    message.frame_sequence = int(snapshot.frame_sequence)
    message.current_mode = int(snapshot.current_mode_value)
    message.current_mode_name = snapshot.current_mode_name
    message.selected_color = snapshot.selected_color
    message.camera_ok = bool(snapshot.camera_ok)
    message.result_available = bool(snapshot.result_available)
    message.processing_fps = float(snapshot.processing_fps)

    surface = snapshot.surface
    surface_message = message_types.surface_result()
    surface_message.active = bool(surface.active)
    surface_message.tracking_valid = bool(surface.tracking_valid)
    surface_message.measured = bool(surface.measured)
    surface_message.color = surface.color
    surface_message.center_u_px = (
        0.0 if surface.center_px is None else float(surface.center_px[0])
    )
    surface_message.center_v_px = (
        0.0 if surface.center_px is None else float(surface.center_px[1])
    )
    surface_message.velocity_u_px_s = (
        0.0 if surface.velocity_px_s is None else float(surface.velocity_px_s[0])
    )
    surface_message.velocity_v_px_s = (
        0.0 if surface.velocity_px_s is None else float(surface.velocity_px_s[1])
    )
    surface_message.speed_px_s = float(surface.speed_px_s)
    surface_message.radius_px = float(surface.radius_px)
    surface_message.missing_frames = int(surface.missing_frames)
    surface_message.depth_applied = bool(surface.depth_applied)
    surface_message.surface_3d_valid = surface.surface_center_camera_m is not None
    surface_message.surface_center_camera_m = _point(
        message_types, surface.surface_center_camera_m
    )
    surface_message.surface_normal_camera = _vector(
        message_types, surface.surface_normal_camera
    )
    surface_message.surface_inlier_ratio = float(surface.surface_inlier_ratio)
    surface_message.normal_to_gravity_deg = _optional_number(
        surface.normal_to_gravity_deg
    )
    message.surface = surface_message

    ring = snapshot.ring
    ring_message = message_types.ring_result()
    ring_message.active = bool(ring.active)
    ring_message.ellipse_candidate_count = int(ring.ellipse_candidate_count)
    ring_message.ring_count = int(ring.ring_count)
    ring_message.valid_3d_count = int(ring.valid_3d_count)
    ring_messages = []
    for center in ring.rings:
        center_message = message_types.ring_center()
        center_message.order = int(center.order)
        center_message.center_u_px = float(center.center_px[0])
        center_message.center_v_px = float(center.center_px[1])
        center_message.position_3d_valid = center.position_camera_m is not None
        center_message.position_camera_m = _point(
            message_types, center.position_camera_m
        )
        center_message.major_axis_px = float(center.major_axis_px)
        center_message.minor_axis_px = float(center.minor_axis_px)
        center_message.ellipse_angle_deg = float(center.ellipse_angle_deg)
        center_message.fit_error = float(center.fit_error)
        ring_messages.append(center_message)
    ring_message.rings = ring_messages
    ring_message.exactly_three = bool(ring.exactly_three)
    ring_message.all_depth_valid = bool(ring.all_depth_valid)
    ring_message.collinear = bool(ring.collinear)
    ring_message.collinearity_error_px = _optional_number(
        ring.collinearity_error_px
    )
    ring_message.middle_error_u_px = (
        0.0 if ring.middle_error_px is None else float(ring.middle_error_px[0])
    )
    ring_message.middle_error_v_px = (
        0.0 if ring.middle_error_px is None else float(ring.middle_error_px[1])
    )
    ring_message.middle_center_error_px = _optional_number(
        ring.middle_center_error_px
    )
    ring_message.middle_center_camera_m = _point(
        message_types, ring.middle_center_camera_m
    )
    ring_message.middle_error_camera_m = _vector(
        message_types, ring.middle_error_camera_m
    )
    ring_message.target_distance_m = float(ring.target_distance_m)
    ring_message.distance_error_m = _optional_number(ring.distance_error_m)
    ring_message.line_direction_camera = _vector(
        message_types, ring.line_direction_camera
    )
    ring_message.line_to_x_axis_angle_deg = _optional_number(
        ring.line_to_x_axis_angle_deg
    )
    ring_message.center_aligned = bool(ring.center_aligned)
    ring_message.distance_aligned = bool(ring.distance_aligned)
    ring_message.angle_aligned = bool(ring.angle_aligned)
    ring_message.calibration_ready = bool(ring.calibration_ready)
    ring_message.failure_reason = ring.failure_reason
    message.ring = ring_message
    return message


class LatestStatusCache:
    """Thread-safe snapshot cache with monotonic result age."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._message: Optional[Any] = None
        self._updated_monotonic_s = 0.0

    def update(self, message: Any, updated_monotonic_s: float) -> None:
        with self._lock:
            self._message = deepcopy(message)
            self._updated_monotonic_s = float(updated_monotonic_s)

    def clear(self) -> None:
        with self._lock:
            self._message = None
            self._updated_monotonic_s = 0.0

    def read(self, now_monotonic_s: float) -> Tuple[Optional[Any], float]:
        with self._lock:
            if self._message is None:
                return None, -1.0
            age_s = max(0.0, float(now_monotonic_s) - self._updated_monotonic_s)
            return deepcopy(self._message), age_s


class RosStatusBridge:
    """Own the status publisher/service while sharing a mode-control node."""

    def __init__(
        self,
        *,
        message_types: RosMessageTypes,
        service_type: Any,
        status_topic: str,
        service_name: str,
        frame_id: str,
        status_qos: Any,
        target_distance_m: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._message_types = message_types
        self._service_type = service_type
        self._status_topic = status_topic
        self._service_name = service_name
        self._frame_id = frame_id
        self._status_qos = status_qos
        self._target_distance_m = float(target_distance_m)
        self._monotonic = monotonic
        self._cache = LatestStatusCache()
        # This lock makes control changes, cache invalidation, and accepted
        # frame publication one ordered operation across executor threads.
        self._control_lock = threading.RLock()
        self._mode_value = 0
        self._mode_name = "idle"
        self._color = ""
        self._node: Optional[Any] = None
        self._publisher: Optional[Any] = None
        self._service: Optional[Any] = None

    def start(self, node: Any) -> None:
        if self._node is not None:
            return
        self._node = node
        try:
            self._publisher = node.create_publisher(
                self._message_types.vision_status,
                self._status_topic,
                self._status_qos,
            )
            self._service = node.create_service(
                self._service_type,
                self._service_name,
                self.handle_get_status,
            )
        except Exception:
            self.close()
            raise

    def set_control_state(
        self,
        mode_value: int,
        mode_name: str,
        color: str,
    ) -> None:
        new_state = (int(mode_value), str(mode_name), str(color))
        with self._control_lock:
            old_state = (self._mode_value, self._mode_name, self._color)
            self._mode_value, self._mode_name, self._color = new_state
            if new_state != old_state:
                self._cache.clear()

    def publish(self, snapshot: VisionStatusSnapshot) -> None:
        if self._node is None or self._publisher is None:
            raise RuntimeError("ROS status bridge has not been started")
        stamp = self._node.get_clock().now().to_msg()
        message = vision_status_to_ros(
            snapshot,
            self._message_types,
            stamp=stamp,
            frame_id=self._frame_id,
        )
        snapshot_state = (
            int(snapshot.current_mode_value),
            str(snapshot.current_mode_name),
            str(snapshot.selected_color),
        )
        with self._control_lock:
            current_state = (self._mode_value, self._mode_name, self._color)
            if snapshot_state != current_state:
                return
            self._cache.update(message, self._monotonic())
            self._publisher.publish(message)

    def handle_get_status(self, _request: Any, response: Any) -> Any:
        now_monotonic_s = self._monotonic()
        with self._control_lock:
            message, age_s = self._cache.read(now_monotonic_s)
            response.cache_available = message is not None
            response.result_age_s = age_s
            if message is None:
                if self._node is None:
                    raise RuntimeError("ROS status bridge has not been started")
                message = vision_status_to_ros(
                    self._control_snapshot(),
                    self._message_types,
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=self._frame_id,
                )
        response.status = message
        return response

    def _control_snapshot(self) -> VisionStatusSnapshot:
        with self._control_lock:
            mode_value = self._mode_value
            mode_name = self._mode_name
            color = self._color
        return VisionStatusSnapshot(
            frame_sequence=0,
            current_mode_value=mode_value,
            current_mode_name=mode_name,
            selected_color=color,
            camera_ok=False,
            result_available=False,
            processing_fps=0.0,
            surface=SurfaceTrackingStatus.inactive(color),
            ring=RingCalibrationStatus.inactive(self._target_distance_m),
        )

    def close(self) -> None:
        node = self._node
        service = self._service
        publisher = self._publisher
        self._service = None
        self._publisher = None
        self._node = None
        self._cache.clear()
        if node is not None and service is not None:
            with suppress(Exception):
                node.destroy_service(service)
        if node is not None and publisher is not None:
            with suppress(Exception):
                node.destroy_publisher(publisher)

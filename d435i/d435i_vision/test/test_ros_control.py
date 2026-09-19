from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from d435i_vision.algorithm.ring_detector import RingConfig
from d435i_vision.ros_control import (
    RosModeControl,
    RosSettings,
    StaticModeControl,
    VisionMode,
    build_mode_control,
)


class FakeUInt8:
    def __init__(self, data=0):
        self.data = data


class FakeString:
    def __init__(self, data=""):
        self.data = data


class FakeSubscription:
    def __init__(self, message_type, topic, callback, qos_depth):
        self.message_type = message_type
        self.topic = topic
        self.callback = callback
        self.qos_depth = qos_depth


class FakePublisher:
    def __init__(self, message_type, topic, qos_depth):
        self.message_type = message_type
        self.topic = topic
        self.qos_depth = qos_depth
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.subscription = None
        self.publisher = None
        self.publishers = []
        self.service = None
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, qos_depth):
        self.subscription = FakeSubscription(message_type, topic, callback, qos_depth)
        return self.subscription

    def create_publisher(self, message_type, topic, qos_depth):
        self.publisher = FakePublisher(message_type, topic, qos_depth)
        self.publishers.append(self.publisher)
        return self.publisher

    def create_service(self, service_type, name, callback):
        self.service = SimpleNamespace(
            service_type=service_type, name=name, callback=callback
        )
        return self.service

    def destroy_publisher(self, publisher):
        publisher.destroyed = True

    def destroy_service(self, service):
        service.destroyed = True

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: "ros-stamp")
        )

    def destroy_node(self):
        self.destroyed = True


class FakeRclpy:
    def __init__(self, initially_ok=False):
        self.ok_value = initially_ok
        self.init_calls = 0
        self.shutdown_calls = 0
        self.nodes = []
        self.spin_calls = []

    def ok(self):
        return self.ok_value

    def init(self):
        self.init_calls += 1
        self.ok_value = True

    def create_node(self, name):
        node = FakeNode(name)
        self.nodes.append(node)
        return node

    def spin_once(self, node, timeout_sec):
        self.spin_calls.append((node, timeout_sec))

    def shutdown(self):
        if not self.ok_value:
            raise RuntimeError("ROS context is already shut down")
        self.shutdown_calls += 1
        self.ok_value = False


def install_fake_ros(monkeypatch, fake_rclpy):
    std_msgs = types.ModuleType("std_msgs")
    messages = types.ModuleType("std_msgs.msg")
    messages.UInt8 = FakeUInt8
    messages.String = FakeString
    std_msgs.msg = messages
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", messages)

    class EmptyMessage:
        pass

    cv3_interfaces = types.ModuleType("cv3_interfaces")
    cv3_messages = types.ModuleType("cv3_interfaces.msg")
    for name in (
        "RingCalibrationResult",
        "RingCenter",
        "SurfaceTrackingResult",
        "VisionStatus",
    ):
        setattr(cv3_messages, name, type(name, (EmptyMessage,), {}))
    cv3_services = types.ModuleType("cv3_interfaces.srv")
    cv3_services.GetVisionStatus = type("GetVisionStatus", (), {})
    cv3_interfaces.msg = cv3_messages
    cv3_interfaces.srv = cv3_services
    monkeypatch.setitem(sys.modules, "cv3_interfaces", cv3_interfaces)
    monkeypatch.setitem(sys.modules, "cv3_interfaces.msg", cv3_messages)
    monkeypatch.setitem(sys.modules, "cv3_interfaces.srv", cv3_services)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_messages = types.ModuleType("geometry_msgs.msg")
    geometry_messages.Point = type("Point", (), {})
    geometry_messages.Vector3 = type("Vector3", (), {})
    geometry_msgs.msg = geometry_messages
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_messages)

    qos = types.ModuleType("rclpy.qos")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep-last")
    qos.ReliabilityPolicy = SimpleNamespace(BEST_EFFORT="best-effort")
    qos.QoSProfile = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)


def test_ros_settings_use_documented_defaults():
    settings = RosSettings.from_mapping({"unrelated": {"enabled": False}})

    assert settings.enabled is True
    assert settings.node_name == "cv3_vision_controller"
    assert settings.mode_topic == "/vision_mode"
    assert settings.qos_depth == 10
    assert settings.resolve(settings.initial_mode) is VisionMode.IDLE
    assert settings.resolve(0) is VisionMode.IDLE
    assert settings.resolve(1) is VisionMode.SURFACE_TRACKING
    assert settings.resolve(2) is VisionMode.RING_DETECTION
    assert settings.resolve(99) is None
    assert settings.result.enabled is False
    assert settings.result.topic == "/vision/result"
    assert settings.result.qos_depth == 10
    assert settings.camera_frame_id == "camera_color_optical_frame"
    assert settings.status.topic == "/vision/status"
    assert settings.status.qos_depth == 1
    assert settings.service.name == "/vision/get_status"


def test_ros_settings_parse_status_service_and_frame_id():
    settings = RosSettings.from_mapping({"ros": {
        "camera_frame_id": "tool_camera_optical_frame",
        "status": {"topic": "/custom/status", "qos_depth": 2},
        "service": {"name": "/custom/get_status"},
    }})

    assert settings.camera_frame_id == "tool_camera_optical_frame"
    assert settings.status.topic == "/custom/status"
    assert settings.status.qos_depth == 2
    assert settings.service.name == "/custom/get_status"


@pytest.mark.parametrize(
    "ros",
    [
        {"camera_frame_id": ""},
        {"status": {"topic": ""}},
        {"status": {"qos_depth": 0}},
        {"status": {"qos_depth": True}},
        {"service": {"name": ""}},
        {"status": []},
        {"service": []},
    ],
)
def test_invalid_status_configuration_is_rejected(ros):
    with pytest.raises(ValueError):
        RosSettings.from_mapping({"ros": ros})


def test_ros_settings_parse_custom_values():
    settings = RosSettings.from_mapping({"ros": {
        "enabled": True, "node_name": "picker", "mode_topic": "/picker/mode",
        "qos_depth": 3, "initial_mode": 7,
        "modes": {"idle": 7, "surface_tracking": 8, "ring_detection": 9},
        "result": {"enabled": False, "topic": "/picker/result", "qos_depth": 4},
    }})

    assert settings.resolve(8) is VisionMode.SURFACE_TRACKING
    assert settings.value_for(VisionMode.RING_DETECTION) == 9
    assert settings.resolve(settings.initial_mode) is VisionMode.IDLE


@pytest.mark.parametrize(
    "ros_values",
    [
        {"modes": {"idle": 0, "surface_tracking": 0, "ring_detection": 2}},
        {"modes": {"idle": -1, "surface_tracking": 1, "ring_detection": 2}},
        {"modes": {"idle": 0, "surface_tracking": 1, "ring_detection": 256}},
        {"initial_mode": 99},
        {"node_name": ""},
        {"mode_topic": ""},
        {"qos_depth": 0},
        {"result": {"topic": ""}},
        {"result": {"qos_depth": 0}},
    ],
)
def test_ros_settings_reject_invalid_values(ros_values):
    with pytest.raises(ValueError):
        RosSettings.from_mapping({"ros": ros_values})


@pytest.mark.parametrize(
    "document",
    [
        "ros:\n  initial_mode: true\n",
        "ros:\n  initial_mode: 1.0\n",
        'ros:\n  enabled: "false"\n',
        'ros:\n  result:\n    enabled: "false"\n',
    ],
)
def test_ros_settings_reject_non_exact_yaml_scalar_types(document):
    with pytest.raises(ValueError):
        RosSettings.from_mapping(yaml.safe_load(document))


def test_ros_mode_control_lifecycle_updates_mode_and_publishes(monkeypatch, capsys):
    fake_rclpy = FakeRclpy()
    install_fake_ros(monkeypatch, fake_rclpy)
    settings = RosSettings.from_mapping({"ros": {"result": {"enabled": True}}})

    control = build_mode_control(settings)
    assert isinstance(control, RosModeControl)
    control.start()
    fake_node = fake_rclpy.nodes[-1]
    fake_publisher = next(
        publisher
        for publisher in fake_node.publishers
        if publisher.topic == "/vision/result"
    )

    assert fake_node.name == "cv3_vision_controller"
    assert fake_node.subscription.topic == "/vision_mode"
    assert fake_node.subscription.qos_depth == 10
    assert any(item.topic == "/vision/status" for item in fake_node.publishers)
    assert fake_node.service.name == "/vision/get_status"
    assert control.current_mode is VisionMode.IDLE
    fake_node.subscription.callback(FakeUInt8(data=2))
    assert control.current_mode is VisionMode.RING_DETECTION

    fake_node.subscription.callback(FakeUInt8(data=99))
    assert control.current_mode is VisionMode.RING_DETECTION
    assert "99" in capsys.readouterr().err

    assert control.spin_once() is True
    assert fake_rclpy.spin_calls == [(fake_node, 0.0)]
    control.publish_result('{"ok":true}')
    assert fake_publisher.topic == "/vision/result"
    assert fake_publisher.qos_depth == 10
    assert fake_publisher.messages[-1].data == '{"ok":true}'

    control.close()
    assert fake_node.destroyed is True
    assert fake_rclpy.shutdown_calls == 1


def test_disabled_result_publishing_is_a_no_op(monkeypatch):
    fake_rclpy = FakeRclpy()
    install_fake_ros(monkeypatch, fake_rclpy)
    control = build_mode_control(RosSettings.from_mapping({}))

    control.start()
    fake_node = fake_rclpy.nodes[-1]
    assert all(item.topic != "/vision/result" for item in fake_node.publishers)
    control.publish_result('{"ignored":true}')
    assert all(item.topic != "/vision/result" for item in fake_node.publishers)
    control.close()


def test_static_mode_control_imports_no_ros_and_stays_at_initial_mode(monkeypatch):
    settings = RosSettings.from_mapping({"ros": {
        "enabled": False,
        "initial_mode": 8,
        "modes": {"idle": 7, "surface_tracking": 8, "ring_detection": 9},
    }})
    original_import = builtins.__import__

    def reject_ros_import(name, *args, **kwargs):
        if (
            name == "rclpy"
            or name.startswith("std_msgs")
            or name.startswith("cv3_interfaces")
            or name.startswith("geometry_msgs")
        ):
            raise AssertionError(f"unexpected ROS import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ros_import)
    control = build_mode_control(settings)

    assert isinstance(control, StaticModeControl)
    control.start()
    assert control.current_mode is VisionMode.SURFACE_TRACKING
    assert control.wire_value(VisionMode.RING_DETECTION) == 9
    assert control.spin_once() is True
    control.publish_result('{"ignored":true}')
    control.publish_status(object())
    control.close()


def test_ros_control_updates_status_bridge_only_for_accepted_control_changes():
    class BridgeSpy:
        def __init__(self):
            self.states = []
            self.started_node = None
            self.closed = False

        def start(self, node):
            self.started_node = node

        def set_control_state(self, mode_value, mode_name, color):
            self.states.append((mode_value, mode_name, color))

        def publish(self, snapshot):
            self.snapshot = snapshot

        def close(self):
            self.closed = True

    bridge = BridgeSpy()
    fake_rclpy = FakeRclpy()
    settings = RosSettings.from_mapping({})
    control = RosModeControl(
        settings,
        fake_rclpy,
        FakeUInt8,
        FakeString,
        initial_color="green",
        available_colors=("green", "yellow"),
        status_bridge=bridge,
    )

    control.start()
    node = fake_rclpy.nodes[-1]
    assert bridge.started_node is node
    assert bridge.states == [(0, "idle", "green")]

    control._on_mode_message(FakeUInt8(2))
    control._on_color_message(FakeString("yellow"))
    control._on_mode_message(FakeUInt8(99))
    control._on_color_message(FakeString("purple"))
    assert bridge.states == [
        (0, "idle", "green"),
        (2, "ring_detection", "green"),
        (2, "ring_detection", "yellow"),
    ]

    marker = object()
    control.publish_status(marker)
    assert bridge.snapshot is marker
    control.close()
    assert bridge.closed is True
    assert node.destroyed is True


def test_ros_control_does_not_shutdown_a_preexisting_context(monkeypatch):
    fake_rclpy = FakeRclpy(initially_ok=True)
    install_fake_ros(monkeypatch, fake_rclpy)
    control = build_mode_control(RosSettings.from_mapping({}))

    control.start()
    control.close()

    assert fake_rclpy.init_calls == 0
    assert fake_rclpy.nodes[-1].destroyed is True
    assert fake_rclpy.shutdown_calls == 0


def test_ros_control_close_is_idempotent_after_owned_context_stops(monkeypatch):
    fake_rclpy = FakeRclpy()
    install_fake_ros(monkeypatch, fake_rclpy)
    control = build_mode_control(RosSettings.from_mapping({}))

    control.start()
    fake_node = fake_rclpy.nodes[-1]
    fake_rclpy.ok_value = False
    control.close()
    control.close()

    assert fake_node.destroyed is True
    assert fake_rclpy.shutdown_calls == 0


def test_repository_yaml_contains_valid_ros_and_ring_configuration():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    root = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    ros = RosSettings.from_mapping(root)
    ring = RingConfig.from_mapping(root["ring"])

    assert ros.resolve(2) is VisionMode.RING_DETECTION
    assert ros.result.topic == "/vision/result"
    assert ros.camera_frame_id == "camera_color_optical_frame"
    assert ros.status.topic == "/vision/status"
    assert ros.status.qos_depth == 1
    assert ros.service.name == "/vision/get_status"
    assert ring.dp == 1.5
    assert ring.max_radius == 100
    assert ring.ellipse_require_hough_proximity is True
    assert ring.ellipse_binary_threshold == 127
    assert ring.ellipse_edge_sample_count == 72
    assert ring.ellipse_edge_search_radius_px == 3
    assert ring.ellipse_min_edge_support == 0.50
    assert ring.ellipse_coverage_bin_count == 24
    assert ring.ellipse_min_covered_bins == 14
    assert ring.ellipse_min_major_axis_cm == 5.0
    assert ring.ellipse_max_major_axis_cm == 11.0
    assert ring.tracking_enabled is True
    assert ring.tracking_smoothing == 0.45
    assert ring.tracking_max_match_distance_px == 40.0
    assert ring.tracking_max_missing_frames == 5

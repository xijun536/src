"""ROS-independent mode configuration and optional ROS 2 lifecycle control."""

from __future__ import annotations

import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Collection, Mapping, Optional, Protocol

from .ros_status import RosMessageTypes, RosStatusBridge
from .vision_status import VisionStatusSnapshot


class VisionMode(str, Enum):
    """Semantic vision modes independent of their configured wire values."""

    IDLE = "idle"
    SURFACE_TRACKING = "surface_tracking"
    RING_DETECTION = "ring_detection"


@dataclass(frozen=True)
class ModeValues:
    idle: int = 0
    surface_tracking: int = 1
    ring_detection: int = 2

    def value_for(self, mode: VisionMode) -> int:  #从语义获取数值
        return {
            VisionMode.IDLE: self.idle,
            VisionMode.SURFACE_TRACKING: self.surface_tracking,
            VisionMode.RING_DETECTION: self.ring_detection,
        }[mode]  #先创建了一个字典，然后用[]取值

    def resolve(self, value: int) -> Optional[VisionMode]:  #从数值获取语义
        for mode in VisionMode:
            if self.value_for(mode) == value:
                return mode
        return None


@dataclass(frozen=True)
class ResultPublisherSettings:
    enabled: bool = False
    topic: str = "/vision/result"
    qos_depth: int = 10


@dataclass(frozen=True)
class StatusPublisherSettings:
    topic: str = "/vision/status"
    qos_depth: int = 1


@dataclass(frozen=True)
class StatusServiceSettings:
    name: str = "/vision/get_status"


@dataclass(frozen=True)
class RosSettings:
    enabled: bool = True
    node_name: str = "cv3_vision_controller"
    mode_topic: str = "/vision_mode"
    color_topic: str = "/vision_color"
    camera_frame_id: str = "camera_color_optical_frame"
    qos_depth: int = 10
    initial_mode: int = 0
    modes: ModeValues = ModeValues()
    result: ResultPublisherSettings = ResultPublisherSettings()
    status: StatusPublisherSettings = StatusPublisherSettings()
    service: StatusServiceSettings = StatusServiceSettings()

    @classmethod
    def from_mapping(cls, root: Mapping[str, Any]) -> "RosSettings":  #cls代表当前类，常常出现在@classmethod之后
        ros = root.get("ros", {})   #.get()方法用于获取指定键的值，或者返回默认值，这里默认值设置为空字典，root应该为一个yami文件爱你解析之后的字典（一种典型的Mapping）
        if not isinstance(ros, Mapping):
            raise ValueError("ros configuration must be a mapping")

        mode_data = ros.get("modes", {})
        result_data = ros.get("result", {})
        status_data = ros.get("status", {})
        service_data = ros.get("service", {})
        if not isinstance(mode_data, Mapping):
            raise ValueError("ros.modes must be a mapping")
        if not isinstance(result_data, Mapping):
            raise ValueError("ros.result must be a mapping")
        if not isinstance(status_data, Mapping):
            raise ValueError("ros.status must be a mapping")
        if not isinstance(service_data, Mapping):
            raise ValueError("ros.service must be a mapping")

        modes = ModeValues(
            idle=mode_data.get("idle", 0),
            surface_tracking=mode_data.get("surface_tracking", 1),
            ring_detection=mode_data.get("ring_detection", 2),
        )
        result = ResultPublisherSettings(
            enabled=result_data.get("enabled", False),
            topic=result_data.get("topic", "/vision/result"),
            qos_depth=result_data.get("qos_depth", 10),
        )
        status = StatusPublisherSettings(
            topic=status_data.get("topic", "/vision/status"),
            qos_depth=status_data.get("qos_depth", 1),
        )
        service = StatusServiceSettings(
            name=service_data.get("name", "/vision/get_status"),
        )
        settings = cls(
            enabled=ros.get("enabled", True),
            node_name=ros.get("node_name", "cv3_vision_controller"),
            mode_topic=ros.get("mode_topic", "/vision_mode"),
            color_topic=ros.get("color_topic", "/vision_color"),
            camera_frame_id=ros.get(
                "camera_frame_id", "camera_color_optical_frame"
            ),
            qos_depth=ros.get("qos_depth", 10),
            initial_mode=ros.get("initial_mode", 0),
            modes=modes,
            result=result,
            status=status,
            service=service,
        )
        settings._validate()
        return settings  #从配置文件解析出来RosSettings对象

    def _validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("ros.enabled must be a boolean")
        if type(self.result.enabled) is not bool:
            raise ValueError("ros.result.enabled must be a boolean")
        if type(self.initial_mode) is not int:
            raise ValueError("ros.initial_mode must be an integer")
        values = tuple(self.value_for(mode) for mode in VisionMode)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("ros mode values must be integers")
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("ros mode values must be in the UInt8 range 0..255")
        if len(set(values)) != len(values):
            raise ValueError("ros mode values must be unique")
        if self.resolve(self.initial_mode) is None:
            raise ValueError("ros.initial_mode must match a configured mode value")
        if not isinstance(self.node_name, str) or not self.node_name.strip():
            raise ValueError("ros.node_name must be a non-empty string")
        if not isinstance(self.mode_topic, str) or not self.mode_topic.strip():
            raise ValueError("ros.mode_topic must be a non-empty string")
        if not isinstance(self.color_topic, str) or not self.color_topic.strip():
            raise ValueError("ros.color_topic must be a non-empty string")
        if (
            not isinstance(self.camera_frame_id, str)
            or not self.camera_frame_id.strip()
        ):
            raise ValueError("ros.camera_frame_id must be a non-empty string")
        if not isinstance(self.result.topic, str) or not self.result.topic.strip():
            raise ValueError("ros.result.topic must be a non-empty string")
        if not isinstance(self.status.topic, str) or not self.status.topic.strip():
            raise ValueError("ros.status.topic must be a non-empty string")
        if not isinstance(self.service.name, str) or not self.service.name.strip():
            raise ValueError("ros.service.name must be a non-empty string")
        if (not isinstance(self.qos_depth, int) or isinstance(self.qos_depth, bool)
                or self.qos_depth < 1):
            raise ValueError("ros.qos_depth must be at least 1")
        if (not isinstance(self.result.qos_depth, int)
                or isinstance(self.result.qos_depth, bool)
                or self.result.qos_depth < 1):
            raise ValueError("ros.result.qos_depth must be at least 1")
        if (not isinstance(self.status.qos_depth, int)
                or isinstance(self.status.qos_depth, bool)
                or self.status.qos_depth < 1):
            raise ValueError("ros.status.qos_depth must be at least 1")

    def resolve(self, value: int) -> Optional[VisionMode]:
        return self.modes.resolve(value)

    def value_for(self, mode: VisionMode) -> int:
        return self.modes.value_for(mode)


class ModeControl(Protocol):  #构建了一个接口协议，与普通的显性父类继承不同，这里的（Protocol）代表所有具有如下属性和方法的类都是ModeControl的子类
    """狭义上的ModeControl接口协议，具备这些属性和方法的类都是这个类的子类"""

    def start(self) -> None:   #开启模式控制
        ...

    def spin_once(self) -> bool:  #处理一次控制事件，并且返回是否应该继续
        ...

    @property
    def current_mode(self) -> VisionMode:  #取值当前的模式
        ...

    @property
    def current_color(self) -> str:
        ...

    def wire_value(self, mode: VisionMode) -> int:  #返回当前模式对应的数值（0-2）
        ...

    def publish_result(self, text: str) -> None:  #发送JSON字符串
        ...

    def publish_status(self, snapshot: VisionStatusSnapshot) -> None:
        ...

    def close(self) -> None:  #关闭控制
        ...


class StaticModeControl:
    """Fixed mode control for offline operation with ROS disabled."""

    def __init__(
        self,
        settings: RosSettings,
        initial_color: str = "green",
        available_colors: Collection[str] = ("green",),
    ) -> None:
        self._settings = settings
        mode = settings.resolve(settings.initial_mode)  #RosSetting的initial_mode是一个int值，现在要获得语义值
        if mode is None:  # Protected by RosSettings validation.
            raise ValueError("ros.initial_mode must match a configured mode value")
        self._current_mode = mode
        self._available_colors = _validated_colors(initial_color, available_colors)
        self._current_color = initial_color

    def start(self) -> None:
        pass

    def spin_once(self) -> bool:
        return True

    @property
    def current_mode(self) -> VisionMode:  #离线控制依托yaml配置，直接返回
        return self._current_mode

    @property
    def current_color(self) -> str:
        return self._current_color

    def wire_value(self, mode: VisionMode) -> int:
        return self._settings.value_for(mode)

    def publish_result(self, text: str) -> None:   #离线控制不发送信息
        pass

    def publish_status(self, snapshot: VisionStatusSnapshot) -> None:
        pass

    def close(self) -> None:
        pass


class RosModeControl:
    """Live ROS 2 subscription and optional result publisher."""

    def __init__(
        self,
        settings: RosSettings,
        rclpy: Any,
        uint8_type: Any,
        string_type: Any,
        initial_color: str = "green",
        available_colors: Collection[str] = ("green",),
        status_bridge: Optional[RosStatusBridge] = None,
    ) -> None:
        self._settings = settings
        self._rclpy = rclpy
        self._uint8_type = uint8_type
        self._string_type = string_type
        mode = settings.resolve(settings.initial_mode)
        if mode is None:  # Protected by RosSettings validation.
            raise ValueError("ros.initial_mode must match a configured mode value")
        self._current_mode = mode
        self._available_colors = _validated_colors(initial_color, available_colors)
        self._current_color = initial_color
        self._lock = threading.Lock()  #线程上锁
        self._node: Any = None
        self._subscription: Any = None
        self._color_subscription: Any = None
        self._publisher: Any = None
        self._status_bridge = status_bridge
        self._owns_context = False

    def start(self) -> None:
        if self._node is not None:
            return
        try:
            if not self._rclpy.ok():
                self._rclpy.init()
                self._owns_context = True
            self._node = self._rclpy.create_node(self._settings.node_name)  #创建节点
            self._color_subscription = self._node.create_subscription(
                self._string_type,
                self._settings.color_topic,
                self._on_color_message,
                self._settings.qos_depth,
            )
            self._subscription = self._node.create_subscription(  #订阅消息
                self._uint8_type,
                self._settings.mode_topic,
                self._on_mode_message,
                self._settings.qos_depth,
            )
            if self._settings.result.enabled:
                self._publisher = self._node.create_publisher(  #发布消息
                    self._string_type,
                    self._settings.result.topic,
                    self._settings.result.qos_depth,
                )
            if self._status_bridge is not None:
                self._status_bridge.start(self._node)
                self._update_status_control_state()
        except Exception as exc:
            self._cleanup_after_failed_start()
            raise RuntimeError("failed to start ROS mode control") from exc

    def _cleanup_after_failed_start(self) -> None:   #在启动失败之后清理
        if self._status_bridge is not None:
            with suppress(Exception):
                self._status_bridge.close()
        if self._node is not None:
            with suppress(Exception):
                self._node.destroy_node()
        self._node = None
        self._subscription = None
        self._color_subscription = None
        self._publisher = None
        if self._owns_context:
            with suppress(Exception):
                self._rclpy.shutdown()
        self._owns_context = False

    def _on_mode_message(self, message: Any) -> None:
        value = message.data
        with self._lock:
            mode = self._settings.resolve(value)
            if mode is None:
                print(
                    f"warning: ignoring unmapped ROS vision mode value {value!r}",
                    file=sys.stderr,
                )
                return
            self._current_mode = mode
        self._update_status_control_state()

    def _on_color_message(self, message: Any) -> None:
        value = message.data
        with self._lock:
            if value not in self._available_colors:
                print(
                    f"warning: ignoring unconfigured ROS surface color {value!r}",
                    file=sys.stderr,
                )
                return
            self._current_color = value
        self._update_status_control_state()

    def _update_status_control_state(self) -> None:
        if self._status_bridge is None:
            return
        with self._lock:
            mode = self._current_mode
            color = self._current_color
        self._status_bridge.set_control_state(
            self._settings.value_for(mode),
            mode.value,
            color,
        )

    def spin_once(self) -> bool:   #尝试运行一次控制，返回是否可以继续运行
        if self._node is None:
            raise RuntimeError("ROS mode control has not been started")
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            return self._rclpy.ok()
        except Exception as exc:
            raise RuntimeError("failed to process ROS mode messages") from exc

    @property
    def current_mode(self) -> VisionMode:
        with self._lock:  #在查询模式的时候，将线程上锁，防止在读取的时候变化模式
            return self._current_mode

    @property
    def current_color(self) -> str:
        with self._lock:
            return self._current_color

    def wire_value(self, mode: VisionMode) -> int:
        return self._settings.value_for(mode)

    def publish_result(self, text: str) -> None:
        if not self._settings.result.enabled:
            return
        if self._publisher is None:  #没有创建pub
            raise RuntimeError("ROS result publisher has not been started")
        try:
            message = self._string_type()
            message.data = text
            self._publisher.publish(message)
        except Exception as exc:
            raise RuntimeError("failed to publish ROS vision result") from exc

    def publish_status(self, snapshot: VisionStatusSnapshot) -> None:
        if self._status_bridge is None:
            return
        try:
            self._status_bridge.publish(snapshot)
        except Exception as exc:
            raise RuntimeError("failed to publish ROS vision status") from exc

    def close(self) -> None:
        error: Optional[Exception] = None
        if self._status_bridge is not None:
            try:
                self._status_bridge.close()
            except Exception as exc:
                error = exc
        if self._node is not None:
            try:
                self._node.destroy_node()  #关闭节点
            except Exception as exc:
                error = exc
        self._node = None
        self._subscription = None
        self._color_subscription = None
        self._publisher = None
        if self._owns_context:
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception as exc:
                if error is None:
                    error = exc
        self._owns_context = False
        if error is not None:
            raise RuntimeError("failed to close ROS mode control") from error


def _validated_colors(
    initial_color: str,
    available_colors: Collection[str],
) -> frozenset[str]:
    if not isinstance(initial_color, str) or not initial_color:
        raise ValueError("initial surface color must be a non-empty string")
    colors = frozenset(available_colors)
    if any(not isinstance(color, str) or not color for color in colors):
        raise ValueError("available surface colors must be non-empty strings")
    if initial_color not in colors:
        raise ValueError("initial surface color must be configured")
    return colors


def build_mode_control(
    settings: RosSettings,
    initial_color: str = "green",
    available_colors: Collection[str] = ("green",),
    target_distance_m: float = 0.30,
) -> ModeControl:
    """Build static or live control, importing ROS only for live operation."""

    if not settings.enabled:
        return StaticModeControl(settings, initial_color, available_colors)
    try:
        import rclpy
        from cv3_interfaces.msg import (
            RingCalibrationResult,
            RingCenter,
            SurfaceTrackingResult,
            VisionStatus,
        )
        from cv3_interfaces.srv import GetVisionStatus
        from geometry_msgs.msg import Point, Vector3
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String, UInt8
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 control requires rclpy, ROS messages, and the generated "
            "cv3_interfaces package; build ros2/cv3_interfaces and source its "
            "install/setup.bash"
        ) from exc
    status_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=settings.status.qos_depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    status_bridge = RosStatusBridge(
        message_types=RosMessageTypes(
            vision_status=VisionStatus,
            surface_result=SurfaceTrackingResult,
            ring_result=RingCalibrationResult,
            ring_center=RingCenter,
            point=Point,
            vector3=Vector3,
        ),
        service_type=GetVisionStatus,
        status_topic=settings.status.topic,
        service_name=settings.service.name,
        frame_id=settings.camera_frame_id,
        status_qos=status_qos,
        target_distance_m=target_distance_m,
    )
    return RosModeControl(
        settings,
        rclpy,
        UInt8,
        String,
        initial_color,
        available_colors,
        status_bridge,
    )

#!/usr/bin/env python3
"""D435i深度圆形上表面检测与二维跟踪入口。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .algorithm.pipeline import CircleSurfaceTrackingPipeline, PipelineConfig
from .algorithm.ring_detector import RingConfig, RingDetector
from .models import CameraFrame, CameraIntrinsics, ColorRange
from .resource_paths import resolve_default_config_path
from .ros_control import RosSettings, VisionMode, build_mode_control
from .vision_modes import (
    VisionModeDispatcher,
    color_change_message,
    mode_change_message,
    serialize_message,
)
from .vision_status import RingCalibrationConfig, build_vision_snapshot

# YAML默认路径优先来自已安装的ament share，源码运行时回退到新包根目录。
DEFAULT_CONFIG_PATH = resolve_default_config_path()

#yaml配置
def load_yaml(path: Path) -> Mapping[str, Any]:
    '''将路径中的yaml文件中的文本配置读取为键值映射的可操作对象'''
    if not path.is_file():  #检查输入参数是不是一个文件
        raise ValueError(f"找不到YAML配置文件：{path}")
    try:
        import yaml  #引入PyYAML
    except ImportError as exc:
        raise RuntimeError("读取YAML需要安装PyYAML") from exc

    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError("YAML顶层必须是键值映射")
    return loaded


def section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})   #获得制定键的值（这个值本身就是一个键值类型），如果没有返回默认值，这里是空字典
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML中的{name}必须是键值映射")
    return value


def color_ranges(config: Mapping[str, Any], color: str) -> List[ColorRange]:
    colors = section(config, "colors")  #获得颜色配置
    color_config = colors.get(color)   #获得某个颜色的配置
    if not isinstance(color_config, Mapping):
        raise ValueError(f"YAML中没有颜色配置：{color}")
    raw_ranges = color_config.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"颜色{color}至少需要一组HSV范围")

    parsed: List[ColorRange] = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, Mapping):
            raise ValueError(f"颜色{color}的HSV范围格式错误")
        lower = np.asarray(raw_range.get("lower"), dtype=np.uint8)
        upper = np.asarray(raw_range.get("upper"), dtype=np.uint8)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError(f"颜色{color}的HSV上下界都必须包含三个值")
        parsed.append((lower, upper))
    return parsed

#D435i相机
class RealSenseRGBDSource:
    """读取D435i彩色、深度和加速度，并统一到彩色相机坐标系。"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        imu_enabled: bool = True,
        accel_fps: int = 63,  #加速度采样频率
        gravity_smoothing: float = 0.85,  #重力加速度方向的平滑系数
    ) -> None:
        if accel_fps <= 0:
            raise ValueError("imu.accel_fps必须大于0")
        if not 0.0 <= gravity_smoothing < 1.0:
            raise ValueError("imu.gravity_smoothing必须位于[0, 1)内")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.imu_enabled = bool(imu_enabled)
        self.accel_fps = int(accel_fps)
        self.gravity_smoothing = float(gravity_smoothing)
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._motion_sensor: Any = None
        self._imu_queue: Any = None
        self._depth_scale = 0.001
        self._intrinsics: Optional[CameraIntrinsics] = None
        self._accel_to_color_rotation: Optional[np.ndarray] = None
        self._gravity_camera: Optional[np.ndarray] = None
        self._frame_count = 0
        self._device_description = "unknown"

    def open(self) -> None:
        try:
            import pyrealsense2 as rs  #引入依赖库
        except ImportError as exc:
            raise RuntimeError(
                "没有安装pyrealsense2；请先激活yolov11_cu13环境"
            ) from exc
        self._rs = rs

        context = rs.context()  #创建一个可以管理realsense的管理环境
        devices = context.query_devices()  #查询现有的realsense设备，返回一个列表
        if len(devices) == 0:
            raise RuntimeError("没有检测到RealSense设备")
        device = devices[0]   #默认使用第一个设备
        serial = device.get_info(rs.camera_info.serial_number)  #获得相机的serial_number，括号里是rs包的参数
        device_name = device.get_info(rs.camera_info.name)  #获得相机的设备名
        firmware = device.get_info(rs.camera_info.firmware_version)  #获得设备的固件版本
        usb_type = (
            device.get_info(rs.camera_info.usb_type_descriptor)
            if device.supports(rs.camera_info.usb_type_descriptor)
            else "unknown"
        )
        self._device_description = (
            f"name={device_name}, serial={serial}, firmware={firmware}, USB={usb_type}"
        )
        print(f"RealSense设备：{self._device_description}", file=sys.stderr)
        if str(usb_type).startswith("2"):
            print(
                "警告：当前是USB 2.x，640x480彩色+深度@30可能带宽不足；"
                "请换USB 3.x端口和数据线。",
                file=sys.stderr,
            )
        
        pipeline = rs.pipeline(context)  #创建一个数据处理管道，
        config = rs.config()
        config.enable_device(serial)  #使能设备
        config.enable_stream(  #使能深度流
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(  #使能颜色流
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        # IMU与视频使用不同的交付通道，避免wait_for_frames等待63 Hz运动帧。
        if self.imu_enabled:
            self._start_imu_sensor(rs, device)

        try:#进行深度单位转换
            profile = pipeline.start(config)  #开启pipeline，用于记录相机运行配置
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())  #获得深度单位对应实际长度单位（比如：米）
            depth_profile = profile.get_stream(rs.stream.depth)  #深度流的配置
            color_profile = profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()  #颜色流配置，将通用的流配置转换为视频流配置，转换之后可以读取分辨率和相机内参
            intrinsics = color_profile.get_intrinsics()
            self._intrinsics = CameraIntrinsics(
                width=int(intrinsics.width),
                height=int(intrinsics.height),
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                ppx=float(intrinsics.ppx),
                ppy=float(intrinsics.ppy),
            )
            if self.imu_enabled:
                # D435i SDK输出的加速度已经位于深度相机坐标系。
                extrinsics = depth_profile.get_extrinsics_to(color_profile)   #获取深度坐标系到颜色坐标系的外参
                self._accel_to_color_rotation = np.asarray(    #获得颜色方向的旋转矩阵，用于解出方向向量
                    extrinsics.rotation,
                    dtype=np.float64,
                ).reshape((3, 3), order="F")  #构造成3*3的矩阵，采用以列优先的方式
                self._drain_imu_queue()
            self._align = rs.align(rs.stream.color)#深度图配准彩色图
            self._pipeline = pipeline
        except Exception:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
            self._stop_imu_sensor()
            raise

    def read(self) -> CameraFrame:
        if (
            self._rs is None
            or self._pipeline is None
            or self._align is None
            or self._intrinsics is None
        ):
            raise RuntimeError("D435i彩色深度流尚未启动")

        try:
            frames = self._pipeline.wait_for_frames()  #读取画面
        except RuntimeError as exc:
            if "Frame didn't arrive" not in str(exc):
                raise
            imu_state = "on" if self.imu_enabled else "off"
            raise RuntimeError(
                "RGBD帧超时："
                f"此前已收到{self._frame_count}帧，IMU={imu_state}，"
                f"{self._device_description}"
            ) from exc
        self._frame_count += 1
        if self.imu_enabled:
            self._drain_imu_queue()

        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("D435i当前帧缺少彩色图或深度图")

        color = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data())  #没有复制因为之后很快就要进行单位转换
        depth_m = depth_raw.astype(np.float32) * self._depth_scale  #进行单位转换之后就有独立的内存
        return CameraFrame(
            color=color,
            timestamp=time.monotonic(),
            depth_m=depth_m,
            intrinsics=self._intrinsics,
            gravity_camera=(
                tuple(float(value) for value in self._gravity_camera)
                if self._gravity_camera is not None
                else None
            ),
        )

    def _start_imu_sensor(self, rs: Any, device: Any) -> None:
        """让加速度计独立写入队列，不参与RGBD帧同步。"""
        motion_sensor = device.first_motion_sensor()
        candidates = [   #获取IMU的配置
            profile
            for profile in motion_sensor.get_stream_profiles()
            if profile.stream_type() == rs.stream.accel
            and profile.format() == rs.format.motion_xyz32f
        ]
        selected = next(  #.next()方法用于查找第一个符合条件的选项
            (profile for profile in candidates if profile.fps() == self.accel_fps),
            None,
        )
        if selected is None:
            available = sorted({profile.fps() for profile in candidates})
            raise RuntimeError(
                f"D435i不支持{self.accel_fps}Hz加速度流；可用频率：{available}"
            )

        queue = rs.frame_queue(8, keep_frames=False)
        motion_sensor.open(selected)
        try:
            motion_sensor.start(queue)
        except Exception:
            motion_sensor.close()
            raise
        self._motion_sensor = motion_sensor
        self._imu_queue = queue

    def _drain_imu_queue(self) -> None:
        """取走队列中的全部样本，用最新重力方向更新低通滤波器。"""
        if self._imu_queue is None:
            return
        while True:
            frame = self._imu_queue.poll_for_frame()
            if not frame:
                break
            motion_frame = frame.as_motion_frame()
            if not motion_frame:
                continue
            acceleration = motion_frame.get_motion_data()
            self._update_gravity(
                np.asarray(
                    (acceleration.x, acceleration.y, acceleration.z),
                    dtype=np.float64,
                )
            )

    def _update_gravity(self, acceleration_imu: np.ndarray) -> None:
        """把加速度方向变换到彩色相机坐标系并进行低通滤波。"""
        if self._accel_to_color_rotation is None:
            return
        gravity = self._accel_to_color_rotation @ acceleration_imu  # @代表矩阵乘法
        norm = float(np.linalg.norm(gravity))    #计算向量的欧几里德长度
        if not np.all(np.isfinite(gravity)) or norm < 1e-6:
            return
        gravity /= norm  #加速度向量归一化
        if self._gravity_camera is None:   #self._gravity_camera保存的是经过平滑过的重力方向
            self._gravity_camera = gravity
            return

        alpha = self.gravity_smoothing  #取得平滑系数
        filtered = alpha * self._gravity_camera + (1.0 - alpha) * gravity
        filtered_norm = float(np.linalg.norm(filtered))  #计算向量的欧几里德长度
        if filtered_norm >= 1e-6:  #当新旧加速度相差过大的时候，产生的filtered会很小，此时不保存结果，也就是说静止情况下的重力加速的是比较准的
            self._gravity_camera = filtered / filtered_norm

    def close(self) -> None:
        try:
            if self._pipeline is not None:
                self._pipeline.stop()
        finally:
            self._pipeline = None
            self._stop_imu_sensor()
            self._align = None
            self._rs = None
            self._accel_to_color_rotation = None
            self._gravity_camera = None
            self._frame_count = 0

    def _stop_imu_sensor(self) -> None:
        motion_sensor = self._motion_sensor
        self._motion_sensor = None
        self._imu_queue = None
        if motion_sensor is None:
            return
        try:
            motion_sensor.stop()
        finally:
            motion_sensor.close()

#普通摄像头输入
class OpenCVFrameSource:
    """普通摄像头或视频输入；仅供将depth_surface.required改为false后调试。"""

    def __init__(self, source: Any, width: int, height: int, fps: int) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频输入：{self.source}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = capture

    def read(self) -> CameraFrame:
        if self._capture is None:
            raise RuntimeError("OpenCV视频输入尚未启动")
        ok, frame = self._capture.read()
        if not ok:
            raise EOFError("视频结束或摄像头读取失败")
        return CameraFrame(color=frame, timestamp=time.monotonic())

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

#使用命令行参数
def build_argument_parser() -> argparse.ArgumentParser:
    '''告诉命令行，哪些参数是合法的，怎么样解析和检查参数，以及参数的默认值'''
    parser = argparse.ArgumentParser(description="D435i深度圆形上表面检测与跟踪")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)   #将输入的路径设定为config文件
    parser.add_argument(
        "--color",
        choices=("red", "green", "blue", "yellow", "black", "light_blue"),
    )
    parser.add_argument("--source", choices=("realsense", "camera", "video"))
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--video", type=Path)
    parser.add_argument(
        "--no-imu",
        action="store_true",   #参数出现时为true，没出现时为false
        help="仅启动彩色和深度，用于排查IMU与RGBD资源冲突",
    )
    parser.add_argument("--no-display", action="store_true", default=None)
    return parser


def parse_application_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse application options after removing launch-injected ROS options."""

    application_args = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in application_args:
        from rclpy.utilities import remove_ros_args

        application_args = remove_ros_args(application_args)
    return build_argument_parser().parse_args(application_args)

#将yaml配置分别组装
def build_pipeline(
    config: Mapping[str, Any],
    color: str,
    use_gravity_override: Optional[bool] = None,
) -> CircleSurfaceTrackingPipeline:
    preprocess = section(config, "preprocess")
    depth = section(config, "depth_surface")
    circle = section(config, "circle")
    tracker = section(config, "tracker")

    pipeline_config = PipelineConfig(
        gaussian_size=int(preprocess.get("gaussian_size", 5)),
        morphology_size=int(preprocess.get("morphology_size", 3)),
        open_iterations=int(preprocess.get("open_iterations", 1)),
        close_iterations=int(preprocess.get("close_iterations", 1)),
        depth_enabled=bool(depth.get("enabled", True)),
        depth_required=bool(depth.get("required", True)),
        min_depth_m=float(depth.get("min_depth_m", 0.15)),
        max_depth_m=float(depth.get("max_depth_m", 2.0)),
        plane_distance_m=float(depth.get("plane_distance_m", 0.005)),
        ransac_iterations=int(depth.get("ransac_iterations", 180)),
        min_plane_points=int(depth.get("min_points", 120)),
        max_sample_points=int(depth.get("max_sample_points", 4000)),
        min_plane_inlier_ratio=float(depth.get("min_inlier_ratio", 0.25)),
        use_gravity=(
            bool(depth.get("use_gravity", True))
            if use_gravity_override is None
            else bool(use_gravity_override)
        ),
        gravity_required=(
            bool(depth.get("gravity_required", True))
            if use_gravity_override is None
            else bool(use_gravity_override)
        ),
        max_normal_gravity_angle_deg=float(
            depth.get("max_normal_gravity_angle_deg", 30.0)
        ),
        surface_close_iterations=int(depth.get("close_iterations", 0)),
        min_area=float(circle.get("min_area", 150.0)),
        min_contour_points=int(circle.get("min_contour_points", 20)),
        min_radius_px=float(circle.get("min_radius_px", 10.0)),
        max_radius_px=float(circle.get("max_radius_px", 300.0)),
        min_circularity=float(circle.get("min_circularity", 0.65)),
        min_fill_ratio=float(circle.get("min_fill_ratio", 0.65)),
        alpha=float(tracker.get("alpha", 0.75)),
        beta=float(tracker.get("beta", 0.15)),
        max_missing=int(tracker.get("max_missing", 5)),
        association_distance=float(tracker.get("association_distance", 160.0)),
    )
    return CircleSurfaceTrackingPipeline(
        color_ranges=color_ranges(config, color),
        config=pipeline_config,
    )

def emit_message(message: Mapping[str, Any], mode_control: Any) -> None:
    """Print and optionally publish one identical compact JSON payload."""

    text = serialize_message(message)
    print(text, flush=True)
    mode_control.publish_result(text)  #调用ModeControl的方法

#附属窗口定义
_AUXILIARY_WINDOWS = {
    VisionMode.IDLE: (),
    VisionMode.SURFACE_TRACKING: (
        "hsv-color-mask",
        "depth-top-surface-mask",
    ),
    VisionMode.RING_DETECTION: (
        "ring-gray",
        "ring-gradient",
        "ring-threshold",
    ),
}


def close_inactive_windows(
    previous: VisionMode,
    current: VisionMode,
    cv_module: Any = cv2,
) -> None:
    """Close only auxiliary windows that are not used by the new mode."""

    current_names = set(_AUXILIARY_WINDOWS[current])
    for name in _AUXILIARY_WINDOWS[previous]:
        if name not in current_names:
            cv_module.destroyWindow(name)


def run_mode_loop(
    source: Any,  #数据源
    mode_control: Any,  #控制模式：默认固定控制、ros通信控制
    dispatcher: Any,  #核心任务调度器
    print_hz: float,  #打印频率
    no_display: bool,  #是否显示
    cv_module: Any = cv2,  #opencv的模块，默认是opencv2
    target_distance_m: float = 0.30,
) -> int:
    """Own one shared camera and dispatch its frames according to ROS mode state."""

    print_interval = 1.0 / print_hz if print_hz > 0.0 else float("inf")   #如果频率小于等于零，打印间隔的间隔等于无穷大
    last_print_time = float("-inf")
    previous_frame_time: Optional[float] = None
    display_fps = 0.0
    displayed_mode: Optional[VisionMode] = None
    previous_color: Optional[str] = None
    control_started = False
    source_opened = False
    frame_sequence = 0

    try:
        try:
            mode_control.start()   #开启控制，如果是静态控制跳过，ros控制则调用rosModeControl的start方法
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("failed to start ROS mode control") from exc
        control_started = True   #开启模式控制

        previous_mode = mode_control.current_mode
        previous_color = mode_control.current_color
        dispatcher.switch(previous_mode)  #调用核心调度器的方法，切换到当前状态
        source.open()
        source_opened = True   #打开图像源

        while True:
            try:
                keep_running = mode_control.spin_once()
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError("failed to poll ROS mode control") from exc
            if not keep_running:
                break

            mode = mode_control.current_mode
            current_color = mode_control.current_color
            wire_value = mode_control.wire_value(mode)
            if current_color != previous_color:
                if dispatcher.switch_color(current_color):
                    emit_message(
                        color_change_message(previous_color, current_color),
                        mode_control,
                    )
                    last_print_time = float("-inf")
                previous_color = current_color
            if dispatcher.switch(mode):   #切换模式
                if not no_display and displayed_mode is not None:
                    close_inactive_windows(displayed_mode, mode, cv_module)  #关闭原来的窗口
                emit_message(
                    mode_change_message(previous_mode, mode, wire_value),
                    mode_control,
                )
                last_print_time = float("-inf")
            previous_mode = mode

            try:
                camera_frame = source.read()  #读取图像源
            except EOFError:
                break

            timestamp = camera_frame.timestamp
            if previous_frame_time is not None:
                dt = max(timestamp - previous_frame_time, 1e-6)
                current_fps = 1.0 / dt
                display_fps = (
                    current_fps
                    if display_fps == 0.0
                    else 0.90 * display_fps + 0.10 * current_fps
                )
            previous_frame_time = timestamp

            mode_result = dispatcher.process(  #图像处理的入口
                mode,
                wire_value,
                camera_frame,
                display_fps,
                render=not no_display,
            )
            frame_sequence += 1
            status = build_vision_snapshot(
                frame_sequence=frame_sequence,
                mode_value=wire_value,
                mode_name=mode.value,
                selected_color=current_color,
                processing_fps=display_fps,
                surface_status=mode_result.surface_status,
                ring_status=mode_result.ring_status,
                target_distance_m=target_distance_m,
            )
            mode_control.publish_status(status)
            if (
                print_hz > 0.0
                and mode_result.message is not None
                and timestamp - last_print_time >= print_interval
            ):
                emit_message(mode_result.message, mode_control)  #大于打印间隔的就打印时被结果的msg
                last_print_time = timestamp  #记录上一次打印时间

            if not no_display:
                if mode_result.main_image is None:
                    raise RuntimeError("display mode did not produce a main image")
                cv_module.imshow("vision-main", mode_result.main_image)  #显示主窗口
                for name, image in mode_result.auxiliary_images.items():   #显示附属窗口
                    cv_module.imshow(name, image)
                displayed_mode = mode
                key = cv_module.waitKey(1) & 0xFF    #每1ms读取键盘输入
                if key in (ord("q"), 27):    #当输入为q时退出
                    break
    finally:
        try:
            if source_opened:   #关闭图像源
                source.close()
        finally:
            try:
                if not no_display:
                    cv_module.destroyAllWindows()   #关闭所有窗口
            finally:
                if control_started:
                    mode_control.close()

    return 0


#主循环入口 组装配置、相机、模式控制和两个视觉流水线
def run(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)  #从yaml文件中读取配置，返回值是一个键值映射对象
    camera = section(config, "camera")   #将对应键的值给变量，这些变量在被赋值之后也是字典类型（一种典型的Mapping的子类）
    imu = section(config, "imu")
    target = section(config, "target")
    output = section(config, "output")

    source_name = str(args.source or camera.get("source", "realsense"))  #.get()方法用于获得某个建的值，否则返回默认值
    color = str(args.color or target.get("color", "green"))
    width = int(camera.get("width", 640))
    height = int(camera.get("height", 480))
    fps = int(camera.get("fps", 30))
    camera_index = int(
        args.camera_index if args.camera_index is not None else camera.get("index", 0)  #优先选取命令行参数传入的值，如果没有的话，就是用config的默认值
    )
    surface_pipeline = build_pipeline(
        config,
        color,
        use_gravity_override=False if args.no_imu else None,
    )
    ring_config = RingConfig.from_mapping(section(config, "ring"))  #色环检测配置
    ring_calibration = RingCalibrationConfig.from_mapping(
        section(config, "ring_calibration")
    )
    ros_settings = RosSettings.from_mapping(config)   #ros配置
    configured_colors = {
        str(name): color_ranges(config, str(name))
        for name in section(config, "colors")
    }
    ring_detector = RingDetector(ring_config)   #色环检测器
    dispatcher = VisionModeDispatcher(
        surface_pipeline,
        ring_detector,
        color,
        configured_colors,
        ring_calibration,
    )    #核心任务调度器
    mode_control = build_mode_control(
        ros_settings,
        initial_color=color,
        available_colors=tuple(configured_colors),
        target_distance_m=ring_calibration.target_distance_m,
    )  #模式控制：固定控制或者ros控制

    if source_name == "realsense":  #获取图形信息：从realsense、camera、ideo中选择一个
        imu_enabled = bool(imu.get("enabled", True)) and not args.no_imu
        if (
            surface_pipeline.config.depth_enabled
            and surface_pipeline.config.use_gravity
            and surface_pipeline.config.gravity_required
            and not imu_enabled
        ):
            raise ValueError(
                "depth_surface.gravity_required为true时，imu.enabled不能为false"
            )
        source: Any = RealSenseRGBDSource(
            width,
            height,
            fps,
            imu_enabled=imu_enabled,
            accel_fps=int(imu.get("accel_fps", 63)),
            gravity_smoothing=float(imu.get("gravity_smoothing", 0.85)),
        )
    elif source_name == "camera":
        source = OpenCVFrameSource(camera_index, width, height, fps)
    elif source_name == "video":
        video_path = args.video or camera.get("video")
        if video_path is None:
            raise ValueError("source为video时必须提供视频路径")
        source = OpenCVFrameSource(str(video_path), width, height, fps)
    else:
        raise ValueError(f"不支持的camera.source：{source_name}")

    print_hz = float(output.get("print_hz", 10.0))  #是否显示视频
    no_display = (
        bool(args.no_display)
        if args.no_display is not None
        else not bool(output.get("display", True))
    )
    print(   #终端打印信息
        f"视觉模式控制已配置：source={source_name}, color={color}, "
        f"resolution={width}x{height}@{fps}, "
        f"imu={'off' if args.no_imu else 'configured'}",
        file=sys.stderr,
    )
    return run_mode_loop(
        source,
        mode_control,
        dispatcher,
        print_hz,
        no_display,
        target_distance_m=ring_calibration.target_distance_m,
    )

#main入口
def main() -> int:
    args = parse_application_args()
    try:
        return run(args)
    except (RuntimeError, TypeError, ValueError, cv2.error) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

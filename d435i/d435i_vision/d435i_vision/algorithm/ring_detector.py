"""RGB ring detector adapted from
``/home/book/Desktop/smart-logistics-vision/src/logistics_vision/detectors/circles.py``.

The preprocessing and Hough-circle approach derives from the CSDN article
“工创赛2025-智能物流搬运塔吊方案视觉开源（2分15秒）” by its original author:
https://blog.csdn.net/2303_80277100/article/details/150211965
The original article and this adapted work are licensed under Creative
Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0):
https://creativecommons.org/licenses/by-sa/4.0/deed.zh-hans
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np

from ..models import EllipseObservation, RingObservation
from .ring_tracker import RingTracker


@dataclass(frozen=True)
class RingConfig:
    #色环检测参数默认值
    dp: float = 1.5
    min_distance: float = 50.0
    param1: float = 100.0
    param2: float = 0.95
    min_radius: int = 15
    max_radius: int = 50
    collinearity_tolerance_px: float = 5.0
    ellipse_enabled: bool = True
    ellipse_min_contour_points: int = 20
    ellipse_min_axis_ratio: float = 0.35
    ellipse_max_fit_error: float = 0.20
    ellipse_max_center_distance_cm: float = 2.0
    tracking_enabled: bool = True
    tracking_smoothing: float = 0.45
    tracking_max_match_distance_px: float = 40.0
    tracking_max_missing_frames: int = 5
    ellipse_require_hough_proximity: bool = False
    ellipse_binary_threshold: int = 127
    ellipse_edge_sample_count: int = 72
    ellipse_edge_search_radius_px: int = 3
    ellipse_min_edge_support: float = 0.50
    ellipse_coverage_bin_count: int = 24
    ellipse_min_covered_bins: int = 14
    ellipse_min_major_axis_cm: float = 5.0
    ellipse_max_major_axis_cm: float = 11.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RingConfig":  #Mapping 代表映射关系，dictionary的父类
        names = {field.name for field in fields(cls)}  #fields()函数用于获取数据类的字段，返回值元素为Field对象的元组，.name是filed对象的属性，代表字段名
        values = {name: getattr(cls, name) for name in names}   #不能使用cls.name 因为这会查找RingConfig中叫做name的变量，而不是存储再name里面的变量
        values.update({name: data[name] for name in names if name in data})  #.update()方法用于更新
        for name in (
            "dp",
            "min_distance",
            "param1",
            "param2",
            "collinearity_tolerance_px",
            "ellipse_min_axis_ratio",
            "ellipse_max_fit_error",
            "ellipse_max_center_distance_cm",
            "tracking_smoothing",
            "tracking_max_match_distance_px",
            "ellipse_min_edge_support",
            "ellipse_min_major_axis_cm",
            "ellipse_max_major_axis_cm",
        ):
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):   #如果是bool型或者不是int或者float型，因为bool型是int型的子类
                raise TypeError(f"ring.{name} must be a finite number")
            if not math.isfinite(float(value)):   #检测是否是有限数
                raise ValueError(f"ring.{name} must be a finite number")
        for name in (
            "min_radius",
            "max_radius",
            "ellipse_min_contour_points",
            "tracking_max_missing_frames",
            "ellipse_binary_threshold",
            "ellipse_edge_sample_count",
            "ellipse_edge_search_radius_px",
            "ellipse_coverage_bin_count",
            "ellipse_min_covered_bins",
        ):
            if type(values[name]) is not int:
                raise TypeError(f"ring.{name} must be an int")
        for name in (
            "ellipse_enabled",
            "ellipse_require_hough_proximity",
            "tracking_enabled",
        ):
            if type(values[name]) is not bool:
                raise TypeError(f"ring.{name} must be a bool")
        config = cls(    #cls是RingConfig类，使用cls()等同于使用RingConfig(),用于创建一个RingConfig对象
            dp=values["dp"],
            min_distance=values["min_distance"],
            param1=values["param1"],
            param2=values["param2"],
            min_radius=values["min_radius"],
            max_radius=values["max_radius"],
            collinearity_tolerance_px=values["collinearity_tolerance_px"],
            ellipse_enabled=values["ellipse_enabled"],
            ellipse_min_contour_points=values["ellipse_min_contour_points"],
            ellipse_min_axis_ratio=values["ellipse_min_axis_ratio"],
            ellipse_max_fit_error=values["ellipse_max_fit_error"],
            ellipse_max_center_distance_cm=values[
                "ellipse_max_center_distance_cm"
            ],
            ellipse_require_hough_proximity=values[
                "ellipse_require_hough_proximity"
            ],
            tracking_enabled=values["tracking_enabled"],
            tracking_smoothing=values["tracking_smoothing"],
            tracking_max_match_distance_px=values[
                "tracking_max_match_distance_px"
            ],
            tracking_max_missing_frames=values["tracking_max_missing_frames"],
            ellipse_binary_threshold=values["ellipse_binary_threshold"],
            ellipse_edge_sample_count=values["ellipse_edge_sample_count"],
            ellipse_edge_search_radius_px=values[
                "ellipse_edge_search_radius_px"
            ],
            ellipse_min_edge_support=values["ellipse_min_edge_support"],
            ellipse_coverage_bin_count=values["ellipse_coverage_bin_count"],
            ellipse_min_covered_bins=values["ellipse_min_covered_bins"],
            ellipse_min_major_axis_cm=values["ellipse_min_major_axis_cm"],
            ellipse_max_major_axis_cm=values["ellipse_max_major_axis_cm"],
        )
        if config.dp <= 0.0:
            raise ValueError("ring.dp must be greater than 0")
        if config.min_distance <= 0.0 or config.param1 <= 0.0:
            raise ValueError("ring distances and thresholds must be greater than 0")
        if not 0.0 < config.param2 <= 1.0:
            raise ValueError("ring.param2 must be in (0, 1]")
        if config.min_radius < 0 or config.max_radius <= config.min_radius:
            raise ValueError("ring radius range must satisfy 0 <= min < max")
        if config.collinearity_tolerance_px < 0.0:
            raise ValueError("ring.collinearity_tolerance_px must be at least 0")
        if config.ellipse_min_contour_points < 5:
            raise ValueError("ring.ellipse_min_contour_points must be at least 5")
        if not 0.0 < config.ellipse_min_axis_ratio <= 1.0:
            raise ValueError("ring.ellipse_min_axis_ratio must be in (0, 1]")
        if not 0.0 < config.ellipse_max_fit_error <= 1.0:
            raise ValueError("ring.ellipse_max_fit_error must be in (0, 1]")
        if config.ellipse_max_center_distance_cm <= 0.0:
            raise ValueError(
                "ring.ellipse_max_center_distance_cm must be greater than 0"
            )
        if not 0.0 < config.tracking_smoothing <= 1.0:
            raise ValueError("ring.tracking_smoothing must be in (0, 1]")
        if config.tracking_max_match_distance_px <= 0.0:
            raise ValueError(
                "ring.tracking_max_match_distance_px must be greater than 0"
            )
        if config.tracking_max_missing_frames < 0:
            raise ValueError(
                "ring.tracking_max_missing_frames must be at least 0"
            )
        if not 0 <= config.ellipse_binary_threshold <= 255:
            raise ValueError("ring.ellipse_binary_threshold must be in [0, 255]")
        if not 8 <= config.ellipse_edge_sample_count <= 4096:
            raise ValueError(
                "ring.ellipse_edge_sample_count must be between 8 and 4096"
            )
        if config.ellipse_edge_search_radius_px < 0:
            raise ValueError(
                "ring.ellipse_edge_search_radius_px must be at least 0"
            )
        if not 0.0 < config.ellipse_min_edge_support <= 1.0:
            raise ValueError("ring.ellipse_min_edge_support must be in (0, 1]")
        if not 4 <= config.ellipse_coverage_bin_count <= config.ellipse_edge_sample_count:
            raise ValueError(
                "ring.ellipse_coverage_bin_count must be between 4 and "
                "ellipse_edge_sample_count"
            )
        if not 1 <= config.ellipse_min_covered_bins <= config.ellipse_coverage_bin_count:
            raise ValueError(
                "ring.ellipse_min_covered_bins must be between 1 and "
                "ellipse_coverage_bin_count"
            )
        if not (
            0.0 < config.ellipse_min_major_axis_cm
            <= config.ellipse_max_major_axis_cm
        ):
            raise ValueError(
                "ring ellipse major-axis range must satisfy 0 < min <= max"
            )
        return config


class RingDetector:
    """Detect zero or more in-frame rings and return them left-to-right."""

    def __init__(self, config: RingConfig | None = None) -> None:
        self.config = config or RingConfig()  #当输入的参数为None时，使用RingConfig()创建对象
        self.intermediates: Dict[str, np.ndarray] = {}
        self.ellipse_observations: Tuple[EllipseObservation, ...] = ()
        self.tracker = RingTracker(
            smoothing=self.config.tracking_smoothing,
            max_match_distance_px=self.config.tracking_max_match_distance_px,
            max_missing_frames=self.config.tracking_max_missing_frames,
        )

    def reset(self) -> None:
        """清除跨帧色环状态。"""

        self.tracker.reset()
        self.ellipse_observations = ()

    def detect(self, frame: np.ndarray) -> Tuple[RingObservation, ...]:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:   #判断是不是up.ndarray类型，是不是三通道数组
            raise ValueError("RingDetector requires a non-empty three-channel BGR frame")
        if frame.shape[0] == 0 or frame.shape[1] == 0:     #判断长宽是否为零
            raise ValueError("RingDetector requires a non-empty three-channel BGR frame")

        eroded = cv2.erode(frame, None, iterations=2)   #腐蚀图像，消除噪点，让亮色区域变小，暗色区域变大，第二个参数默认3*3,迭代两侧
        dilated = cv2.dilate(eroded, np.ones((7, 7), np.uint8), iterations=1)  #膨胀图像，与erode（）函数相反
        gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)  #将三通道RGB图像转化为单通道灰度图像
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   #获得一个椭圆形形状的结构元素，大小为5*5,cv2.MORPH_ELLIPSE代表椭圆形
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)  #高级形态学操作函数，cv2.MORPH_GRADIENT代表进行的是形态梯度学操作
        processed = cv2.GaussianBlur(gradient, (7, 7), 3, 3)   #高斯模糊处理，实际上是加权平均计算，（7,7）是高斯核大小，高斯核的大小一般都是正奇数，3,3分别代表在xy方向的标准差，标准差越小，中心权重越大，图像越清晰
        processed = cv2.convertScaleAbs(processed, alpha=4, beta=0)   #缩放像素值，增强对比度，alpha是缩放因子，beta是偏移量，最后取绝对值，高于255的取255
        processed = cv2.GaussianBlur(processed, (7, 7), 3, 3)  #再次高斯模糊处理，降低噪声
        _, threshold = cv2.threshold(processed, 70, 255, cv2.THRESH_BINARY)   #二值化灰度图，将高于70的像度设置为255,将低于70的像素值设置为0
        threshold = cv2.GaussianBlur(threshold, (9, 9), 3, 3)  #再次进行高斯模糊，降低噪声
        _, ellipse_binary = cv2.threshold(
            threshold,
            self.config.ellipse_binary_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        self.intermediates = {"gray": gray, "gradient": gradient, "threshold": threshold}
        self.ellipse_observations = (
            self._detect_ellipses(ellipse_binary)
            if self.config.ellipse_enabled
            else ()
        )

        circles = cv2.HoughCircles(
            threshold, cv2.HOUGH_GRADIENT_ALT, self.config.dp, self.config.min_distance,  #两个圆之间的最小距离
            param1=self.config.param1, param2=self.config.param2,
            minRadius=self.config.min_radius, maxRadius=self.config.max_radius,
        )
        height, width = frame.shape[:2]
        candidates = []
        if circles is not None:
            for x, y, radius in np.rint(circles[0]).astype(int):
                if (self.config.min_radius <= radius <= self.config.max_radius
                        and radius > 0 and radius <= x < width - radius
                        and radius <= y < height - radius):
                    candidates.append(RingObservation((int(x), int(y)), int(radius)))
        observations = []
        for candidate in sorted(candidates, key=lambda item: item.radius, reverse=True):
            is_concentric = any(
                np.hypot(candidate.center[0] - existing.center[0],
                         candidate.center[1] - existing.center[1])
                < max(4.0, self.config.min_distance / 4)
                for existing in observations
            )
            if not is_concentric:
                observations.append(candidate)
        observations.sort(key=lambda item: item.center[0])
        if not self.config.tracking_enabled:
            return tuple(observations)
        return self.tracker.update(observations)

    def _detect_ellipses(
        self,
        threshold: np.ndarray,
    ) -> Tuple[EllipseObservation, ...]:
        """从边缘二值图轮廓中拟合并筛选椭圆候选。"""

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )
        distance_to_edge = cv2.distanceTransform(
            np.where(threshold > 0, 0, 255).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        height, width = threshold.shape[:2]
        candidates = []
        for contour in contours:
            if len(contour) < self.config.ellipse_min_contour_points:
                continue
            try:
                (center_x, center_y), (axis_a, axis_b), angle = cv2.fitEllipse(
                    contour
                )
            except cv2.error:
                continue
            values = (center_x, center_y, axis_a, axis_b, angle)
            if not all(math.isfinite(float(value)) for value in values):
                continue
            if axis_a <= 0.0 or axis_b <= 0.0:
                continue

            if axis_a >= axis_b:
                major_axis = float(axis_a)
                minor_axis = float(axis_b)
                major_angle = float(angle) % 180.0
            else:
                major_axis = float(axis_b)
                minor_axis = float(axis_a)
                major_angle = (float(angle) + 90.0) % 180.0

            major_radius = major_axis / 2.0
            if not self.config.min_radius <= major_radius <= self.config.max_radius:
                continue
            if minor_axis / major_axis < self.config.ellipse_min_axis_ratio:
                continue
            if not (
                major_radius <= center_x < width - major_radius
                and major_radius <= center_y < height - major_radius
            ):
                continue

            fit_error = self._ellipse_fit_error(
                contour,
                (float(center_x), float(center_y)),
                major_axis,
                minor_axis,
                major_angle,
            )
            if fit_error > self.config.ellipse_max_fit_error:
                continue
            edge_support, covered_bins = self._ellipse_edge_metrics(
                distance_to_edge,
                (float(center_x), float(center_y)),
                major_axis,
                minor_axis,
                major_angle,
                self.config.ellipse_edge_sample_count,
                self.config.ellipse_coverage_bin_count,
                self.config.ellipse_edge_search_radius_px,
            )
            if edge_support < self.config.ellipse_min_edge_support:
                continue
            if covered_bins < self.config.ellipse_min_covered_bins:
                continue
            candidates.append(
                EllipseObservation(
                    center=(float(center_x), float(center_y)),
                    major_axis_px=major_axis,
                    minor_axis_px=minor_axis,
                    angle_deg=major_angle,
                    fit_error=fit_error,
                )
            )

        observations = []
        for candidate in sorted(
            candidates,
            key=lambda item: item.major_axis_px,
            reverse=True,
        ):
            is_concentric = any(
                math.hypot(
                    candidate.center[0] - existing.center[0],
                    candidate.center[1] - existing.center[1],
                )
                < max(4.0, self.config.min_distance / 4.0)
                for existing in observations
            )
            if not is_concentric:
                observations.append(candidate)
        observations.sort(key=lambda item: item.center[0])
        return tuple(observations)

    @staticmethod
    def _ellipse_edge_metrics(
        distance_to_edge: np.ndarray,
        center: Tuple[float, float],
        major_axis: float,
        minor_axis: float,
        angle_deg: float,
        sample_count: int,
        bin_count: int,
        search_radius_px: int,
    ) -> Tuple[float, int]:
        """沿真实周长均匀采样，计算边缘支持比例和覆盖区数量。"""

        major_radius = major_axis / 2.0
        minor_radius = minor_axis / 2.0
        dense_count = max(720, sample_count * 16)
        dense_angles = np.linspace(
            0.0,
            2.0 * np.pi,
            dense_count + 1,
            endpoint=True,
        )
        dense_x = major_radius * np.cos(dense_angles)
        dense_y = minor_radius * np.sin(dense_angles)
        segment_lengths = np.hypot(np.diff(dense_x), np.diff(dense_y))
        cumulative_lengths = np.concatenate(
            (np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths))
        )
        target_lengths = np.linspace(
            0.0,
            cumulative_lengths[-1],
            sample_count,
            endpoint=False,
        )
        angles = np.interp(target_lengths, cumulative_lengths, dense_angles)
        local_x = major_radius * np.cos(angles)
        local_y = minor_radius * np.sin(angles)
        rotation = math.radians(angle_deg)
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        columns = np.rint(center[0] + cosine * local_x - sine * local_y).astype(int)
        rows = np.rint(center[1] + sine * local_x + cosine * local_y).astype(int)

        height, width = distance_to_edge.shape
        inside = (
            (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        supported = np.zeros(sample_count, dtype=bool)
        supported[inside] = (
            distance_to_edge[rows[inside], columns[inside]] <= search_radius_px
        )
        support_ratio = float(np.count_nonzero(supported) / sample_count)
        bin_indices = np.arange(sample_count) * bin_count // sample_count
        covered_bins = int(np.unique(bin_indices[supported]).size)
        return support_ratio, covered_bins

    @staticmethod
    def _ellipse_fit_error(
        contour: np.ndarray,
        center: Tuple[float, float],
        major_axis: float,
        minor_axis: float,
        angle_deg: float,
    ) -> float:
        """返回轮廓点相对拟合椭圆的平均归一化径向误差。"""

        points = contour.reshape((-1, 2)).astype(np.float64, copy=False)
        delta_x = points[:, 0] - center[0]
        delta_y = points[:, 1] - center[1]
        angle = math.radians(angle_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated_x = cosine * delta_x + sine * delta_y
        rotated_y = -sine * delta_x + cosine * delta_y
        radial = np.sqrt(
            (rotated_x / (major_axis / 2.0)) ** 2
            + (rotated_y / (minor_axis / 2.0)) ** 2
        )
        return float(np.mean(np.abs(radial - 1.0)))

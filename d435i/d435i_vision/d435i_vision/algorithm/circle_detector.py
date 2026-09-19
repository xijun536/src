"""从上表面二值掩膜中检测近圆形轮廓。"""


# 这里仅仅进行圆形检测：它不直接判断原始图像，而是从生成的二值掩膜中筛选符合条件的圆形。

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from ..models import CircleObservation


class CircleDetector:
    """使用圆形度和最小外接圆检测近圆目标，不拟合椭圆。"""

    def __init__(
        self,
        min_area: float = 150.0,  #可以允许的最小的目标形状的面积，否则认为是干扰物
        min_contour_points: int = 20,   #可以通过的最小的轮廓点，否则认为不足以判断为目标形状
        min_radius_px: float = 10.0,   #最小半径
        max_radius_px: float = 300.0,   #最大半径
        min_circularity: float = 0.65,   #可以通过的最小的圆度
        min_fill_ratio: float = 0.65,    #最小的轮廓对外接圆的填充度
    ) -> None:
        if min_area < 0.0:
            raise ValueError("min_area 不能小于0")
        if not 0.0 < min_radius_px < max_radius_px:
            raise ValueError("圆半径范围必须满足 0 < min < max")
        if not 0.0 < min_circularity <= 1.0:
            raise ValueError("min_circularity 必须位于 (0, 1] 内")
        if not 0.0 < min_fill_ratio <= 1.0:
            raise ValueError("min_fill_ratio 必须位于 (0, 1] 内")

        self.min_area = float(min_area)
        self.min_contour_points = max(5, int(min_contour_points))
        self.min_radius_px = float(min_radius_px)
        self.max_radius_px = float(max_radius_px)
        self.min_circularity = float(min_circularity)
        self.min_fill_ratio = float(min_fill_ratio)

    def detect(self, mask: np.ndarray) -> List[CircleObservation]:
        """返回全部合格圆形，按轮廓面积从大到小排列。"""
        if mask.ndim != 2:
            raise ValueError("CircleDetector需要单通道二值掩膜")

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        height, width = mask.shape
        observations: List[CircleObservation] = []

        for contour in contours:  #contour是轮廓边界的若干点
            if len(contour) < self.min_contour_points:
                continue

            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            radius = float(radius)
            if not self.min_radius_px <= radius <= self.max_radius_px:
                continue
            if not (0.0 <= center_x < width and 0.0 <= center_y < height):
                continue

            circle_area = math.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0.0 else 0.0
            if fill_ratio < self.min_fill_ratio:
                continue

            observations.append(
                CircleObservation(
                    center=(float(center_x), float(center_y)),
                    radius=radius,
                    contour_area=area,
                    circularity=float(circularity),
                    fill_ratio=float(fill_ratio),
                )
            )

        observations.sort(key=lambda item: item.contour_area, reverse=True)
        return observations

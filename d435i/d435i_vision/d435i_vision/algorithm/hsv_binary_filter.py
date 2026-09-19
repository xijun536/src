"""HSV 二值化：只留下目标颜色对应的像素。"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..models import ColorRange


class HSVBinaryFilter:
    """把 BGR 图像转换为指定颜色的 0/255 二值掩膜。"""

    def __init__(self, color_ranges: Sequence[ColorRange]) -> None:
        if not color_ranges:
            raise ValueError("至少需要一组 HSV 颜色范围")

        self.color_ranges = []
        for lower, upper in color_ranges:
            lower_array = np.asarray(lower, dtype=np.uint8)
            upper_array = np.asarray(upper, dtype=np.uint8)
            if lower_array.shape != (3,) or upper_array.shape != (3,):
                raise ValueError("HSV 范围的上下界都必须包含 H、S、V 三个值")
            self.color_ranges.append((lower_array, upper_array))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """合并多段 HSV 范围，红色等跨 0 度颜色可以使用两段范围。"""
        if frame is None or frame.size == 0:
            raise ValueError("HSV 二值化收到空图像")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   #从bgr图像转换成hsv图像
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.color_ranges:
            current = cv2.inRange(hsv, lower, upper)   #满足hsv要求的像素为一，否则为零
            mask = cv2.bitwise_or(mask, current)   #进行按位或运算，因为红色有两个范围，符合任意一个都可以认为是红色
        return mask

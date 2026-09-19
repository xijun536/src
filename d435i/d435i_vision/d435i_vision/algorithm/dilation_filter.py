"""膨胀滤波：扩张二值区域并连接轮廓上的小缺口。"""

from __future__ import annotations

import cv2
import numpy as np


class DilationFilter:
    """使用椭圆结构元素执行膨胀。"""

    def __init__(self, kernel_size: int = 3) -> None:
        kernel_size = int(kernel_size)
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("morphology_size 必须是正奇数")
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )

    def apply(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """iterations=0 时直接返回掩膜副本。"""
        iterations = int(iterations)
        if iterations < 0:
            raise ValueError("膨胀次数不能小于 0")
        if iterations == 0:
            return mask.copy()
        return cv2.dilate(mask, self.kernel, iterations=iterations)

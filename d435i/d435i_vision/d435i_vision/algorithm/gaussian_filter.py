"""高斯滤波：抑制彩色图像中的细小噪声。"""

from __future__ import annotations

import cv2
import numpy as np


class GaussianFilter:
    """对 BGR 图像执行高斯模糊。"""

    def __init__(self, kernel_size: int = 5, sigma: float = 0.0) -> None:
        kernel_size = int(kernel_size)
        if kernel_size < 0:
            raise ValueError("gaussian_size 不能小于 0")
        if kernel_size > 1 and kernel_size % 2 == 0:
            raise ValueError("gaussian_size 必须是奇数；0 或 1 表示关闭")
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """返回滤波图像，不修改输入帧。"""
        if frame is None or frame.size == 0:
            raise ValueError("高斯滤波收到空图像")
        if self.kernel_size <= 1:
            return frame.copy()
        size = (self.kernel_size, self.kernel_size)
        return cv2.GaussianBlur(frame, size, self.sigma)

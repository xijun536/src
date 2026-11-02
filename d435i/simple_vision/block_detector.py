"""彩色圆形物块的二维识别，仅依赖 OpenCV 和 NumPy。

用法：blocks = detect_blocks(bgr_image, color="green")
返回按面积从大到小排列的字典列表；未识别到物块时返回 []。
"""

import math

import cv2
import numpy as np


# 沿用原项目 config.yaml 中的 HSV 阈值，红色跨越色相两端。
COLOR_RANGES = {
    "red": [((156, 60, 60), (180, 255, 255)), ((0, 60, 60), (6, 255, 255))],
    "green": [((38, 80, 45), (90, 255, 255))],
    "blue": [((106, 100, 45), (124, 255, 255))],
    "yellow": [((14, 80, 80), (22, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 55))],
    "light_blue": [((95, 50, 80), (104, 255, 255))],
}


def detect_blocks(image: np.ndarray, color: str = "green", min_area: float = 150.0) -> list[dict]:
    """输入 uint8 BGR 图像和目标颜色，返回圆心、半径、面积及圆形度。

    center_px 为 (横坐标, 纵坐标)，radius_px 单位为像素，area_px2 为轮廓面积。
    这里只按颜色和二维轮廓识别圆形物块，不区分物块上表面与同色侧面。
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8):
        raise ValueError("image 必须是非空的 uint8 BGR 三通道图像")
    if color not in COLOR_RANGES:
        raise ValueError(f"未知颜色 {color!r}，可选：{', '.join(COLOR_RANGES)}")
    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area 必须是非负有限数")

    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    blocks = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)
        if len(contour) < 20 or area < min_area or perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / perimeter**2
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if not 10 <= radius <= 300 or circularity < 0.45:
            continue
        if area / (math.pi * radius**2) < 0.65:
            continue
        blocks.append({
            "color": color,
            "center_px": (float(x), float(y)),
            "radius_px": float(radius),
            "area_px2": area,
            "circularity": float(circularity),
        })
    return sorted(blocks, key=lambda block: block["area_px2"], reverse=True)

"""色环的二维椭圆识别，仅依赖 OpenCV 和 NumPy。

用法：rings = detect_rings(bgr_image)
返回按中心从左到右排列的字典列表；未识别到色环时返回 []。
不区分色环颜色；轴长为完整直径，角度为长轴相对图像 X 轴的角度。

从原项目 d435i_vision/algorithm/ring_detector.py 精简而来。
预处理源自 xijun536/smart-logistics-vision：
https://github.com/xijun536/smart-logistics-vision
原文：https://blog.csdn.net/2303_80277100/article/details/150211965
改编内容继续采用 CC BY-SA 4.0：
https://creativecommons.org/licenses/by-sa/4.0/
"""

import math

import cv2
import numpy as np


def detect_rings(image: np.ndarray, min_radius: int = 15, max_radius: int = 100) -> list[dict]:
    """输入 uint8 BGR 图像，返回中心、长短轴、角度和拟合误差。

    min_radius/max_radius 是允许的长半轴像素范围。
    识别使用灰度边缘及椭圆几何，不进行深度测量、Hough 筛选或时序跟踪。
    """
    if (not isinstance(image, np.ndarray) or image.ndim != 3
            or image.shape[2] != 3 or image.size == 0 or image.dtype != np.uint8):
        raise ValueError("image 必须是非空的 uint8 BGR 三通道图像")
    if not (math.isfinite(min_radius) and math.isfinite(max_radius)
            and 0 < min_radius < max_radius):
        raise ValueError("半径范围必须满足 0 < min_radius < max_radius 且数值有限")

    eroded = cv2.erode(image, None, iterations=2)
    dilated = cv2.dilate(eroded, np.ones((7, 7), np.uint8))
    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    blurred = cv2.GaussianBlur(gradient, (7, 7), 3, 3)
    enhanced = cv2.convertScaleAbs(blurred, alpha=4)
    enhanced = cv2.GaussianBlur(enhanced, (7, 7), 3, 3)
    _, binary = cv2.threshold(enhanced, 70, 255, cv2.THRESH_BINARY)
    binary = cv2.GaussianBlur(binary, (9, 9), 3, 3)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    distance_to_edge = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 3)

    height, width = image.shape[:2]
    candidates = []
    for index, contour in enumerate(contours):
        # 闭合环边缘应围住内部区域，短弧形成的实心斑块不作为色环候选。
        if len(contour) < 20 or hierarchy[0, index, 2] < 0:
            continue
        try:
            (x, y), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        if not all(math.isfinite(v) for v in (x, y, axis_a, axis_b, angle)):
            continue
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        if not min_radius <= major / 2 <= max_radius or minor / major < 0.35:
            continue
        angle = (angle + (90 if axis_b > axis_a else 0)) % 180
        theta = math.radians(angle)
        cosine, sine = math.cos(theta), math.sin(theta)
        half_width = math.hypot(major / 2 * cosine, minor / 2 * sine)
        half_height = math.hypot(major / 2 * sine, minor / 2 * cosine)
        if not (half_width <= x < width - half_width
                and half_height <= y < height - half_height):
            continue
        points = contour.reshape(-1, 2) - (x, y)
        local_x = (points[:, 0] * cosine + points[:, 1] * sine) / (major / 2)
        local_y = (-points[:, 0] * sine + points[:, 1] * cosine) / (minor / 2)
        fit_error = float(np.mean(np.abs(np.hypot(local_x, local_y) - 1)))
        if fit_error > 0.20:
            continue
        if not _has_enough_edge(distance_to_edge, x, y, major, minor, theta):
            continue
        candidates.append({
            "center_px": (float(x), float(y)),
            "major_axis_px": float(major),
            "minor_axis_px": float(minor),
            "angle_deg": float(angle),
            "fit_error": fit_error,
        })

    # 同一色环的内外边缘会生成多个轮廓，优先保留外圈。
    rings = []
    for candidate in sorted(candidates, key=lambda ring: ring["major_axis_px"], reverse=True):
        if not any(math.dist(candidate["center_px"], ring["center_px"]) < 12.5 for ring in rings):
            rings.append(candidate)
    return sorted(rings, key=lambda ring: ring["center_px"][0])


def _has_enough_edge(distance: np.ndarray, x: float, y: float,
                     major: float, minor: float, theta: float) -> bool:
    """沿椭圆等弧长采样，过滤只有一小段弧线的误检。"""
    dense_angles = np.linspace(0, 2 * np.pi, 1153)
    dense_x = major / 2 * np.cos(dense_angles)
    dense_y = minor / 2 * np.sin(dense_angles)
    lengths = np.r_[0, np.cumsum(np.hypot(np.diff(dense_x), np.diff(dense_y)))]
    angles = np.interp(np.linspace(0, lengths[-1], 72, endpoint=False), lengths, dense_angles)
    local_x, local_y = major / 2 * np.cos(angles), minor / 2 * np.sin(angles)
    columns = np.rint(x + math.cos(theta) * local_x - math.sin(theta) * local_y).astype(int)
    rows = np.rint(y + math.sin(theta) * local_x + math.cos(theta) * local_y).astype(int)
    inside = (columns >= 0) & (columns < distance.shape[1]) & (rows >= 0) & (rows < distance.shape[0])
    supported = np.zeros(72, dtype=bool)
    supported[inside] = distance[rows[inside], columns[inside]] <= 3
    covered_bins = np.unique((np.arange(72) * 24 // 72)[supported]).size
    return bool(np.mean(supported) >= 0.5 and covered_bins >= 14)


if __name__ == "__main__":
    if __package__:
        from .camera_preview import main
    else:
        from camera_preview import main
    main(default_mode="rings")

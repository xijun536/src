"""在彩色图上标注深度上表面、圆心和运动状态。"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

from .models import EllipseObservation, PipelineResult, RingObservation
from .ring_geometry import analyze_ellipse_geometry


def draw_idle(frame: np.ndarray) -> np.ndarray:
    """Draw the persistent-camera idle state on a copy of the RGB frame."""

    output = frame.copy()
    cv2.putText(
        output,  #目标图像
        "IDLE",  #文本内容
        (12, 30),  #文本起点位置
        cv2.FONT_HERSHEY_SIMPLEX,  #字体类型
        0.8,  #字体大小
        (0, 220, 255),  #字体颜色
        2,  #字体粗细
        cv2.LINE_AA,  #线条类型
    )
    return output


def draw_rings(
    frame: np.ndarray,
    rings: Sequence[RingObservation],
    collinearity_tolerance_px: float = 5.0,
    distances_from_first_cm: Optional[Sequence[Optional[float]]] = None,
    ellipses: Sequence[EllipseObservation] = (),
) -> np.ndarray:
    """只绘制椭圆及其中心几何；Hough圆参数仅表示内部检测结果。"""

    output = frame.copy()
    geometry = analyze_ellipse_geometry(ellipses, collinearity_tolerance_px)
    centimeter_distances = (
        tuple(distances_from_first_cm)
        if distances_from_first_cm is not None
        else (None,) * len(ellipses)
    )
    if len(centimeter_distances) != len(ellipses):
        raise ValueError("厘米距离数量必须与椭圆数量一致")

    cv2.putText(
        output,
        f"ELLIPSE={len(ellipses)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if geometry is not None:
        origin = tuple(int(round(value)) for value in ellipses[0].center)
        for ellipse in ellipses[1:]:
            center = tuple(int(round(value)) for value in ellipse.center)
            cv2.line(output, origin, center, (255, 255, 0), 1, cv2.LINE_AA)

        for index, (ellipse, relative, distance_cm) in enumerate(
            zip(ellipses, geometry.relative_centers, centimeter_distances),
            start=1,
        ):
            row_y = (index + 1) * 24
            cv2.putText(
                output,
                f"E{index} = ({relative[0]:.0f}, {relative[1]:.0f})",
                (12, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                (
                    f"axes{index} = {ellipse.major_axis_px:.0f}x"
                    f"{ellipse.minor_axis_px:.0f} px"
                ),
                (430, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                (
                    f"d1-{index} = {distance_cm:.1f} cm"
                    if distance_cm is not None
                    else f"d1-{index} = N/A cm"
                ),
                (220, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    for index, ellipse in enumerate(ellipses, start=1):
        center = tuple(int(round(value)) for value in ellipse.center)
        axes = (
            max(1, int(round(ellipse.major_axis_px / 2.0))),
            max(1, int(round(ellipse.minor_axis_px / 2.0))),
        )
        cv2.ellipse(
            output,
            center,
            axes,
            ellipse.angle_deg,
            0.0,
            360.0,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        angle = math.radians(ellipse.angle_deg)
        direction_end = (
            int(round(center[0] + axes[0] * math.cos(angle))),
            int(round(center[1] + axes[0] * math.sin(angle))),
        )
        cv2.line(output, center, direction_end, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(output, center, 3, (255, 0, 255), -1)
        cv2.putText(
            output,
            f"E{index}",
            (center[0] + 6, center[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    status_y = (len(ellipses) + 2) * 24
    if geometry is not None and geometry.collinear is not None:
        status = "YES" if geometry.collinear else "NO"
        status_color = (0, 255, 0) if geometry.collinear else (0, 0, 255)
        cv2.putText(
            output,
            (
                f"COLLINEAR {status} | error={geometry.collinearity_error_px:.1f}px"
                f" tol={collinearity_tolerance_px:.1f}px"
            ),
            (12, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            status_color,
            2,
            cv2.LINE_AA,
        )
    elif ellipses:
        cv2.putText(
            output,
            "COLLINEAR N/A | exactly 3 ellipses required",
            (12, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_debug(
    frame: np.ndarray,
    result: PipelineResult,
    color_name: str,
    fps: float,
) -> np.ndarray:
    output = frame.copy()

    # 绿色半透明区域就是深度判断出的圆台上表面。
    if np.any(result.surface_mask):
        overlay = output.copy()
        overlay[result.surface_mask > 0] = (0, 180, 0)  #result.surface_mask>0代表逐元素比较，比较结果是有bool值构成的数组，起到了将所有符合条件的像素点都选出来的作用
        output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0.0)

    for observation in result.observations:
        center = tuple(int(round(value)) for value in observation.center)
        cv2.circle(output, center, int(round(observation.radius)), (140, 140, 140), 1)

    if result.selected is not None:
        center = tuple(int(round(value)) for value in result.selected.center)
        radius = int(round(result.selected.radius))
        cv2.circle(output, center, radius, (0, 220, 255), 2)
        cv2.circle(output, center, 4, (0, 220, 255), -1)

    if result.estimate is not None:
        center = tuple(int(round(value)) for value in result.estimate.center)
        velocity_end = (
            int(round(result.estimate.center[0] + result.estimate.velocity[0] * 0.20)),
            int(round(result.estimate.center[1] + result.estimate.velocity[1] * 0.20)),
        )
        cv2.circle(output, center, 6, (0, 255, 0), 2)
        cv2.arrowedLine(output, center, velocity_end, (255, 255, 0), 2, tipLength=0.25)

        speed = math.hypot(*result.estimate.velocity)
        status = "MEASURED" if result.estimate.measured else "PREDICTED"
        cv2.putText(
            output,
            (
                f"{color_name} {status} center=({result.estimate.center[0]:.1f},"
                f"{result.estimate.center[1]:.1f}) speed={speed:.1f}px/s"
            ),
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    depth_text = "DEPTH OFF"
    if result.depth_applied:
        depth_text = f"DEPTH ON | top planes={len(result.surface_planes)}"
        if result.surface_planes:
            plane = result.surface_planes[0]
            depth_text += (
                f" | z={plane.center[2]:.3f}m"
                f" | inliers={plane.inlier_ratio:.2f}"
            )
            if plane.normal_to_gravity_deg is not None:
                depth_text += f" | normal-gravity={plane.normal_to_gravity_deg:.1f}deg"
    cv2.putText(
        output,
        depth_text,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 220, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"FPS {fps:.1f} | q/ESC quit",
        (12, output.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output

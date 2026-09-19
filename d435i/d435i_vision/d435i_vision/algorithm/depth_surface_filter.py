"""用对齐深度和D435i重力方向分离圆台上表面与同色侧面。"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..models import CameraIntrinsics, SurfacePlaneObservation


class DepthSurfaceFilter:
    """在颜色区域中拟合三维平面，并只保留法向量朝上的平面。"""

    def __init__(
        self,
        min_depth_m: float = 0.15,
        max_depth_m: float = 2.0,
        plane_distance_m: float = 0.005,
        ransac_iterations: int = 180,
        min_points: int = 120,
        max_sample_points: int = 4000,
        min_inlier_ratio: float = 0.25,
        use_gravity: bool = True,
        gravity_required: bool = True,
        max_normal_gravity_angle_deg: float = 30.0,
        random_seed: int = 7,
    ) -> None:
        if not 0.0 < min_depth_m < max_depth_m:
            raise ValueError("深度范围必须满足 0 < min_depth_m < max_depth_m")
        if plane_distance_m <= 0.0:
            raise ValueError("plane_distance_m 必须大于0")
        if ransac_iterations < 1:
            raise ValueError("ransac_iterations 必须大于0")
        if min_points < 3 or max_sample_points < 3:
            raise ValueError("平面拟合点数不能小于3")
        if not 0.0 < min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio 必须位于 (0, 1] 内")
        if not 0.0 < max_normal_gravity_angle_deg < 90.0:
            raise ValueError("max_normal_gravity_angle_deg 必须位于 (0, 90) 内")

        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.plane_distance_m = float(plane_distance_m)
        self.ransac_iterations = int(ransac_iterations)
        self.min_points = int(min_points)
        self.max_sample_points = int(max_sample_points)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.use_gravity = bool(use_gravity)
        self.gravity_required = bool(gravity_required)
        self.max_normal_gravity_angle_deg = float(max_normal_gravity_angle_deg)
        self.min_gravity_alignment = math.cos(
            math.radians(self.max_normal_gravity_angle_deg)
        )
        self._rng = np.random.default_rng(random_seed)

    def apply(
        self,
        color_mask: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        gravity_camera: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, List[SurfacePlaneObservation]]:
        if color_mask.ndim != 2:
            raise ValueError("颜色掩膜必须是单通道图像")
        if depth_m.shape != color_mask.shape:
            raise ValueError("深度图必须已经对齐到彩色图")
        if (intrinsics.width, intrinsics.height) != (
            color_mask.shape[1],
            color_mask.shape[0],
        ):
            raise ValueError("相机内参尺寸必须与图像尺寸一致")

        gravity_unit = self._normalize_gravity(gravity_camera)
        if self.use_gravity and gravity_unit is None and self.gravity_required:
            # 安全优先：没有IMU方向时不能区分“水平上表面”和“较大的侧面”。
            return np.zeros_like(color_mask), []

        valid = (
            (color_mask > 0)  #颜色滤波，将元素改成bool值
            & np.isfinite(depth_m)   #有限深度
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )

        #表明有多少个连通的区域，labels是标明不同区域的二维数组，不同区域值不一样，stats承载每个区域的统计信息，第四个返回每个区域的质心
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            valid.astype(np.uint8),   #从bool值转换成uint8格式
            connectivity=8,   #相连考虑像素的相邻8个像素
        )
        surface_mask = np.zeros_like(color_mask)
        planes: List[SurfacePlaneObservation] = []

        for label in range(1, component_count):
            if int(stats[label, cv2.CC_STAT_AREA]) < self.min_points:   #降噪，联通区域过小的不算
                continue

            rows, columns = np.nonzero(labels == label)  #返回指定联通区域的坐标
            points = self._deproject(columns, rows, depth_m[rows, columns], intrinsics)  #计算出二维点的三维坐标
            initial_plane = self._ransac(
                points,
                gravity_unit if self.use_gravity else None,
            )
            if initial_plane is None:
                continue

            normal, offset = initial_plane
            initial_inliers = (
                np.abs(points @ normal + offset) <= self.plane_distance_m    #将每个点带入平面方程，然后计算距离
            )
            if int(np.count_nonzero(initial_inliers)) < self.min_points:  #两层循环
                continue

            normal, offset, center = self._refine(points[initial_inliers])
            gravity_alignment: Optional[float] = None
            if self.use_gravity and gravity_unit is not None:
                gravity_alignment = float(abs(np.dot(normal, gravity_unit)))
                # 再次检查精拟合后的法向量，避免混合点让法向量偏向侧面。
                if gravity_alignment < self.min_gravity_alignment:
                    continue

            inliers = np.abs(points @ normal + offset) <= self.plane_distance_m
            inlier_count = int(np.count_nonzero(inliers))
            inlier_ratio = inlier_count / len(points)
            if inlier_count < self.min_points or inlier_ratio < self.min_inlier_ratio:
                continue

            surface_mask[rows[inliers], columns[inliers]] = 255
            planes.append(
                SurfacePlaneObservation(
                    normal=tuple(float(value) for value in normal),
                    offset=float(offset),
                    center=tuple(float(value) for value in center),
                    inlier_count=inlier_count,
                    candidate_count=len(points),
                    gravity_alignment=gravity_alignment,
                )
            )

        planes.sort(key=lambda plane: plane.inlier_count, reverse=True)
        return surface_mask, planes


    #返回像素点的实际坐标
    @staticmethod
    def _deproject(
        columns: np.ndarray,
        rows: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        z = depth.astype(np.float64, copy=False)
        x = (columns.astype(np.float64) - intrinsics.ppx) * z / intrinsics.fx
        y = (rows.astype(np.float64) - intrinsics.ppy) * z / intrinsics.fy
        return np.column_stack((x, y, z))

    def _ransac(
        self,
        points: np.ndarray,
        gravity_unit: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, float]]:
        if len(points) < self.min_points:
            return None

        if len(points) > self.max_sample_points:    #如果点云中的点个数过多，随机取若干点
            indices = self._rng.choice(
                len(points),
                size=self.max_sample_points,
                replace=False,
            )
            samples = points[indices]
        else:
            samples = points

        best_normal: Optional[np.ndarray] = None
        best_offset = 0.0
        best_count = 0

        for _ in range(self.ransac_iterations):  #尝试self.ransac_iterations次构造平面
            chosen = self._rng.choice(len(samples), size=3, replace=False)
            point_1, point_2, point_3 = samples[chosen]
            normal = np.cross(point_2 - point_1, point_3 - point_1)   #计算两个向量的叉积
            norm = float(np.linalg.norm(normal))  #计算法向量的模
            if norm < 1e-9:   #如果两个向量共线，得到的向量为零向量，模很小
                continue
            normal /= norm  #将向量归一化 
            if (
                gravity_unit is not None  
                and abs(float(np.dot(normal, gravity_unit)))  #计算两个向量的点积
                < self.min_gravity_alignment
            ):
                # 圆台侧面即使点更多，也不能成为“上表面”候选。
                continue
            offset = -float(np.dot(normal, point_1))  #计算出平面的偏移
            count = int(
                np.count_nonzero(
                    np.abs(samples @ normal + offset) <= self.plane_distance_m
                )
            )
            if count > best_count:  #如果靠近这个平面的点数量大于现有的最大数量
                best_count = count
                best_normal = normal.copy()
                best_offset = offset

        if best_normal is None:
            return None
        return best_normal, best_offset

    @staticmethod
    def _normalize_gravity(
        gravity_camera: Optional[Tuple[float, float, float]],
    ) -> Optional[np.ndarray]:
        if gravity_camera is None:
            return None
        gravity = np.asarray(gravity_camera, dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            return None
        norm = float(np.linalg.norm(gravity))
        if norm < 1e-6:
            return None
        return gravity / norm

    @staticmethod
    def _refine(
        points: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        center = np.mean(points, axis=0)
        _, _, right_vectors = np.linalg.svd(points - center, full_matrices=False)   #奇异值分解，找到这些带点在哪个方向上分布最广，在哪个方向上分布最薄
        normal = right_vectors[-1]  #取最后一行，点云变化最小的方向，就是平面的法向量
        normal /= np.linalg.norm(normal)  #法向量归一化
        if normal[2] > 0.0:
            normal = -normal
        offset = -float(np.dot(normal, center))
        return normal, offset, center

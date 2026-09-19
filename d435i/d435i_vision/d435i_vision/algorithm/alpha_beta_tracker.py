"""轻量 alpha-beta 二维位置/速度跟踪器。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models import Point2D, TrackEstimate


class AlphaBetaTracker:
    """平滑椭圆中心，并在短时漏检时预测下一位置。"""

    def __init__(
        self,
        alpha: float = 0.75,
        beta: float = 0.15,
        max_missing: int = 5,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha 必须位于 (0, 1] 内")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta 必须位于 [0, 1] 内")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_missing = max(0, int(max_missing))
        self.position: Optional[np.ndarray] = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_timestamp: Optional[float] = None
        self.missing_frames = 0

    @property
    def initialized(self) -> bool:
        return self.position is not None and self.last_timestamp is not None

    def reset(self) -> None:
        """丢弃当前轨迹，下一次测量将建立新轨迹。"""
        self.position = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_timestamp = None
        self.missing_frames = 0

    def predict(self, timestamp: float) -> Optional[np.ndarray]:
        """计算指定时刻的位置预测，但不改变内部状态。"""
        if not self.initialized:
            return None
        assert self.position is not None
        assert self.last_timestamp is not None
        dt = max(0.0, float(timestamp) - self.last_timestamp)
        return self.position + self.velocity * dt

    def update(
        self,
        measurement: Optional[Point2D],  #检测到的圆心坐标
        timestamp: float,
    ) -> Optional[TrackEstimate]:
        """融合本帧测量；measurement=None 表示本帧漏检。"""
        timestamp = float(timestamp)

        if not self.initialized:
            if measurement is None:
                return None
            self.position = np.asarray(measurement, dtype=np.float64)    #将第一次测量到的位置设置为初始位置
            self.velocity.fill(0.0)   #将速度设置为（0.0,0.0）
            self.last_timestamp = timestamp
            self.missing_frames = 0
            return self._estimate(measured=True)

        assert self.position is not None    #断言，如果不成立就中断
        assert self.last_timestamp is not None

        dt = max(timestamp - self.last_timestamp, 1e-3)
        predicted = self.position + self.velocity * dt

        if measurement is None:     #没有检测到圆心
            self.position = predicted
            self.last_timestamp = timestamp
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.reset()
                return None
            return self._estimate(measured=False)

        measured_position = np.asarray(measurement, dtype=np.float64)
        residual = measured_position - predicted   #计算残差，实际测量的位置和实际位置差多少
        self.position = predicted + self.alpha * residual
        self.velocity = self.velocity + self.beta * residual / dt
        self.last_timestamp = timestamp
        self.missing_frames = 0
        return self._estimate(measured=True)

    def _estimate(self, measured: bool) -> TrackEstimate:
        assert self.position is not None
        return TrackEstimate(
            center=(float(self.position[0]), float(self.position[1])),
            velocity=(float(self.velocity[0]), float(self.velocity[1])),
            measured=measured,
            missing_frames=self.missing_frames,
        )

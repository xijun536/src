"""把本帧圆形候选与正在跟踪的目标关联。"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .models import CircleObservation


def choose_observation(
    observations: Sequence[CircleObservation],
    predicted: Optional[np.ndarray],
    max_distance: float,
) -> Optional[CircleObservation]:
    '''从可选的范围里选择最靠近predicted的那一个并返回'''
    if not observations:
        return None
    if predicted is None:
        return observations[0]

    predicted_x = float(predicted[0])
    predicted_y = float(predicted[1])
    closest = min(
        observations,
        key=lambda item: math.hypot(
            item.center[0] - predicted_x,
            item.center[1] - predicted_y,
        ),
    )
    distance = math.hypot(
        closest.center[0] - predicted_x,
        closest.center[1] - predicted_y,
    )
    return closest if distance <= float(max_distance) else None

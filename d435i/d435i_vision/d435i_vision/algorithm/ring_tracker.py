"""色环圆心和半径的轻量级多目标时序跟踪。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

from ..models import RingObservation


@dataclass
class _RingTrack:
    center_x: float
    center_y: float
    radius: float
    missing_frames: int = 0


@dataclass
class _FlowEdge:
    target: int
    reverse_index: int
    capacity: int
    cost: float


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    target: int,
    cost: float,
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _optimal_matches(
    tracks: Sequence[_RingTrack],
    detections: Sequence[RingObservation],
    maximum_distance: float,
) -> list[tuple[int, int]]:
    """返回最大匹配数量下总圆心距离最小的一对一关联。"""

    track_count = len(tracks)
    detection_count = len(detections)
    source = 0
    first_track = 1
    first_detection = first_track + track_count
    sink = first_detection + detection_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    candidate_edges = []

    for track_index in range(track_count):
        _add_flow_edge(graph, source, first_track + track_index, 0.0)
    for detection_index in range(detection_count):
        _add_flow_edge(graph, first_detection + detection_index, sink, 0.0)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            distance = math.hypot(
                detection.center[0] - track.center_x,
                detection.center[1] - track.center_y,
            )
            if distance <= maximum_distance:
                edge = _add_flow_edge(
                    graph,
                    first_track + track_index,
                    first_detection + detection_index,
                    distance,
                )
                candidate_edges.append((track_index, detection_index, edge))

    node_count = len(graph)
    while True:
        distances = [math.inf] * node_count
        previous_nodes = [-1] * node_count
        previous_edges = [-1] * node_count
        distances[source] = 0.0

        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not math.isfinite(distances[node]):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate_distance = distances[node] + edge.cost
                    if candidate_distance < distances[edge.target] - 1e-12:
                        distances[edge.target] = candidate_distance
                        previous_nodes[edge.target] = node
                        previous_edges[edge.target] = edge_index
                        changed = True
            if not changed:
                break

        if previous_nodes[sink] < 0:
            break
        node = sink
        while node != source:
            previous_node = previous_nodes[node]
            edge = graph[previous_node][previous_edges[node]]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous_node

    return [
        (track_index, detection_index)
        for track_index, detection_index, edge in candidate_edges
        if edge.capacity == 0
    ]


class RingTracker:
    """按圆心最近距离匹配色环，并平滑圆心与半径。"""

    def __init__(
        self,
        smoothing: float = 0.45,
        max_match_distance_px: float = 40.0,
        max_missing_frames: int = 5,
    ) -> None:
        self.smoothing = float(smoothing)
        self.max_match_distance_px = float(max_match_distance_px)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: list[_RingTrack] = []

    def reset(self) -> None:
        """清除全部历史轨迹。"""

        self._tracks.clear()

    def update(
        self,
        detections: Sequence[RingObservation],
    ) -> Tuple[RingObservation, ...]:
        matched_tracks = set()
        matched_detections = set()
        for track_index, detection_index in _optimal_matches(
            self._tracks,
            detections,
            self.max_match_distance_px,
        ):
            track = self._tracks[track_index]
            detection = detections[detection_index]
            alpha = self.smoothing
            track.center_x += alpha * (detection.center[0] - track.center_x)
            track.center_y += alpha * (detection.center[1] - track.center_y)
            track.radius += alpha * (detection.radius - track.radius)
            track.missing_frames = 0
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        retained_tracks = []
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.missing_frames += 1
            if track.missing_frames <= self.max_missing_frames:
                retained_tracks.append(track)
        self._tracks = retained_tracks
        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._tracks.append(
                    _RingTrack(
                        float(detection.center[0]),
                        float(detection.center[1]),
                        float(detection.radius),
                    )
                )

        observations = (
            RingObservation(
                (int(round(track.center_x)), int(round(track.center_y))),
                int(round(track.radius)),
            )
            for track in self._tracks
        )
        return tuple(sorted(observations, key=lambda item: item.center[0]))

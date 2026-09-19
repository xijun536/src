from __future__ import annotations

import json
import sys
from dataclasses import replace

import cv2
import numpy as np
import pytest

from d435i_vision import main
from d435i_vision.algorithm.hsv_binary_filter import HSVBinaryFilter
from d435i_vision.models import CameraFrame
from d435i_vision.ros_control import VisionMode
from d435i_vision.vision_modes import ModeFrameResult
from d435i_vision.vision_status import RingCalibrationStatus, SurfaceTrackingStatus


class FakeCV:
    def __init__(self, events=None, keys=()):
        self.closed = []
        self.shown = []
        self.destroy_all_calls = 0
        self.events = events
        self.keys = iter(keys)

    def destroyWindow(self, name):
        self.closed.append(name)

    def destroyAllWindows(self):
        self.destroy_all_calls += 1
        if self.events is not None:
            self.events.append("windows.close")

    def imshow(self, name, image):
        self.shown.append((name, image))

    def waitKey(self, delay):
        return next(self.keys, -1)


class FakeModeControl:
    def __init__(
        self,
        modes,
        events=None,
        spin_error=None,
        initial_mode=VisionMode.IDLE,
    ):
        self._modes = iter(modes)
        self._current_mode = initial_mode
        self._current_color = "green"
        self.events = events
        self.spin_error = spin_error
        self.start_calls = 0
        self.spin_calls = 0
        self.close_calls = 0
        self.published = []
        self.statuses = []

    def start(self):
        self.start_calls += 1
        if self.events is not None:
            self.events.append("control.start")

    def spin_once(self):
        self.spin_calls += 1
        if self.spin_error is not None:
            raise self.spin_error
        try:
            self._current_mode = next(self._modes)
        except StopIteration:
            return False
        return True

    @property
    def current_mode(self):
        return self._current_mode

    @property
    def current_color(self):
        return self._current_color

    def wire_value(self, mode):
        return {
            VisionMode.IDLE: 0,
            VisionMode.SURFACE_TRACKING: 1,
            VisionMode.RING_DETECTION: 2,
        }[mode]

    def publish_result(self, text):
        self.published.append(text)

    def publish_status(self, status):
        self.statuses.append(status)

    def close(self):
        self.close_calls += 1
        if self.events is not None:
            self.events.append("control.close")


class FakeSource:
    def __init__(self, frames, events=None):
        self.frames = iter(frames)
        self.events = events
        self.open_calls = 0
        self.read_calls = 0
        self.close_calls = 0

    def open(self):
        self.open_calls += 1
        if self.events is not None:
            self.events.append("source.open")

    def read(self):
        self.read_calls += 1
        return next(self.frames)

    def close(self):
        self.close_calls += 1
        if self.events is not None:
            self.events.append("source.close")


class FakeDispatcher:
    def __init__(self, fail=False):
        self.mode = None
        self.switches = []
        self.processed = []
        self.fail = fail

    def switch(self, mode):
        changed = self.mode is not None and mode is not self.mode
        self.mode = mode
        self.switches.append(mode)
        return changed

    def process(self, mode, wire_value, camera_frame, fps, render):
        if self.fail:
            raise ValueError("processing failed")
        self.processed.append((mode, wire_value, camera_frame, render))
        message = None if mode is VisionMode.IDLE else {
            "mode": wire_value,
            "timestamp": camera_frame.timestamp,
        }
        auxiliaries = {
            VisionMode.IDLE: {},
            VisionMode.SURFACE_TRACKING: {
                "hsv-color-mask": camera_frame.color,
                "depth-top-surface-mask": camera_frame.color,
            },
            VisionMode.RING_DETECTION: {
                "ring-gray": camera_frame.color,
                "ring-gradient": camera_frame.color,
                "ring-threshold": camera_frame.color,
            },
        }[mode]
        surface_status = (
            replace(SurfaceTrackingStatus.inactive("green"), active=True)
            if mode is VisionMode.SURFACE_TRACKING
            else None
        )
        ring_status = (
            replace(
                RingCalibrationStatus.inactive(0.30),
                active=True,
                failure_reason="waiting_for_three_rings",
            )
            if mode is VisionMode.RING_DETECTION
            else None
        )
        return ModeFrameResult(
            message,
            camera_frame.color if render else None,
            auxiliaries if render else {},
            surface_status,
            ring_status,
        )


def frames(*timestamps):
    return [
        CameraFrame(np.full((2, 2, 3), index, np.uint8), timestamp)
        for index, timestamp in enumerate(timestamps)
    ]


@pytest.mark.parametrize("color", ("yellow", "black", "light_blue"))
def test_argument_parser_accepts_new_surface_colors(color):
    args = main.build_argument_parser().parse_args(["--color", color])

    assert args.color == color


@pytest.mark.parametrize(
    "ros_args",
    (
        ("--ros-args",),
        ("--ros-args", "-r", "__node:=d435i_test"),
    ),
)
def test_main_removes_ros_args_and_preserves_application_options(
    monkeypatch, ros_args
):
    parsed = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["d435i_vision", "--color", "blue", "--no-display", *ros_args],
    )
    monkeypatch.setattr(main, "run", lambda args: parsed.append(args) or 0)

    assert main.main() == 0
    assert len(parsed) == 1
    assert parsed[0].color == "blue"
    assert parsed[0].no_display is True


def test_main_rejects_unknown_non_ros_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["d435i_vision", "--colro", "blue"])
    monkeypatch.setattr(
        main,
        "run",
        lambda args: pytest.fail("run must not receive invalid application arguments"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main.main()

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("color", "matching_hsv", "nonmatching_hsv"),
    (
        ("yellow", (17, 255, 169), (0, 255, 255)),
        ("black", (0, 0, 20), (0, 0, 255)),
        ("light_blue", (101, 255, 147), (108, 255, 107)),
    ),
)
def test_new_surface_color_configs_select_representative_pixels(
    color,
    matching_hsv,
    nonmatching_hsv,
):
    config = main.load_yaml(main.DEFAULT_CONFIG_PATH)
    hsv_pixels = np.asarray([[matching_hsv, nonmatching_hsv]], dtype=np.uint8)
    bgr_pixels = cv2.cvtColor(hsv_pixels, cv2.COLOR_HSV2BGR)
    mask = HSVBinaryFilter(main.color_ranges(config, color)).apply(bgr_pixels)

    assert mask.tolist() == [[255, 0]]


def test_emit_message_prints_and_publishes_identical_serialization(capsys):
    control = FakeModeControl(())

    main.emit_message({"mode": 2, "label": "色环"}, control)

    printed = capsys.readouterr().out.strip()
    assert printed == '{"mode":2,"label":"色环"}'
    assert control.published == [printed]


def test_close_inactive_windows_closes_only_old_auxiliary_windows():
    fake_cv = FakeCV()

    main.close_inactive_windows(
        VisionMode.SURFACE_TRACKING, VisionMode.RING_DETECTION, fake_cv
    )
    assert fake_cv.closed == ["hsv-color-mask", "depth-top-surface-mask"]

    fake_cv.closed.clear()
    main.close_inactive_windows(VisionMode.RING_DETECTION, VisionMode.IDLE, fake_cv)
    assert fake_cv.closed == ["ring-gray", "ring-gradient", "ring-threshold"]
    assert "vision-main" not in fake_cv.closed


def test_shared_source_switches_idle_surface_ring_idle(capsys):
    events = []
    camera_frames = frames(0.0, 0.1, 0.2, 0.3)
    source = FakeSource(camera_frames, events)
    control = FakeModeControl(
        [
            VisionMode.IDLE,
            VisionMode.SURFACE_TRACKING,
            VisionMode.RING_DETECTION,
            VisionMode.IDLE,
        ],
        events,
    )
    dispatcher = FakeDispatcher()
    fake_cv = FakeCV(events)

    assert main.run_mode_loop(
        source,
        control,
        dispatcher,
        print_hz=0.5,
        no_display=True,
        cv_module=fake_cv,
    ) == 0

    assert events == [
        "control.start",
        "source.open",
        "source.close",
        "control.close",
    ]
    assert source.open_calls == source.close_calls == 1
    assert source.read_calls == 4
    assert control.start_calls == control.close_calls == 1
    assert [call[0] for call in dispatcher.processed] == [
        VisionMode.IDLE,
        VisionMode.SURFACE_TRACKING,
        VisionMode.RING_DETECTION,
        VisionMode.IDLE,
    ]
    assert [call[2] for call in dispatcher.processed] == camera_frames
    assert all(call[3] is False for call in dispatcher.processed)
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item.get("event") for item in messages].count("mode_changed") == 3
    assert [item["mode"] for item in messages if "timestamp" in item] == [1, 2]
    assert control.published == [
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in messages
    ]
    assert fake_cv.shown == []
    assert fake_cv.closed == []
    assert fake_cv.destroy_all_calls == 0


def test_display_uses_main_and_mode_specific_windows_and_q_exits():
    source = FakeSource(frames(1.0, 2.0))
    control = FakeModeControl(
        [VisionMode.SURFACE_TRACKING, VisionMode.RING_DETECTION]
    )
    dispatcher = FakeDispatcher()
    fake_cv = FakeCV(keys=(-1, ord("q")))

    main.run_mode_loop(source, control, dispatcher, 10.0, False, fake_cv)

    assert [name for name, _ in fake_cv.shown] == [
        "vision-main",
        "hsv-color-mask",
        "depth-top-surface-mask",
        "vision-main",
        "ring-gray",
        "ring-gradient",
        "ring-threshold",
    ]
    assert fake_cv.closed == ["hsv-color-mask", "depth-top-surface-mask"]
    assert fake_cv.destroy_all_calls == 1


def test_first_spin_does_not_close_initial_mode_windows_before_they_are_shown():
    class StrictFakeCV(FakeCV):
        def destroyWindow(self, name):
            created = {shown_name for shown_name, _ in self.shown}
            if name not in created:
                raise RuntimeError(f"window was never created: {name}")
            super().destroyWindow(name)

    source = FakeSource(frames(1.0))
    control = FakeModeControl(
        [VisionMode.RING_DETECTION],
        initial_mode=VisionMode.SURFACE_TRACKING,
    )
    fake_cv = StrictFakeCV(keys=(ord("q"),))

    main.run_mode_loop(source, control, FakeDispatcher(), 10.0, False, fake_cv)

    assert [name for name, _ in fake_cv.shown] == [
        "vision-main",
        "ring-gray",
        "ring-gradient",
        "ring-threshold",
    ]
    assert fake_cv.closed == []


def test_spin_false_exits_without_reading_a_frame():
    source = FakeSource([])
    control = FakeModeControl([])

    main.run_mode_loop(source, control, FakeDispatcher(), 10.0, True, FakeCV())

    assert source.read_calls == 0
    assert source.close_calls == control.close_calls == 1


def test_processing_exception_closes_source_windows_then_control():
    events = []
    source = FakeSource(frames(1.0), events)
    control = FakeModeControl([VisionMode.SURFACE_TRACKING], events)
    fake_cv = FakeCV(events)

    with pytest.raises(ValueError, match="processing failed"):
        main.run_mode_loop(
            source, control, FakeDispatcher(fail=True), 10.0, False, fake_cv
        )

    assert events[-3:] == ["source.close", "windows.close", "control.close"]
    assert source.close_calls == control.close_calls == 1


def test_polling_failure_is_runtime_error_and_still_cleans_up():
    source = FakeSource([])
    control = FakeModeControl([], spin_error=OSError("executor broke"))

    with pytest.raises(RuntimeError, match="poll ROS mode control"):
        main.run_mode_loop(source, control, FakeDispatcher(), 10.0, True, FakeCV())

    assert source.close_calls == control.close_calls == 1


def test_every_processed_frame_publishes_one_typed_status():
    source = FakeSource(frames(0.0, 0.1, 0.2))
    control = FakeModeControl([
        VisionMode.IDLE,
        VisionMode.SURFACE_TRACKING,
        VisionMode.RING_DETECTION,
    ])

    main.run_mode_loop(source, control, FakeDispatcher(), 10.0, True, FakeCV())

    assert [status.frame_sequence for status in control.statuses] == [1, 2, 3]
    assert [status.current_mode_name for status in control.statuses] == [
        "idle",
        "surface_tracking",
        "ring_detection",
    ]


def test_typed_status_is_not_limited_by_json_print_rate():
    source = FakeSource(frames(0.0, 0.01, 0.02))
    control = FakeModeControl([VisionMode.IDLE] * 3)

    main.run_mode_loop(source, control, FakeDispatcher(), 0.0, True, FakeCV())

    assert len(control.statuses) == 3
    assert all(not status.surface.active for status in control.statuses)
    assert all(not status.ring.active for status in control.statuses)


def test_failed_frame_does_not_publish_partial_status():
    source = FakeSource(frames(0.0))
    control = FakeModeControl([VisionMode.RING_DETECTION])

    with pytest.raises(ValueError, match="processing failed"):
        main.run_mode_loop(
            source, control, FakeDispatcher(fail=True), 10.0, True, FakeCV()
        )

    assert control.statuses == []

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from d435i_vision import main
from d435i_vision.algorithm.pipeline import CircleSurfaceTrackingPipeline, PipelineConfig
from d435i_vision.models import CameraFrame
from d435i_vision.ros_control import (
    RosModeControl,
    RosSettings,
    StaticModeControl,
    VisionMode,
    build_mode_control,
)
from d435i_vision.vision_modes import ModeFrameResult, VisionModeDispatcher


class FakeUInt8:
    def __init__(self, data=0):
        self.data = data


class FakeString:
    def __init__(self, data=""):
        self.data = data


class FakeSubscription:
    def __init__(self, message_type, topic, callback, qos_depth):
        self.message_type = message_type
        self.topic = topic
        self.callback = callback
        self.qos_depth = qos_depth


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.subscriptions = []
        self.destroyed = False

    def create_subscription(self, message_type, topic, callback, qos_depth):
        subscription = FakeSubscription(message_type, topic, callback, qos_depth)
        self.subscriptions.append(subscription)
        return subscription

    def destroy_node(self):
        self.destroyed = True


class FakeRclpy:
    def __init__(self):
        self.ok_value = False
        self.nodes = []
        self.spin_calls = []

    def ok(self):
        return self.ok_value

    def init(self):
        self.ok_value = True

    def create_node(self, name):
        node = FakeNode(name)
        self.nodes.append(node)
        return node

    def spin_once(self, node, timeout_sec):
        self.spin_calls.append((node, timeout_sec))

    def shutdown(self):
        self.ok_value = False


class RosColorControlTests(unittest.TestCase):
    def test_ros_settings_parse_configurable_color_topic(self):
        default = RosSettings.from_mapping({})
        custom = RosSettings.from_mapping(
            {"ros": {"color_topic": "/picker/surface_color"}}
        )

        self.assertEqual(default.color_topic, "/vision_color")
        self.assertEqual(custom.color_topic, "/picker/surface_color")

    def test_ros_control_updates_valid_color_and_ignores_invalid_color(self):
        fake_rclpy = FakeRclpy()
        settings = RosSettings.from_mapping({})
        control = RosModeControl(
            settings,
            fake_rclpy,
            FakeUInt8,
            FakeString,
            initial_color="green",
            available_colors=("green", "yellow"),
        )
        control.start()
        node = fake_rclpy.nodes[-1]
        subscriptions = {item.topic: item for item in node.subscriptions}

        subscriptions["/vision_color"].callback(FakeString("yellow"))
        self.assertEqual(control.current_color, "yellow")

        errors = io.StringIO()
        with redirect_stderr(errors):
            subscriptions["/vision_color"].callback(FakeString("purple"))
        self.assertEqual(control.current_color, "yellow")
        self.assertIn("purple", errors.getvalue())

        self.assertTrue(control.spin_once())
        self.assertEqual(fake_rclpy.spin_calls, [(node, 0.0)])
        control.close()

    def test_static_control_keeps_configured_initial_color(self):
        settings = RosSettings.from_mapping({"ros": {"enabled": False}})

        control = build_mode_control(
            settings,
            initial_color="light_blue",
            available_colors=("green", "light_blue"),
        )

        self.assertIsInstance(control, StaticModeControl)
        self.assertEqual(control.current_color, "light_blue")


class SurfaceColorSwitchTests(unittest.TestCase):
    def test_pipeline_replaces_hsv_ranges_and_resets_tracking(self):
        pipeline = CircleSurfaceTrackingPipeline(
            [((38, 80, 45), (90, 255, 255))],
            PipelineConfig(
                gaussian_size=1,
                open_iterations=0,
                close_iterations=0,
                depth_enabled=False,
                depth_required=False,
            ),
        )
        pipeline.tracker.update((10.0, 10.0), 1.0)
        self.assertTrue(pipeline.tracker.initialized)

        pipeline.set_color_ranges([((20, 80, 80), (35, 255, 255))])

        self.assertFalse(pipeline.tracker.initialized)
        yellow_bgr = np.asarray([[[0, 255, 255]]], dtype=np.uint8)
        self.assertEqual(int(pipeline.hsv_binary.apply(yellow_bgr)[0, 0]), 255)

    def test_dispatcher_changes_only_to_a_configured_color(self):
        class SurfacePipelineSpy:
            def __init__(self):
                self.applied_ranges = []

            def set_color_ranges(self, ranges):
                self.applied_ranges.append(ranges)

            def reset(self):
                pass

        surface = SurfacePipelineSpy()
        yellow_ranges = [((20, 80, 80), (35, 255, 255))]
        dispatcher = VisionModeDispatcher(
            surface,
            object(),
            "green",
            {"green": [((38, 80, 45), (90, 255, 255))], "yellow": yellow_ranges},
        )

        self.assertTrue(dispatcher.switch_color("yellow"))
        self.assertEqual(dispatcher.color, "yellow")
        self.assertEqual(surface.applied_ranges, [yellow_ranges])
        self.assertFalse(dispatcher.switch_color("yellow"))
        with self.assertRaises(ValueError):
            dispatcher.switch_color("purple")


class RuntimeColorSwitchTests(unittest.TestCase):
    def test_main_loop_applies_ros_color_change_before_processing_next_frame(self):
        class Control:
            def __init__(self):
                self.colors = iter(("green", "yellow"))
                self._current_color = "green"

            def start(self):
                pass

            def spin_once(self):
                try:
                    self._current_color = next(self.colors)
                except StopIteration:
                    pass
                return True

            @property
            def current_mode(self):
                return VisionMode.IDLE

            @property
            def current_color(self):
                return self._current_color

            def wire_value(self, mode):
                return 0

            def publish_result(self, text):
                pass

            def publish_status(self, status):
                pass

            def close(self):
                pass

        class Source:
            def __init__(self):
                self.frames = iter(
                    (
                        CameraFrame(np.zeros((2, 2, 3), np.uint8), 1.0),
                        CameraFrame(np.zeros((2, 2, 3), np.uint8), 2.0),
                    )
                )

            def open(self):
                pass

            def read(self):
                try:
                    return next(self.frames)
                except StopIteration as exc:
                    raise EOFError from exc

            def close(self):
                pass

        class Dispatcher:
            def __init__(self):
                self.colors = []

            def switch(self, mode):
                return False

            def switch_color(self, color):
                self.colors.append(color)
                return True

            def process(self, mode, wire_value, frame, fps, render):
                return ModeFrameResult(None, None, {})

        dispatcher = Dispatcher()

        output = io.StringIO()
        with redirect_stdout(output):
            result = main.run_mode_loop(
                Source(), Control(), dispatcher, 0.0, True
            )

        self.assertEqual(result, 0)
        self.assertEqual(dispatcher.colors, ["yellow"])
        self.assertIn('"event":"surface_color_changed"', output.getvalue())


if __name__ == "__main__":
    unittest.main()

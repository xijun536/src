from __future__ import annotations

import unittest

import cv2
import numpy as np

from d435i_vision import main
from d435i_vision.algorithm.hsv_binary_filter import HSVBinaryFilter


class SurfaceColorRangeTests(unittest.TestCase):
    def test_measured_blue_light_blue_and_yellow_are_mutually_exclusive(self):
        config = main.load_yaml(main.DEFAULT_CONFIG_PATH)
        measured_hsv = np.asarray(
            [[(108, 255, 107), (101, 255, 147), (17, 255, 169)]],
            dtype=np.uint8,
        )
        measured_bgr = cv2.cvtColor(measured_hsv, cv2.COLOR_HSV2BGR)

        masks = {
            color: HSVBinaryFilter(main.color_ranges(config, color))
            .apply(measured_bgr)
            .tolist()[0]
            for color in ("blue", "light_blue", "yellow")
        }

        self.assertEqual(masks["blue"], [255, 0, 0])
        self.assertEqual(masks["light_blue"], [0, 255, 0])
        self.assertEqual(masks["yellow"], [0, 0, 255])


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest

import numpy as np


fake_api = types.ModuleType("rknnlite.api")
fake_api.RKNNLite = object
fake_package = types.ModuleType("rknnlite")
fake_package.api = fake_api
sys.modules.setdefault("rknnlite", fake_package)
sys.modules.setdefault("rknnlite.api", fake_api)

from inference.rknn.mobilenetv3_age import MobileNetV3Age


class MobileNetV3AgeTest(unittest.TestCase):
    def test_prepare_input_converts_bgr_to_rgb_float32(self):
        bgr = np.zeros((8, 6, 3), dtype=np.uint8)
        bgr[:, :, :] = [10, 20, 30]

        result = MobileNetV3Age.prepare_input(bgr)

        self.assertEqual((1, 112, 112, 3), result.shape)
        self.assertEqual(np.float32, result.dtype)
        np.testing.assert_array_equal(result[0, 0, 0], [30.0, 20.0, 10.0])

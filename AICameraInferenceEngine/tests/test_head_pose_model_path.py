import unittest

from factory.HeadPoseModelPath import resolve_head_pose_model_path


class HeadPoseModelPathTest(unittest.TestCase):
    def test_default_platform_uses_rk3566_model(self):
        self.assertIn("/rk3566/head_pose_", resolve_head_pose_model_path())

    def test_rk3588_platform_uses_rk3588_model(self):
        self.assertIn("/rk3588/head_pose_", resolve_head_pose_model_path("rk3588"))

    def test_invalid_platform_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RKNPU platform"):
            resolve_head_pose_model_path("rk9999")

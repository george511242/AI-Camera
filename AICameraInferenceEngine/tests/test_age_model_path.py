import unittest

from factory.AgeModelPath import resolve_age_model, resolve_age_model_path


class AgeModelPathTest(unittest.TestCase):
    def test_default_platform_uses_rk3566_model(self):
        self.assertEqual(
            "/model/versions/v2_finetuned/rknn/rk3566/age_ssrnet_v2_64x64_fp16.rknn",
            resolve_age_model_path(),
        )

    def test_rk3588_platform_uses_rk3588_model(self):
        self.assertIn(
            "/v2_finetuned/rknn/rk3588/age_ssrnet_v2_",
            resolve_age_model_path("rk3588"),
        )

    def test_v4_rk3566_selection(self):
        model = resolve_age_model("v4", "rk3566")
        self.assertEqual("mobilenetv3", model["implementation"])
        self.assertIn("/v4_age_mobilenetv3/rknn/rk3566/", model["path"])

    def test_v4_rk3588_selection(self):
        model = resolve_age_model("V4", "RK3588")
        self.assertEqual("v4", model["version"])
        self.assertIn("/rk3588/age_v4_mobilenetv3_large_112_fp16.rknn", model["path"])

    def test_invalid_platform_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported RKNPU platform"):
            resolve_age_model_path("rk9999")

    def test_invalid_version_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Age model version"):
            resolve_age_model("v999", "rk3588")

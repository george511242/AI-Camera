import unittest

from factory.GenderModelPath import resolve_gender_model_path


class GenderModelPathTest(unittest.TestCase):
    def test_default_platform_uses_rk3566_model(self):
        self.assertEqual(
            resolve_gender_model_path(),
            "/model/versions/v1_baseline/rknn/rk3566/gender_ssrnet_wiki_64x64_fp16.rknn",
        )

    def test_rk3588_platform_uses_fp16_model(self):
        self.assertEqual(
            resolve_gender_model_path("rk3588"),
            "/model/versions/v1_baseline/rknn/rk3588/gender_ssrnet_wiki_64x64_fp16.rknn",
        )

    def test_invalid_platform_fails_with_clear_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported RKNPU platform 'rk9999'. Supported platforms: rk3566, rk3588",
        ):
            resolve_gender_model_path("rk9999")


if __name__ == "__main__":
    unittest.main()

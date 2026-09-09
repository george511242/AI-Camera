import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools" / "train_ssrnet_age_v3.py"


@unittest.skipUnless(
    importlib.util.find_spec("tensorflow"), "TensorFlow is only in the Age training env"
)
class AgeV3MetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(MODULE_PATH.parent))
        spec = importlib.util.spec_from_file_location("train_ssrnet_age_v3", MODULE_PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_metrics_include_bias_and_age_buckets(self):
        import numpy as np

        rows = [{"age": 5}, {"age": 25}, {"age": 80}]
        result = self.module.metrics_from_predictions(rows, np.array([10, 24, 70]))
        self.assertAlmostEqual(result["mae"], 16 / 3, places=6)
        self.assertAlmostEqual(result["mean_signed_error"], -2.0)
        self.assertEqual(result["age_buckets"]["0-9"]["count"], 1)
        self.assertEqual(result["age_buckets"]["70+"]["count"], 1)


if __name__ == "__main__":
    unittest.main()

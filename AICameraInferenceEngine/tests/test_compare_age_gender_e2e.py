import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "compare_age_gender_e2e.py"
SPEC = importlib.util.spec_from_file_location("compare_age_gender_e2e", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def row(source_id, gender, age, predicted_age, predicted_gender, detected=True):
    return {
        "dataset": "utkface",
        "source_id": source_id,
        "target_face_height_px": 80,
        "gt_gender": gender,
        "gt_age": age,
        "predicted_age": predicted_age,
        "predicted_gender": predicted_gender,
        "detected_bbox": [0.1, 0.1, 0.2, 0.2] if detected else None,
        "yunet_score": 0.9 if detected else None,
        "iou": 0.8 if detected else 0.0,
    }


class AgeGenderComparisonTest(unittest.TestCase):
    def test_conditional_metrics_exclude_only_yunet_failures(self):
        rows = [
            row("male-hit", 0, 20, 22, 0.9),
            row("female-miss", 1, 30, 40, 0.9),
            row("male-undetected", 0, 40, None, None, detected=False),
            row("female-hit", 1, 50, 48, 0.1),
        ]

        age = comparison.age_metrics(rows)
        self.assertEqual((age["e2e_hits"], age["total"]), (2, 4))
        self.assertEqual((age["conditional_hits"], age["detected"]), (2, 3))
        self.assertAlmostEqual(age["mae"], 14 / 3)

        e2e = comparison.balanced_accuracy(rows)
        conditional = comparison.balanced_accuracy(rows, conditional=True)
        self.assertAlmostEqual(e2e["value"], 0.5)
        self.assertAlmostEqual(conditional["value"], 0.75)

    def test_pair_validation_rejects_changed_yunet_result(self):
        before = [row("sample", 0, 20, 20, 0.9)]
        after = [dict(before[0], iou=0.4)]

        with self.assertRaisesRegex(ValueError, "YuNet results differ"):
            comparison.validate_pair(before, after)


if __name__ == "__main__":
    unittest.main()

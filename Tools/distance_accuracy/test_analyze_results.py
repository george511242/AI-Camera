import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name('analyze_results.py')
SPEC = importlib.util.spec_from_file_location('analyze_results', MODULE_PATH)
analyze_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_results)


class MetricsTest(unittest.TestCase):
    def test_yunet_failure_excludes_downstream_predictions(self):
        row = {
            'failure_reason': 'wrong_face_matched',
            'predicted_age': 30.0,
            'gt_age': 30,
            'predicted_gender': 0.9,
            'gt_gender': 0,
        }

        result = analyze_results.metrics(
            {('utkface', None): [row]}, gender_positive='male'
        )

        self.assertEqual(result['age'], (0, 1))
        self.assertEqual(result['gender'], (0.0, 1))

    def test_gender_mapping_is_explicit(self):
        rows = [
            {'failure_reason': None, 'predicted_age': None, 'gt_age': 20,
             'predicted_gender': 0.9, 'gt_gender': 0},
            {'failure_reason': None, 'predicted_age': None, 'gt_age': 20,
             'predicted_gender': 0.1, 'gt_gender': 1},
        ]

        male_positive = analyze_results.metrics(
            {('utkface', None): rows}, gender_positive='male'
        )
        female_positive = analyze_results.metrics(
            {('utkface', None): rows}, gender_positive='female'
        )

        self.assertEqual(male_positive['gender'], (1.0, 2))
        self.assertEqual(female_positive['gender'], (0.0, 2))

    def test_age_gender_diagnostics(self):
        rows = [
            {'failure_reason': None, 'predicted_age': 18.0, 'gt_age': 20,
             'predicted_gender': 0.9, 'gt_gender': 0},
            {'failure_reason': None, 'predicted_age': 26.0, 'gt_age': 20,
             'predicted_gender': 0.1, 'gt_gender': 1},
        ]

        mae, age_n, recalls = analyze_results.age_gender_diagnostics(rows, 'male')

        self.assertEqual(mae, 4.0)
        self.assertEqual(age_n, 2)
        self.assertEqual(recalls, {'male': (1, 1), 'female': (1, 1)})


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from src.model_evaluation import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    """
    Test suite for the evaluate_model function.
    """
    @classmethod
    def setUpClass(cls):
        """Setup reusable assets for all tests."""
        # Create test data
        cls.y_val = np.array([0, 1, 1, 2, 2, 0, 1, 3, 4, 5, 6])
        cls.val_predictions = np.array([0, 1, 0, 2, 2, 0, 1, 3, 4, 5, 6])
        cls.evaluation_results_path = Path("test_output/evaluation_results.txt")

        # Create the output directory if it doesn't exist
        cls.evaluation_results_path.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up after each test."""
        if self.evaluation_results_path.exists():
            self.evaluation_results_path.unlink()
        confusion_matrix_path = self.evaluation_results_path.parent / "confusion_matrix.png"
        if confusion_matrix_path.exists():
            confusion_matrix_path.unlink()
        if self.evaluation_results_path.parent.exists():
            self.evaluation_results_path.parent.rmdir()

    @patch("src.model_evaluation.plt.savefig")
    def test_evaluate_model_happy_path(self, mock_savefig):
        """Test that evaluate_model function works correctly."""
        accuracy, class_report, conf_matrix = evaluate_model(
            self.y_val, self.val_predictions, self.evaluation_results_path
        )
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(class_report)
        self.assertIsNotNone(conf_matrix)

        # Check if the results are saved correctly
        self.assertTrue(self.evaluation_results_path.exists())
        mock_savefig.assert_called_once()

    def test_evaluate_model_with_empty_data(self):
        """Test that evaluate_model function handles empty test data."""
        empty_y_val = np.array([])
        empty_val_predictions = np.array([])
        accuracy, class_report, conf_matrix = evaluate_model(
            empty_y_val, empty_val_predictions, self.evaluation_results_path
        )
        self.assertIsNone(accuracy)
        self.assertIsNone(class_report)
        self.assertIsNone(conf_matrix)

    @patch("src.model_evaluation.logger.error")
    def test_evaluate_model_value_error(self, mock):
        """Test that evaluate_model function handles ValueError during evaluation."""
        with patch("src.model_evaluation.accuracy_score", side_effect=ValueError("test error")):
            accuracy, class_report, conf_matrix = evaluate_model(
                self.y_val, self.val_predictions, self.evaluation_results_path
            )
            self.assertIsNone(accuracy)
            self.assertIsNone(class_report)
            self.assertIsNone(conf_matrix)
            self.assertRaises(ValueError)

if __name__ == "__main__":
    unittest.main()

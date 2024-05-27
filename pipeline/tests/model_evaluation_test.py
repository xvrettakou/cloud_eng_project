import unittest
from unittest.mock import MagicMock
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.model_evaluation import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    """
    Test suite for the evaluate_model function.

    This class contains various unit tests to ensure the correct execution
    and error handling of the evaluate_model function. The tests cover:
    1. Correct functionality with valid inputs (happy path).
    2. Handling of missing required features in the test data.
    3. Handling of file system errors when saving results.
    4. Handling of empty test data.
    """
    @classmethod
    def setUpClass(cls):
        """Setup reusable assets for all tests."""
        cls.save_path = Path("test_output")

        # Create a mock model with predict and predict_proba methods
        cls.mock_model = MagicMock(spec=RandomForestClassifier)
        cls.mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
        cls.mock_model.predict.return_value = np.array([0, 1, 1])

        # Create test data
        cls.x_test = pd.DataFrame({
            "log_entropy": [0.5, 0.6, 0.7],
            "IR_norm_range": [0.2, 0.3, 0.4],
            "entropy_x_contrast": [0.1, 0.2, 0.3]
        })
        cls.y_test = pd.Series([0, 1, 1])

    def tearDown(self):
        """Clean up after each test."""
        if self.save_path.exists() and self.save_path.is_dir():
            for file in self.save_path.iterdir():
                file.unlink()
            self.save_path.rmdir()

    def test_evaluate_model_happy_path(self):
        """Test that evaluate_model function works correctly."""
        auc, accuracy, confusion, classification_rep = evaluate_model(
            self.mock_model, self.x_test, self.y_test, self.save_path
        )
        self.assertIsNotNone(auc)
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(confusion)
        self.assertIsNotNone(classification_rep)

        # Check if the results are saved correctly
        self.assertTrue((self.save_path / "evaluation_results.txt").exists())

    def test_evaluate_model_missing_features(self):
        """Test that evaluate_model function handles missing initial features."""
        x_test_missing_features = self.x_test.drop(columns=["log_entropy"])
        auc, accuracy, confusion, classification_rep = evaluate_model(
            self.mock_model, x_test_missing_features, self.y_test, self.save_path
        )
        self.assertIsNone(auc)
        self.assertIsNone(accuracy)
        self.assertIsNone(confusion)
        self.assertIsNone(classification_rep)

    def test_evaluate_model_file_not_found(self):
        """Test that evaluate_model function handles file not found errors."""
        invalid_save_path = Path("/invalid_path/test_output")
        auc, accuracy, confusion, classification_rep = evaluate_model(
            self.mock_model, self.x_test, self.y_test, invalid_save_path
        )
        self.assertIsNone(auc)
        self.assertIsNone(accuracy)
        self.assertIsNone(confusion)
        self.assertIsNone(classification_rep)

    def test_evaluate_model_with_empty_data(self):
        """Test that evaluate_model function handles empty test data."""
        x_test_empty = pd.DataFrame(columns=["log_entropy", "IR_norm_range", "entropy_x_contrast"])
        y_test_empty = pd.Series(dtype=int)
        auc, accuracy, confusion, classification_rep = evaluate_model(
            self.mock_model, x_test_empty, y_test_empty, self.save_path
        )
        self.assertIsNone(auc)
        self.assertIsNone(accuracy)
        self.assertIsNone(confusion)
        self.assertIsNone(classification_rep)

if __name__ == "__main__":
    unittest.main()

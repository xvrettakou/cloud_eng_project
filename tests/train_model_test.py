import unittest
import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification
from src.train_model import split_data, train_model

# Append the parent directory containing 'src' to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class TestTrainModel(unittest.TestCase):
    """
    Test suite for split_data and train_model functions.
    This class contains various unit tests that ensure the correct execution
    and error handling of these functions.
    """

    @classmethod
    def setUpClass(cls):
        """Setup reusable assets for all tests."""
        cls.x_data, cls.y_data = make_classification(n_samples=100, n_features=20, random_state=42)
        cls.x_data = pd.DataFrame(cls.x_data, columns=[f"feature_{i}" for i in range(20)])
        cls.save_path = Path("test_output")

    def tearDown(self):
        """Clean up after each test."""
        if self.save_path.exists() and self.save_path.is_dir():
            for file in self.save_path.iterdir():
                file.unlink()
            self.save_path.rmdir()

    def test_split_data_happy_path(self):
        """Test that split_data function correctly splits data."""
        x_train, x_val, y_train, y_val = split_data(self.x_data, self.y_data)
        self.assertIsNotNone(x_train)
        self.assertIsNotNone(x_val)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_val)
        self.assertEqual(len(x_train) + len(x_val), 100)

    def test_split_data_unhappy_path(self):
        """Test that split_data function handles invalid test_size."""
        x_train, x_val, y_train, y_val = split_data(self.x_data, self.y_data, test_size=1.5)
        self.assertIsNone(x_train)
        self.assertIsNone(x_val)
        self.assertIsNone(y_train)
        self.assertIsNone(y_val)

    def test_train_model_happy_path(self):
        """Test that train_model function trains and saves model correctly."""
        x_train, x_val, y_train, y_val = split_data(self.x_data, self.y_data)
        model = train_model(x_train, y_train, x_val, y_val, self.save_path)
        self.assertIsNotNone(model)
        self.assertTrue((self.save_path / "trained_model.pkl").exists())

    def test_train_model_unhappy_path(self):
        """Test that train_model function handles errors gracefully."""
        x_train, x_val, y_train, y_val = split_data(self.x_data, self.y_data)
        invalid_save_path = Path("/invalid_path/test_output")
        model = train_model(x_train, y_train, x_val, y_val, invalid_save_path)
        self.assertIsNone(model)

if __name__ == "__main__":
    unittest.main()

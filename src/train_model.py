import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_data(x_data, y_data, test_size=0.2, random_state=42, stratify=True):
    """Split the data into training and validation sets."""
    try:
        stratify_param = y_data if stratify else None
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        logger.info("Data splitting completed successfully.")
        return x_train, x_val, y_train, y_val
    except ValueError as value_error:
        logger.error("ValueError occurred during data splitting: %s", value_error)
        return None, None, None, None

def train_model(x_train, y_train, x_val, y_val, save_path):
    """Train a Random Forest classifier and save the model."""
    try:
        # Flatten the images for the classifier
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)

        # Train a Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x_train_flat, y_train)
        logger.info("Model training completed successfully.")

        # Optionally evaluate the model on validation data
        val_score = clf.score(x_val_flat, y_val)
        logger.info("Validation score: %s", val_score)

        # Save the trained model
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model_path = Path(save_path) / "trained_model.pkl"
        joblib.dump(clf, model_path)
        logger.info("Model saved to %s", model_path)

        return clf

    except ValueError as value_error:
        logger.error("ValueError occurred during model training: %s", value_error)
        return None

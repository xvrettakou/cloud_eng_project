import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, x_test, y_test, save_path):
    """Evaluate the trained model's performance and save the results."""
    try:
        initial_features = ["log_entropy", "IR_norm_range", "entropy_x_contrast"]

        # Ensure that the test data contains the required initial features
        if not set(initial_features).issubset(x_test.columns):
            raise ValueError("Test data does not contain all required initial features")

        y_pred_proba = model.predict_proba(x_test[initial_features])[:, 1]
        y_pred = model.predict(x_test[initial_features])

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        output_path = Path(save_path)
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Save the results
        with open(output_path / "evaluation_results.txt", "w") as file:
            file.write(f"AUC: {auc}\n")
            file.write(f"Accuracy: {accuracy}\n")
            file.write(f"Confusion Matrix:\n{confusion}\n")
            file.write(f"Classification Report:\n{classification_rep}\n")

        logger.info("Model evaluation completed successfully and results saved.")
        return auc, accuracy, confusion, classification_rep

    except ValueError as e:
        logger.error("Value error during model evaluation: %s", e)
        return None, None, None, None
    except OSError as e:
        logger.error("OS error during model evaluation: %s", e)
        return None, None, None, None

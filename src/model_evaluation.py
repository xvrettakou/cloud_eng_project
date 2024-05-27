import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_val, val_predictions, evaluation_results_path):
    """Evaluate the trained model's performance and save the results."""
    try:
        # Calculate performance metrics
        accuracy = accuracy_score(y_val, val_predictions)
        class_report = classification_report(y_val, val_predictions, target_names=[
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ])
        conf_matrix = confusion_matrix(y_val, val_predictions)

        # Print performance metrics
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + class_report)

        # Save evaluation results to the specified path
        with open(evaluation_results_path, "w") as file:
            file.write(f"Validation Accuracy: {accuracy:.4f}\n")
            file.write("Classification Report:\n" + class_report + "\n")
            file.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ], yticklabels=[
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(evaluation_results_path.parent / "confusion_matrix.png")
        plt.close()

        logger.info("Model evaluation completed successfully.")
        return accuracy, class_report, conf_matrix

    except ValueError as e:
        logger.error("Value error during model evaluation: %s", e)
        return None, None, None
    except Exception as e:
        logger.error("An error occurred during model evaluation: %s", e)
        return None, None, None

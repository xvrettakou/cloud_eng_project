import logging
import pandas as pd

# Set up logging configuration
logger = logging.getLogger(__name__)

def score_model(model, x_val_flat, output_dir):
    """Score the trained model on the validation set and save predictions to a CSV file."""
    try:
        # Predict on the validation set
        val_predictions = model.predict(x_val_flat)
        logger.info("Model scoring completed successfully.")

        # Convert predictions to a pandas DataFrame
        model_scoring = pd.DataFrame({"predictions": val_predictions})

        # Save the model scoring to a CSV file
        model_scoring.to_csv(output_dir / "model_scoring.csv", index=False)
        logger.info("Model scoring artifacts saved to %s", output_dir / "model_scoring.csv")

        return model_scoring
    except FileNotFoundError as e:
        logger.error("Output directory not found: %s", e)
        return None

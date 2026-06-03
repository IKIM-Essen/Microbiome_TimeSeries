import logging
import os
import numpy as np
import keras

from src.model_building.create_models import ensemble_predict
from src.preprocessing.scaling import inverse_scale_data
from src.utils.config import reshape

# Setup logging for model-building and training operations
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
LOG_DIR = os.path.join(ROOT_DIR, "logs", "predicting")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "predicting.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)

def predict(X_train, X_val, X_test, tcn_path, lstm_path, scaler_path=None, output_path=None):
    tcn = keras.models.load_model(tcn_path)
    lstm = keras.models.load_model(lstm_path)
    pred_train = ensemble_predict(tcn, lstm, X_train)
    pred_val = ensemble_predict(tcn, lstm, X_val)
    pred_test = ensemble_predict(tcn, lstm, X_test)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez_compressed(
            output_path,
            pred_train=pred_train,
            pred_val=pred_val,
            pred_test=pred_test,
        )
        logger.info("Saved predictions to %s", output_path)

    return pred_train, pred_val, pred_test

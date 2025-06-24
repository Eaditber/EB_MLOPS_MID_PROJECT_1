import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from config.paths_config import MODEL_OUTPUT_PATH, TRAIN_PATH
from src.logger import get_logger
from scipy import stats
import threading # Ensure this is at the top or at least before its first use

# Prometheus imports
from prometheus_client import start_http_server, Counter, Gauge

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")

# Initialize Prometheus counters
prediction_count = Counter('prediction_count', "Number of prediction count")
drift_count = Counter('drift_count', "Number of times data drift is detected")

# --- Model Loading ---
loaded_model = None
try:
    with open(MODEL_OUTPUT_PATH, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    logger.info(f"Model loaded successfully from {MODEL_OUTPUT_PATH}")
except FileNotFoundError:
    logger.error(f"ERROR: Model file not found at {MODEL_OUTPUT_PATH}. Please check the path.", exc_info=True)
except Exception as e:
    logger.error(f"ERROR: An error occurred while loading the model: {e}", exc_info=True)

FEATURE_NAMES = ['TotalCharges', 'Month-to-month', 'One year', 'Two year',
                 'PhoneService', 'tenure']

# --- Define Reference Data for Drift Detection ---
try:
    reference_data_df = pd.read_csv(TRAIN_PATH)
    reference_data_df['Month-to-month'] = reference_data_df['Month-to-month'].map({'Yes': 1, 'No': 0})
    reference_data_df['One year'] = reference_data_df['One year'].map({'Yes': 1, 'No': 0})
    reference_data_df['Two year'] = reference_data_df['Two year'].map({'Yes': 1, 'No': 0})
    reference_data_df['PhoneService'] = reference_data_df['PhoneService'].map({'Yes': 1, 'No': 0})
    logger.info("Reference data for drift detection loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load reference data for drift detection: {e}", exc_info=True)
    reference_data_df = None

# Define thresholds for drift detection
KS_PVALUE_THRESHOLD = 0.05
CHI2_PVALUE_THRESHOLD = 0.05

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_display_text = "No predictions yet made..."

    if request.method == 'POST':
        if loaded_model is None:
            prediction_display_text = "Error: Prediction model not loaded. Please check server logs."
            return render_template('index.html', prediction_text=prediction_display_text)

        try:
            data = request.form
            TotalCharges = float(data["TotalCharges"])
            Month_to_month_val = 1 if data['Month-to-month'] == 'Yes' else 0
            One_year_val = 1 if data['One year'] == 'Yes' else 0
            Two_year_val = 1 if data['Two year'] == 'Yes' else 0
            PhoneService_val = 1 if data['PhoneService'] == 'Yes' else 0
            tenure = float(data["tenure"])

            # --- Contract Type Validation ---
            contract_sum = Month_to_month_val + One_year_val + Two_year_val
            if contract_sum > 1:
                prediction_display_text = "Error: Only one contract type (Month-to-month, One year, Two year) can be 'Yes'."
                logger.warning(prediction_display_text)
                return render_template('index.html', prediction_text=prediction_display_text)
            elif contract_sum == 0:
                prediction_display_text = "Error: At least one contract type (Month-to-month, One year, Two year) must be 'Yes'."
                logger.warning(prediction_display_text)
                return render_template('index.html', prediction_text=prediction_display_text)

            # Create current features DataFrame for prediction and drift detection
            current_features_df = pd.DataFrame([[TotalCharges, Month_to_month_val, One_year_val, Two_year_val, PhoneService_val, tenure]],
                                             columns=FEATURE_NAMES)

            # --- Data Drift Detection (using scipy.stats) ---
            if reference_data_df is not None:
                drift_detected_overall = False
                drift_details = []

                # Numerical features: TotalCharges, tenure
                numerical_features = ['TotalCharges', 'tenure']
                for feature in numerical_features:
                    stat, p_value = stats.ks_2samp(reference_data_df[feature], current_features_df[feature])
                    if p_value < KS_PVALUE_THRESHOLD:
                        drift_detected_overall = True
                        drift_details.append(f"Drift in {feature} (KS p={p_value:.4f})")
                        logger.warning(f"Drift detected for {feature}: p={p_value:.4f}")

                # Categorical features: Month-to-month, One year, Two year, PhoneService
                categorical_features = ['Month-to-month', 'One year', 'Two year', 'PhoneService']
                for feature in categorical_features:
                    ref_counts = reference_data_df[feature].value_counts()
                    current_counts = current_features_df[feature].value_counts()

                    ref_unique_cats = reference_data_df[feature].unique()
                    current_unique_cats = current_features_df[feature].unique()

                    all_unique_cats = np.unique(np.concatenate([ref_unique_cats, current_unique_cats]))

                    observed_ref = [ref_counts.get(cat, 0) for cat in all_unique_cats]
                    observed_current = [current_counts.get(cat, 0) for cat in all_unique_cats]

                    if np.sum(observed_ref) > 0 and np.sum(observed_current) > 0 and len(all_unique_cats) > 1 :
                        contingency_table = np.array([observed_ref, observed_current])

                        if np.all(np.sum(contingency_table, axis=0) > 0) and np.all(np.sum(contingency_table, axis=1) > 0):
                            try:
                                chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
                                if p_value < CHI2_PVALUE_THRESHOLD:
                                    drift_detected_overall = True
                                    drift_details.append(f"Drift in {feature} (Chi2 p={p_value:.4f})")
                                    logger.warning(f"Drift detected for {feature}: p={p_value:.4f}")
                            except ValueError as ve:
                                logger.warning(f"Chi-squared test failed for {feature} (data too sparse/trivial): {ve}")
                        else:
                            logger.warning(f"Skipping Chi-squared test for {feature} due to zero sums in contingency table.")
                    else:
                         logger.warning(f"Skipping Chi-squared test for {feature} due to insufficient data for comparison.")

                if drift_detected_overall:
                    drift_count.inc()
                    logger.info("Overall Data Drift Detected! Details: " + "; ".join(drift_details))
                else:
                    logger.info("No Data Drift Detected.")
            else:
                logger.warning("Reference data not available. Skipping data drift detection.")

            # --- Model Prediction ---
            prediction_raw = loaded_model.predict(current_features_df)
            prediction_count.inc()

            result = 'Churn' if prediction_raw[0] == 1 else 'Not Churn'
            prediction_display_text = f"The prediction is: {result}"

        except KeyError as e:
            prediction_display_text = f"Error: Missing form field - {e}. Please ensure all fields are filled."
            logger.error(f"KeyError in POST request: {e}", exc_info=True)
        except ValueError as e:
            prediction_display_text = f"Error: Invalid input value - {e}. Please enter valid numbers."
            logger.error(f"ValueError in POST request: {e}", exc_info=True)
        except Exception as e:
            prediction_display_text = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error in POST request: {e}", exc_info=True)

    return render_template('index.html', prediction_text=prediction_display_text)

@app.route('/metrics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response
    return Response(generate_latest(), content_type='text/plain')

if __name__ == "__main__":
    # Start Prometheus http server in a separate thread BEFORE app.run()
    threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()
    logger.info("Prometheus metrics server started on port 8000") # Add a log to confirm

    # Now run the Flask app
    app.run(host='0.0.0.0' , port=5000, debug=True)
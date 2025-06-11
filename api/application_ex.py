import joblib
import numpy as np
import pandas as pd # Import pandas for DataFrame creation
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template,request, jsonify # Import jsonify for error responses

# These imports are from your other provided application.py files, assuming they are used
# in your full project, even if not fully active in the current snippet.
# If these are not used, you can comment them out.
# from src.logger import get_logger
# from alibi_detect.cd import KSDrift
# from src.feature_store import RedisFeatureStore
# from sklearn.preprocessing import StandardScaler
# from prometheus_client import start_http_server, Counter, Gauge
# import threading # For running Prometheus server in a separate thread

# Assuming get_logger is defined elsewhere or will be imported
# If not, you can use Python's built-in logging or just print
# logger = get_logger(__name__)

app = Flask(__name__ , template_folder="templates")

# Initialize Prometheus counters (if you uncomment the imports)
# prediction_count = Counter('prediction_count' , " Number of prediction count" )
# drift_count = Counter('drift_count' , "Numer of times data drift is detected")

# --- Model Loading (Add error handling for robustness) ---
loaded_model = None # Initialize to None
try:
    with open(MODEL_OUTPUT_PATH , 'rb') as model_file:
        loaded_model = joblib.load(model_file)
    # logger.info(f"Model loaded successfully from {MODEL_OUTPUT_PATH}") # Use logger if active
    print(f"Model loaded successfully from {MODEL_OUTPUT_PATH}")
except FileNotFoundError:
    # logger.error(f"ERROR: Model file not found at {MODEL_OUTPUT_PATH}. Please check the path.") # Use logger if active
    print(f"ERROR: Model file not found at {MODEL_OUTPUT_PATH}. Please check the path.")
    # loaded_model remains None
except Exception as e:
    # logger.error(f"ERROR: An error occurred while loading the model: {e}") # Use logger if active
    print(f"ERROR: An error occurred while loading the model: {e}")
    # loaded_model remains None


# Define FEATURE_NAMES (if not already defined globally or imported)
FEATURE_NAMES =['TotalCharges', 'Month-to-month', 'One year', 'Two year',
       'PhoneService', 'tenure']

# --- Scaler and KSDrift Initialization (Assuming these are part of your full setup) ---
# If you plan to use these, ensure Redis is running and you have a proper setup
# for historical data. Otherwise, leave them commented or remove them.
# feature_store = RedisFeatureStore()
# scaler = StandardScaler()
# ksd = None # Initialize ksd to None

# def fit_scaler_on_ref_data():
#     # ... (Your fit_scaler_on_ref_data implementation here) ...
#     pass

# try:
#     historical_data = fit_scaler_on_ref_data()
#     if historical_data is not None and historical_data.size > 0:
#         # from alibi_detect.cd import KSDrift # Uncomment if KSDrift is used
#         ksd = KSDrift(x_ref=historical_data , p_val=0.05)
#         # logger.info("Drift detector (KSDrift) initialized successfully.")
#         print("Drift detector (KSDrift) initialized successfully.")
#     else:
#         # logger.warning("Skipping KSDrift initialization due to missing or empty historical data.")
#         print("Skipping KSDrift initialization due to missing or empty historical data.")
# except Exception as e:
#     # logger.error(f"Failed to initialize KSDrift: {e}", exc_info=True)
#     print(f"Failed to initialize KSDrift: {e}")
#     ksd = None


@app.route('/',methods=['GET','POST'])
def index():
    prediction_display_text = "No predictions yet made..." # Default text for GET request or initial state

    if request.method=='POST':
        if loaded_model is None: # Check if model loaded successfully
            prediction_display_text = "Error: Prediction model not loaded. Please check server logs."
            return render_template('index.html', prediction_text=prediction_display_text)

        try:
            data = request.form
            TotalCharges = float(data["TotalCharges"])
            # Ensure these map operations are robust for single values
            Month_to_month_val = 1 if data['Month-to-month'] == 'Yes' else 0
            One_year_val = 1 if data['One year'] == 'Yes' else 0
            Two_year_val = 1 if data['Two year'] == 'Yes' else 0
            PhoneService_val = 1 if data['PhoneService'] == 'Yes' else 0
            tenure= float(data["tenure"])

            # --- NEW VALIDATION LOGIC FOR CONTRACT TYPES ---
            contract_sum = Month_to_month_val + One_year_val + Two_year_val
            if contract_sum > 1:
                prediction_display_text = "Error: Only one contract type (Month-to-month, One year, Two year) can be 'Yes'."
                # logger.warning(prediction_display_text) # Use logger if active
                print(prediction_display_text)
                return render_template('index.html', prediction_text=prediction_display_text)
            elif contract_sum == 0:
                prediction_display_text = "Error: At least one contract type (Month-to-month, One year, Two year) must be 'Yes'."
                # logger.warning(prediction_display_text) # Use logger if active
                print(prediction_display_text)
                return render_template('index.html', prediction_text=prediction_display_text)
            # --- END NEW VALIDATION LOGIC ---

            # Create DataFrame with the processed values
            # Using pd.DataFrame for consistency with potential scaler/KSDrift later
            features = pd.DataFrame([[TotalCharges, Month_to_month_val, One_year_val, Two_year_val, PhoneService_val, tenure]],
                                     columns=FEATURE_NAMES)

            # --- Data Drift Detection (Uncomment and implement if active) ---
            # if ksd:
            #     features_scaled = scaler.transform(features)
            #     drift = ksd.predict(features_scaled)
            #     drift_response = drift.get('data',{})
            #     is_drift = drift_response.get('is_drift' , None)
            #     if is_drift is not None and is_drift==1:
            #         # logger.info("Drift Detected....") # Use logger if active
            #         print("Drift Detected....")
            #         # drift_count.inc() # Uncomment if prometheus is active
            # else:
            #     # logger.warning("Drift detector (KSDrift) not initialized. Skipping drift detection.")
            #     print("Drift detector (KSDrift) not initialized. Skipping drift detection.")

            # --- Model Prediction ---
            prediction_raw = loaded_model.predict(features)
            # prediction_count.inc() # Uncomment if prometheus is active
            
            # Format the prediction result into the text you want to display
            result = 'Churn' if prediction_raw[0] == 1 else 'Not Churn'
            prediction_display_text = f"The prediction is: {result}"

        except KeyError as e:
            prediction_display_text = f"Error: Missing form field - {e}. Please ensure all fields are filled."
            print(f"KeyError in POST request: {e}") # Log for debugging
        except ValueError as e:
            prediction_display_text = f"Error: Invalid input value - {e}. Please enter valid numbers."
            print(f"ValueError in POST request: {e}") # Log for debugging
        except Exception as e:
            prediction_display_text = f"An unexpected error occurred: {str(e)}"
            print(f"Unexpected error in POST request: {e}") # Log for debugging

    # Always render the template with the 'prediction_text' variable
    return render_template('index.html' , prediction_text = prediction_display_text)

# --- Prometheus Metrics Endpoint (Uncomment if active) ---
# @app.route('/metrics')
# def metrics():
#     # from prometheus_client import generate_latest # Moved inside function for conditional import
#     # from flask import Response # Moved inside function for conditional import
#     # return Response(generate_latest() , content_type='text/plain')
#     pass # Placeholder if commented out


if __name__=="__main__":
    # --- Start Prometheus http server in a separate thread (Uncomment if active) ---
    # Ensure threading is imported: import threading
    # import threading
    # threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()
    
    # Start Flask app
    app.run(host='0.0.0.0' , port=5000, debug=True) # Set debug=True for development to see traceback
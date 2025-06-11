# app.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from config.paths_config import MODEL_OUTPUT_PATH # Assuming you have this config file

app = Flask(__name__)

# --- Model Loading ---
loaded_model = None
try:
    with open(MODEL_OUTPUT_PATH, 'rb') as model_file:
        loaded_model = joblib.load(model_file)
    print(f"API: Model loaded successfully from {MODEL_OUTPUT_PATH}")
except FileNotFoundError:
    print(f"API ERROR: Model file not found at {MODEL_OUTPUT_PATH}. Ensure it exists.")
    # Exit if model is critical and not found
    exit(1) # Or raise an exception if you prefer Flask to handle it later
except Exception as e:
    print(f"API ERROR: An error occurred while loading the model: {e}")
    exit(1) # Or raise an exception

# Define feature names as expected by your model
FEATURE_NAMES = ['TotalCharges', 'Month-to-month', 'One year', 'Two year',
                 'PhoneService', 'tenure']

@app.route('/', methods=['POST'])
def predict():
    # Ensure Content-Type is application/json
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json(force=True) # Get JSON data from request body

        # --- Input Validation ---
        required_params = FEATURE_NAMES
        for param in required_params:
            if param not in data:
                return jsonify({"error": f"Missing required parameter: {param}"}), 400

        # Extract and convert data
        TotalCharges = float(data["TotalCharges"])
        Month_to_month_val = 1 if data['Month-to-month'].lower() == 'yes' else 0
        One_year_val = 1 if data['One year'].lower() == 'yes' else 0
        Two_year_val = 1 if data['Two year'].lower() == 'yes' else 0
        PhoneService_val = 1 if data['PhoneService'].lower() == 'yes' else 0
        tenure = float(data["tenure"])

        # --- Contract Type Validation (Exactly one 'Yes') ---
        contract_sum = Month_to_month_val + One_year_val + Two_year_val
        if contract_sum > 1:
            return jsonify({"error": "Only one contract type (Month-to-month, One year, Two year) can be 'Yes'."}), 400
        elif contract_sum == 0:
            return jsonify({"error": "At least one contract type (Month-to-month, One year, Two year) must be 'Yes'."}), 400

        # Create numpy array of features in the correct order
        features = np.array([[TotalCharges, Month_to_month_val, One_year_val, Two_year_val, PhoneService_val, tenure]])

        # --- Model Prediction ---
        prediction_raw = loaded_model.predict(features)
        
        # Format the prediction result
        result_text = 'Churn' if prediction_raw[0] == 1 else 'Not Churn'

        return jsonify({"prediction": result_text})

    except ValueError as e:
        return jsonify({"error": f"Invalid data format or value: {e}. Check numeric fields and boolean 'Yes'/'No' strings."}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing expected key in JSON data: {e}."}), 400
    except Exception as e:
        # Log the full error for debugging in production
        app.logger.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running", "model_loaded": loaded_model is not None}), 200

if __name__ == '__main__':
    # Using Flask's built-in server for development.
    # For production, use a WSGI server like Gunicorn (gunicorn app:app)
    app.run(host='0.0.0.0', port=5000, debug=True)
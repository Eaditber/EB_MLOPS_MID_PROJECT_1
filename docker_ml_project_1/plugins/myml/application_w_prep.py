import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template,request
from data_processing import DataProcessing
import pandas as pd

app = Flask(__name__)

# --- Model Loading (Add error handling for robustness) ---
try:
    loaded_model = joblib.load(MODEL_OUTPUT_PATH)
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_OUTPUT_PATH}. Please check the path.")
    loaded_model = None # Set to None to indicate model loading failure
except Exception as e:
    print(f"ERROR: An error occurred while loading the model: {e}")
    loaded_model = None

@app.route('/',methods=['GET','POST'])
def index():
    prediction_display_text = "No predictions yet made..." # Default text for GET request or initial state

    if request.method=='POST':
        if loaded_model is None: # Check if model loaded successfully
            prediction_display_text = "Error: Prediction model not loaded. Please check server logs."
            return render_template('index.html', prediction_text=prediction_display_text)

        try:
            data = request.form
            df_input = pd.DataFrame([dict(data)])
            data_processor = DataProcessing(df_input)  # preprocess data
            df_processed = data_processor.preprocess_data()
            for col in ['Month-to-month', 'One year', 'Two year']:
                if col not in df_processed.columns:
                   df_processed[col] = 0
            features=['TotalCharges','Month-to-month','One year','Two year','PhoneService','tenure']
            prediction_raw= loaded_model.predict(df_processed[features])
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

if __name__=="__main__":
    app.run(host='0.0.0.0' , port=5000, debug=True) # Set debug=True for development to see traceback
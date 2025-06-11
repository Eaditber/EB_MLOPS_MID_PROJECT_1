import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, render_template,request

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
            TotalCharges = float(data["TotalCharges"])
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
            elif contract_sum == 1:
                features = np.array([[TotalCharges, Month_to_month_val, One_year_val, Two_year_val, PhoneService_val, tenure]])

                prediction_raw = loaded_model.predict(features)
                
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
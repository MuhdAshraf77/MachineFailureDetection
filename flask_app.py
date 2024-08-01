import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import openai
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your model and scaler from the saved files
model = load_model('Development/best_cnn_model_gans.h5')
scaler = joblib.load('Development/scaler.pkl')

# Get the feature names from the scaler
feature_names = scaler.feature_names_in_

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set this environment variable

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bulk_upload')
def bulk_upload():
    return render_template('bulk_upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return "No file part in the request"
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected for uploading")
        return "No selected file"
    if file and file.filename.endswith('.csv'):
        logging.info(f"File {file.filename} received for processing")
        try:
            data = pd.read_csv(file)
            logging.info("File successfully read into a DataFrame")
            # Check if required columns are in the uploaded file
            missing_features = [feature for feature in feature_names if feature not in data.columns]
            if missing_features:
                logging.error(f"Missing features in the uploaded file: {missing_features}")
                return f"Missing features in the uploaded file: {missing_features}"
            
            # Process the file and make predictions
            scaled_data = scaler.transform(data[feature_names])
            scaled_data = np.expand_dims(scaled_data, axis=2)  # Reshape for CNN
            threshold = 0.99627864  # Lower threshold to increase sensitivity
            predictions = (model.predict(scaled_data)[:, 0] > threshold).astype(int)
            failure_count = int(np.sum(predictions))
            non_failure_count = len(predictions) - failure_count
            logging.info(f"Predictions made. Failure count: {failure_count}, Non-failure count: {non_failure_count}")
            
            # Add predictions to the data
            data['Prediction'] = np.where(predictions == 1, 'Failure', 'Non-Failure')
            
            # Convert DataFrame to HTML
            data_html = data.to_html(classes='data', header="true", table_id="example")
            
            return render_template('bulk_upload.html', failure_count=failure_count, non_failure_count=non_failure_count, data_html=data_html)
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return f"Error processing file: {str(e)}"
    logging.error("Invalid file format. Only CSV files are accepted.")
    return "Invalid file format. Only CSV files are accepted."

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    # Create a dictionary with all features, filling missing ones with 0
    features = {name: float(form_data.get(name, 0)) for name in feature_names}

    # Create DataFrame ensuring the order matches the scaler's feature names
    features_df = pd.DataFrame([features], columns=feature_names)

    # Ensure the data is scaled correctly
    scaled_features = scaler.transform(features_df)
    scaled_features = np.expand_dims(scaled_features, axis=2)  # Reshape for CNN
    threshold = 0.99627864  # Lower threshold to increase sensitivity
    predictions = (model.predict(scaled_features)[:, 0] > threshold).astype(int)

    # Determine the prediction result
    prediction_result = "Machine is likely to fail." if predictions[0] == 1 else "Machine is not likely to fail."

    insights = analyze_predictions(prediction_result, features)
    return render_template('index.html', prediction=insights)

def analyze_predictions(prediction_result, features):
    # Determine the scenario based on the prediction result
    if "Machine is likely to fail." in prediction_result.lower():
        scenario = "failure"
    else:
        scenario = "no_failure"

    # Create a structured prompt for OpenAI API based on the scenario
    if scenario == "failure":
        system_message = "You are an AI assistant providing recommendations for a machine that is likely to fail."
        prompt = (
            "The machine is predicted to fail. Provide recommendations for urgent maintenance, critical sensor analysis, failure prevention, operational impact, and safety precautions."
            "critical sensor analysis, failure prevention, operational impact, and safety precautions. "
            "Do not use numbered lists in your response."
        )
    else:
        system_message = "You are an AI assistant providing recommendations for a machine that is not likely to fail."
        prompt = (
            "The machine is predicted to not fail. Provide recommendations for preventive maintenance, performance optimization, longevity strategies, monitoring, and efficiency improvements."
            "performance optimization, longevity strategies, monitoring, and efficiency improvements. "
            "Do not use numbered lists in your response."
        )

    prompt += f"\n\nMachine data:\n"
    for key, value in features.items():
        prompt += f"{key}: {value}\n"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        insights = response.choices[0].message['content'].strip().replace("_", "")  # Remove underscores
    except Exception as e:
        insights = f"Error generating insights: {str(e)}"

    # Format the final output to include the prediction result and detailed insights
    final_output = (
        f"{prediction_result}\n\n"
        f"{insights}"
    )
    
    return final_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

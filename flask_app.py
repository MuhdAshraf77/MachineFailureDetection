import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import openai
import os

app = Flask(__name__)

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
    # Create a structured prompt for OpenAI API
    prompt = (
        f"{prediction_result}\n\n"
        "The following features were used to make this prediction:\n"
        f"S5: {features['S5']}\n"
        f"S8: {features['S8']}\n"
        f"S13: {features['S13']}\n"
        f"S15: {features['S15']}\n"
        f"S16: {features['S16']}\n"
        f"S18: {features['S18']}\n"
        f"S19: {features['S19']}\n"
        f"Machine Age: {features['MACHINE_AGE']}\n"
        f"Well Group: {features['WELL_GROUP']}\n\n"
        "Based on the above data, provide a detailed analysis and maintenance recommendations including:\n"
        "1. Age of the Machine: Discuss the impact of the machine's age on its performance and maintenance needs.\n"
        "2. Sensor Readings: Analyze the provided sensor readings and explain what they might indicate about the machine's condition.\n"
        "3. Well Group: Explain how the well group classification can affect the machine's maintenance strategy.\n"
        "4. Machine Failure Prediction: Interpret the prediction result and suggest any immediate actions if necessary.\n"
        "5. Maintenance Recommendations: Offer actionable steps to maintain the machine based on the provided data.\n"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        insights = response.choices[0].message['content'].strip()
    except Exception as e:
        insights = f"Error generating insights: {str(e)}"

    # Format the final output to include the prediction result and detailed insights
    final_output = f"### Prediction Result:\n{prediction_result}\n\n### Detailed Insights and Maintenance Recommendations:\n{insights}"
    return final_output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
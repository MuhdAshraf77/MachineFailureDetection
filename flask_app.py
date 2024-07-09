import openai
import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Create the uploads directory if it doesn't exist
os.makedirs('uploads', exist_ok=True)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'

openai.api_key = os.getenv('OPENAI_API_KEY')

# Load your model and scaler from the saved files
model = load_model('Development/best_cnn_model_gans.h5')
scaler = joblib.load('Development/scaler.pkl')

def analyze_predictions(predictions, features):
    insights = []
    for prediction, feature_set in zip(predictions, features):
        prompt = (
            "The following prediction has been made for machine failure:\n"
            f"{prediction}\n\n"
            "Based on the following features for the machine:\n"
            f"Machine ID: {feature_set.get('ID', 'N/A')}\n"
            f"Date: {feature_set.get('DATE', 'N/A')}\n"
            f"Region Cluster: {feature_set.get('REGION_CLUSTER', 'N/A')}\n"
            f"Maintenance Vendor: {feature_set.get('MAINTENANCE_VENDOR', 'N/A')}\n"
            f"Manufacturer: {feature_set.get('MANUFACTURER', 'N/A')}\n"
            f"Well Group: {feature_set['WELL_GROUP']}\n"
            f"S5: {feature_set['S5']}\n"
            f"S13: {feature_set.get('S13', 'N/A')}\n"
            f"S15: {feature_set['S15']}\n"
            f"S16: {feature_set['S16']}\n"
            f"S17: {feature_set.get('S17', 'N/A')}\n"
            f"S18: {feature_set['S18']}\n"
            f"S19: {feature_set['S19']}\n"
            f"Age of Equipment: {feature_set['MACHINE_AGE']}\n"
            f"Prediction: {feature_set['MACHINE_FAILURE']}\n"
            "\n"
            "Provide insights and maintenance recommendations based on the above data."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        insight = response.choices[0].message['content'].strip()
        insights.append(insight)
    return insights

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    
    # Only use the slider values
    features = {key: float(value) for key, value in form_data.items() if not key.endswith('_input')}
    
    features_df = pd.DataFrame([features])
    
    # Scale the features
    scaled_features = scaler.transform(features_df)
    scaled_features = np.expand_dims(scaled_features, axis=2)  # Reshape for CNN
    threshold = 0.3  # Adjust the threshold if necessary
    predictions = (model.predict(scaled_features)[:, 0] > threshold).astype(int)

    # Add predictions to features
    features['MACHINE_FAILURE'] = predictions[0]
    
    insights = analyze_predictions(predictions.tolist(), [features])
    return render_template('index.html', prediction=insights)

@app.route('/report', methods=['POST'])
def report():
    data = request.get_json()
    output_summary = data['output_summary']
    report = generate_report(output_summary)
    return jsonify({'report': report})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        
        # Drop non-feature columns for prediction
        columns_to_drop = ['ID', 'DATE', 'REGION_CLUSTER']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Ensure the DataFrame has the expected features
        expected_features = ['WELL_GROUP', 'S5', 'S13', 'S15', 'S16', 'S8', 'S18', 'S19', 'MACHINE_AGE']
        if not all(feature in df.columns for feature in expected_features):
            missing_features = [feature for feature in expected_features if feature not in df.columns]
            return f"Missing features in the uploaded file: {', '.join(missing_features)}", 400
        
        df = df[expected_features]
        
        scaled_features = scaler.transform(df)
        scaled_features = np.expand_dims(scaled_features, axis=2)  # Reshape for CNN
        threshold = 0.3  # Adjust the threshold if necessary
        predictions = (model.predict(scaled_features)[:, 0] > threshold).astype(int)

        # Add predictions to features
        df['MACHINE_FAILURE'] = predictions

        features = df.to_dict(orient='records')
        insights = analyze_predictions(predictions.tolist(), features)
        return jsonify({'insights': insights})
    return redirect(url_for('home'))

@app.route('/bulk_upload')
def bulk_upload():
    return render_template('bulk_upload.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



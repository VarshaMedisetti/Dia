from flask import Flask, render_template, request, jsonify,send_file
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Add
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from lime.lime_tabular import LimeTabularExplainer
import os
# Define the custom layer
class SkipDense(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", **kwargs):
        super(SkipDense, self).__init__(**kwargs)
        self.dense1 = Dense(units, activation=activation)
        self.dense2 = Dense(units, activation=activation)

    def call(self, inputs):
        hidden1 = self.dense1(inputs)
        hidden2 = self.dense2(hidden1)
        return Add()([hidden1, hidden2])

    def get_config(self):
        config = super(SkipDense, self).get_config()
        config.update({"units": self.dense1.units})
        return config

# Initialize Flask app
app = Flask(__name__)

# -----------------------------------
# --- Load Models and Preprocessing ---
# -----------------------------------
try:
    # Load Model
    model = load_model("diabetes_detector.keras", custom_objects={'SkipDense': SkipDense})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Ensure model is None if loading fails


try:
     # Load Scaler and PCA
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)
    print("Scaler and PCA loaded successfully!")
except Exception as e:
    print(f"Error loading scaler or PCA: {e}")
    scaler = None
    pca = None

try:
     # Load X_train and Y_train
    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    print("X_train and Y_train loaded successfully")
except Exception as e:
      print(f"Error loading X_train.npy or Y_train.npy: {e}")
      X_train = None
      Y_train = None
    
try:
    # Set up LIME explainer
    if X_train is not None and Y_train is not None:
        explainer = LimeTabularExplainer(
        X_train,
        training_labels=Y_train,
        mode="classification",
        feature_names=[f"PC{i+1}" for i in range(X_train.shape[1])]
        )
        print("LIME Explainer loaded successfully!")
    else:
        print("LIME Explainer skipped because train data not loaded")
        explainer = None

except Exception as e:
    print(f"Error setting up LIME: {e}")
    explainer = None


# Load CSV data and create HTML table (moved up)
try:
    df = pd.read_csv('diabetes_dataset.csv')  # Ensure path is correct
    data_html = df.to_html(classes='data', header="true", index=False)

    # Example feature explanations (adjust as necessary)
    feature_explanations = {
        'age': 'The age of the individual in years. Older individuals may have a higher risk of diabetes.',
        'gender': 'The gender of the individual. Gender may influence the likelihood of developing diabetes.',
        'hypertension': 'Whether the individual has hypertension (1 for Yes, 0 for No). Hypertension is a risk factor.',
        'heart_disease': 'Whether the individual has heart disease (1 for Yes, 0 for No). Heart disease can be related to diabetes.',
        'smoking_history': 'The smoking history of the individual. Smoking is a risk factor for diabetes.',
        'bmi': 'The body mass index (BMI) of the individual. Higher BMI is associated with higher risk of diabetes.',
        'hba1c_level': 'The HbA1c level, which measures blood sugar over the past 2-3 months. Higher levels indicate poor blood sugar control.',
        'blood_glucose_level': 'The current blood glucose level of the individual. Elevated levels indicate a risk of diabetes.'
    }
except Exception as e:
    print("Error loading dataset:", str(e))
    data_html = "Error loading dataset."
    feature_explanations = {}


# -----------------------------------
# --- Flask Routes ---
# -----------------------------------
@app.route('/')
def index():
    return render_template('index.html', data=data_html, features_info=feature_explanations)

# Prediction function as an API
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    if not model or not scaler or not pca or not explainer or X_train is None or Y_train is None:
        return jsonify({"error": "Model or preprocessing objects not loaded properly"})
    try:
        data = request.json

        # Extract and validate input data
        gender = data["gender"]
        age = int(data["age"])
        hypertension = int(data["hypertension"])
        heart_disease = int(data["heart_disease"])
        smoking_history = data["smoking_history"]
        bmi = float(data["bmi"])
        hba1c_level = float(data["hba1c_level"])
        blood_glucose_level = float(data["blood_glucose_level"])

        gender_map = {"Male": 1, "Female": 0}
        smoking_map = {"Never": 0, "Current": 1, "Former": 2, "Not Current": 3, "No Info": 4, "Ever": 5}

        # Map input values
        gender_num = gender_map[gender]
        smoking_num = smoking_map[smoking_history]

        input_data = np.array([[gender_num, age, hypertension, heart_disease, smoking_num, bmi, hba1c_level, blood_glucose_level]],dtype = np.float32)

        # Standardize and apply PCA
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)

        # Predict
        prediction = model.predict(input_pca)
        print('prediction: ',prediction)
        is_diabetic = np.argmax(prediction, axis=1)[0]
        
        probability = prediction[0][is_diabetic]
        if is_diabetic:
            if probability > 0.9:
                precautions = (
                    "You are at an extremely high risk of diabetes. Immediate medical intervention is critical. "
                    "Start diabetes management under medical supervision, avoid high-sugar foods completely, "
                    "and maintain a structured physical activity routine tailored by a healthcare provider."
                )
            elif probability > 0.8:
                precautions = (
                    "You are at a very high risk of diabetes. Consult a healthcare provider immediately to explore "
                    "medication options. Adopt a strict low-glycemic diet and monitor your blood sugar levels daily."
                )
            elif probability > 0.7:
                precautions = (
                    "Your diabetes risk is high. Begin regular blood sugar monitoring and increase physical activity. "
                    "Focus on reducing refined carbs, and include lean protein sources like fish and legumes in your diet."
                )
            elif probability > 0.6:
                precautions = (
                    "You are moderately at risk of diabetes. Incorporate 30-45 minutes of aerobic exercise daily. "
                    "Reduce your intake of processed foods and sugary beverages. Consult a dietitian for meal planning."
                )
            elif probability > 0.5:
                precautions = (
                    "Your diabetes risk is moderate. Schedule a glucose tolerance test to assess your condition. "
                    "Increase your intake of whole grains and leafy greens, and avoid sedentary behavior."
                )
            elif probability > 0.4:
                precautions = (
                    "You are at a slightly elevated risk of diabetes. Focus on weight management through regular exercise "
                    "and portion control. Stay hydrated and ensure you get at least 7-8 hours of sleep per night."
                )
            elif probability > 0.3:
                precautions = (
                    "Your risk of diabetes is low to moderate. Minimize your consumption of sugary snacks, "
                    "and include stress-relieving activities like yoga or meditation to promote overall health."
                )
            elif probability > 0.2:
                precautions = (
                    "Your diabetes risk is relatively low. Continue with a balanced diet that includes a variety of vegetables, "
                    "and aim for 150 minutes of moderate exercise weekly. Have your blood glucose levels checked annually."
                )
            elif probability > 0.1:
                precautions = (
                    "You are at a minimal risk of diabetes. Maintain your current healthy habits. Avoid prolonged periods of inactivity, "
                    "and include omega-3-rich foods like nuts and seeds in your diet."
                )
            else:
                precautions = (
                    "You have an extremely low risk of diabetes. Keep up your healthy lifestyle. Stay active, eat a diverse diet, "
                    "and avoid unnecessary weight gain to maintain your current health."
                )
        else:
            precautions = (
                "You are not prone to diabetes. Maintain a healthy lifestyle, including regular physical activity, "
                "a balanced diet, and routine health check-ups to prevent any future risks."
            )
        # Perform LIME analysis
        explanation = explainer.explain_instance(input_pca[0], model.predict)
        lime_filename = "lime_analysis.html"
        explanation.save_to_file(lime_filename)
        print("Prediction: ",prediction)
        lime_path = os.path.basename(lime_filename)
        
        return jsonify({
            "Prediction": "Diabetic" if is_diabetic else "Non-Diabetic",
            "Probability": f"{probability * 100:.2f}%",
            "Precautions": precautions,
            "LIME Analysis": f"Open <a href='/lime/{lime_path}'>LIME Analysis</a> in a new tab"
        })

    except Exception as e:
        return jsonify({
            "Error": str(e)
        })

# New Route to serve LIME HTML
@app.route('/lime/<filename>')
def serve_lime(filename):
    lime_file_path = os.path.join(os.getcwd(), filename) # construct file path
    if os.path.exists(lime_file_path): # check if file exists
          return send_file(lime_file_path) #send the file if exists
    else:
          return "File not found", 404
    

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
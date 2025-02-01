# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# import pickle  # Assuming you have a trained model


# app = Flask(__name__)

# @app.route('/')
# def index():
#     # Load CSV data
#     df = pd.read_csv('diabetes_dataset.csv')  # Ensure path is correct
#     # Convert dataframe to HTML table
#     data = df.to_html(classes='data', header="true", index=False)
#     return render_template('index.html', data=data)

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle  # Assuming you have a trained model

app = Flask(__name__)

@app.route('/')
def index():
    try:
        # Load CSV data
        df = pd.read_csv('diabetes_dataset.csv')  # Ensure path is correct
        
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
        
        # Convert the dataset to HTML
        data_html = df.to_html(classes='data', header="true", index=False)

        return render_template('index.html', data=data_html, features_info=feature_explanations)

    except Exception as e:
        print("Error loading dataset:", str(e))
        return jsonify({"error": "Error loading dataset."})

if __name__ == '__main__':
    app.run(debug=True)















# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import pandas as pd
# import pickle
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from lime.lime_tabular import LimeTabularExplainer

# # Initialize Flask app
# app = Flask(__name__)

# # Load dataset
# try:
#     df = pd.read_csv("diabetes_dataset.csv")
#     data_html = df.to_html(classes='data', header="true", index=False)
# except Exception as e:
#     print("Error loading dataset:", str(e))
#     data_html = "Error loading dataset."

# # Load the model
# try:
#     model = load_model("diabetes_detector.h5", custom_objects={"SkipDense": SkipDense})
# except Exception as e:
#     print("Error loading model:", str(e))
#     model = None

# # Load scaler and PCA
# try:
#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     with open("pca.pkl", "rb") as f:
#         pca = pickle.load(f)
# except Exception as e:
#     print("Error loading scaler or PCA:", str(e))
#     scaler = None
#     pca = None

# # Set up LIME explainer
# try:
#     explainer = LimeTabularExplainer(
#         np.array(pd.read_csv("diabetes_dataset.csv").dropna(subset=["diabetes"]).drop("diabetes", axis=1)),
#         training_labels=pd.read_csv("diabetes_dataset.csv")["diabetes"].astype(int),
#         mode="classification",
#         feature_names=[f"PC{i+1}" for i in range(5)],
#         discretize_continuous=False
#     )
# except Exception as e:
#     print("Error setting up LIME:", str(e))

# @app.route('/')
# def index():
#     return render_template('index.html', data=data_html)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if not model or not scaler or not pca:
#         return jsonify({"error": "Model or preprocessing objects not loaded properly"})
    
#     try:
#         # Get form data
#         data = request.json
#         gender = 1 if data['gender'] == 'Male' else 0
#         smoking_map = {"never": 0, "not current": 1, "current": 2}
#         smoking_history = smoking_map[data['smoking_history']]
        
#         # Prepare input
#         input_data = np.array([[ 
#             gender, int(data['age']), int(data['hypertension']), 
#             int(data['heart_disease']), smoking_history, float(data['bmi']), 
#             float(data['hba1c_level']), float(data['blood_glucose_level'])
#         ]])
        
#         # Preprocess input
#         input_scaled = scaler.transform(input_data)
#         input_pca = pca.transform(input_scaled)
        
#         # Predict
#         prediction = model.predict(input_pca)
#         is_diabetic = np.argmax(prediction, axis=1)[0]
#         probability = prediction[0][is_diabetic] * 100
        
#         result = "Diabetic" if is_diabetic else "Non-Diabetic"
        
#         # LIME Explanation
#         explanation = explainer.explain_instance(input_pca[0], model.predict)
#         lime_filename = "lime_analysis.html"
#         explanation.save_to_file(lime_filename)
        
#         return jsonify({
#             "Prediction": result,
#             "Probability": f"{probability:.2f}%",
#             "LIME Analysis": f"Saved to {lime_filename}. Open in a browser."
#         })
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

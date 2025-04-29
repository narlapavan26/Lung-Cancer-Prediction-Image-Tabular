# # import numpy as np
# # import cv2
# # import tensorflow as tf
# # from tensorflow.keras.models import load_model
# # from flask import Flask, request, jsonify, render_template
# # from PIL import Image
# # import io

# # app = Flask(__name__)

# # # Load the trained model
# # def load_cnn_model():
# #     return load_model("model_lung.keras")

# # model = load_cnn_model()

# # # Function for preprocessing image
# # def preprocess_image(image):
# #     image = np.array(image)
# #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
# #     image = cv2.resize(image, (224, 224))  # Resize to match model input size
# #     image = image / 255.0  # Normalize
# #     image = np.expand_dims(image, axis=-1)  # Add channel dimension
# #     image = np.expand_dims(image, axis=0)   # Add batch dimension
# #     return image

# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file uploaded'}), 400
    
# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({'error': 'No file selected'}), 400
    
# #     image = Image.open(io.BytesIO(file.read()))
# #     processed_image = preprocess_image(image)
# #     prediction = model.predict(processed_image)
    
# #     disease_probability = float(prediction[0][0])
# #     result = "Lung Disease" if disease_probability > 0.5 else "Healthy Lungs"
    
# #     return jsonify({'prediction': result, 'disease_probability': disease_probability})

# # if __name__ == '__main__':
# #     app.run(debug=True)




# from flask import Flask, render_template, request
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.models import load_model
# from werkzeug.utils import secure_filename
# import os
# import pandas as pd

# app = Flask(__name__)

# # Load models
# cnn_model = load_model("model_lung.keras")
# tabular_model = load_model("model_tabular.h5")

# UPLOAD_FOLDER = 'static/uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Home Page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Image Prediction Page
# @app.route('/predict_image', methods=['GET', 'POST'])
# def predict_image():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
#             file.save(filepath)
            
#             # Preprocess image
#             image = cv2.imread(filepath)
#             image = cv2.resize(image, (224, 224))
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = np.expand_dims(image, axis=0)
#             image = preprocess_input(image)  # MobileNet Preprocessing
            
#             # Make prediction
#             prediction = cnn_model.predict(image)
#             result = "Lung Disease" if prediction[0][0] > 0.5 else "Healthy Lungs"
            
#             return render_template('predict_image.html', result=result, filepath=filepath)
#     return render_template('predict_image.html', result=None)

# # Tabular Data Prediction Page
# @app.route('/predict_tabular', methods=['GET', 'POST'])
# def predict_tabular():
#     if request.method == 'POST':
#         input_values = [float(request.form[f'feature{i}']) for i in range(1, 6)]
#         input_array = np.array(input_values).reshape(1, -1)
#         prediction = tabular_model.predict(input_array)
#         result = f"Prediction: {prediction[0][0]:.4f}"
#         return render_template('predict_tabular.html', result=result)
#     return render_template('predict_tabular.html', result=None)

# if __name__ == '__main__':
#     app.run(debug=True)




import os
import numpy as np
import pandas as pd
import joblib
import pickle
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

app = Flask(__name__)

# Load models
def load_cnn_model():
    return load_model("model_lung.keras")

def load_tabular_model():
    return joblib.load("lung_cancer_prediction.pkl")

cnn_model = load_cnn_model()
tabular_model = load_tabular_model()

# Preprocess function for images
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)  # Convert to NumPy array
    if image.shape[-1] == 1:  
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel (RGB)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Apply MobileNetV2 preprocessing
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tabular', methods=['GET', 'POST'])
def predict_tabular():
    if request.method == 'POST':
        gender = 1 if request.form['gender'] == 'Male' else 0
        age = int(request.form['age'])
        smoking = 1 if request.form['smoking'] == 'Yes' else 0
        yellow_fingers = 1 if request.form['yellow_fingers'] == 'Yes' else 0
        anxiety = 1 if request.form['anxiety'] == 'Yes' else 0
        peer_pressure = 1 if request.form['peer_pressure'] == 'Yes' else 0
        chronic_disease = 1 if request.form['chronic_disease'] == 'Yes' else 0
        fatigue = 1 if request.form['fatigue'] == 'Yes' else 0
        allergy = 1 if request.form['allergy'] == 'Yes' else 0
        wheezing = 1 if request.form['wheezing'] == 'Yes' else 0
        alcohol = 1 if request.form['alcohol'] == 'Yes' else 0
        coughing = 1 if request.form['coughing'] == 'Yes' else 0
        shortness_of_breath = 1 if request.form['shortness_of_breath'] == 'Yes' else 0
        swallowing_difficulty = 1 if request.form['swallowing_difficulty'] == 'Yes' else 0
        chest_pain = 1 if request.form['chest_pain'] == 'Yes' else 0
        
        # columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
        #            'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 
        #            'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        
        columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                   'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
                   'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                   'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        # values = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
        #           fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]
        
        
        values = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, 
                  fatigue, allergy, wheezing, alcohol, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]
        df = pd.DataFrame([values], columns=columns)
        
        prediction = tabular_model.predict(df)
        
        if prediction[0] == 1:
            return render_template('predict_tabular.html', result="Lung Cancer", prediction_class="error")
        else:
            return render_template('predict_tabular.html', result="No Lung Cancer", prediction_class="success")
    return render_template('predict_tabular.html')

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = cnn_model.predict(processed_image)

            disease_prob = prediction[0][0]
            if disease_prob > 0.5:
                result = "Lung Disease"
                prediction_class = "error"
            else:
                result = "Healthy Lungs"
                prediction_class = "success"
            return render_template('predict_image.html', result=result, disease_prob=disease_prob, prediction_class=prediction_class)
    
    return render_template('predict_image.html')

if __name__ == '__main__':
    app.run(debug=True)

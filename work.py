import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import random
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load models
@st.cache_resource
def load_cnn_model():
    return load_model("model_lung.keras")

# @st.cache_resource
# def load_tabular_model():
#     return joblib.load("lung_cancer_prediction.pkl")

@st.cache_resource
def load_tabular_model():
    # return joblib.load("lung_cancer_prediction.pkl")
    return joblib.load("best_model.pkl")


cnn_model = load_cnn_model()
tabular_model = load_tabular_model()

# Preprocess function for images
# def preprocess_image(image):
#     image = image.resize((224, 224))  # Resize to model input size
#     image = np.array(image)  # Convert to NumPy array
#     if image.shape[-1] == 1:  
#         image = np.stack([image] * 3, axis=-1)  # Convert grayscale to 3-channel (RGB)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     image = preprocess_input(image)  # Apply MobileNetV2 preprocessing
#     return image

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB mode
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image)  # Convert to NumPy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Apply MobileNetV2 preprocessing
    return image


# Title
st.title("Lung Disease Prediction")

# Select prediction type
option = st.radio("Choose a prediction method:", ("Tabular Data Prediction", "Image-Based Prediction"))

if option == "Tabular Data Prediction":
    with st.form("lung_prediction"):
        st.write("### Enter Patient Details:")
    
        # patient_id = st.text_input("Patient ID", "P1")
        # patient_id = st.number_input("Patient Id", min_value=1, max_value=120, step=1)
        patient_id = random.randint(1, 999)
        print(patient_id)
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        gender = 2 if st.radio("Gender", ["Male", "Female"]) == "Male" else 1
        air_pollution = st.slider("Air Pollution Exposure", 1, 5, 3)
        alcohol_use = st.slider("Alcohol Use", 1, 5, 3)
        dust_allergy = st.slider("Dust Allergy", 1, 5, 3)   
        occupational_hazards = st.slider("Occupational Hazards", 1, 5, 3)
        genetic_risk = st.slider("Genetic Risk", 1, 5, 3)
        chronic_lung_disease = st.slider("Chronic Lung Disease", 1, 5, 3)
        balanced_diet = st.slider("Balanced Diet", 1, 5, 3)
        obesity = st.slider("Obesity", 1, 5, 3)
        smoking = st.slider("Smoking", 1, 5, 3)
        smoking_1 = st.slider("Smoking History", 1, 5, 3)
        passive_smoker = st.slider("Passive Smoker", 1, 5, 3)
        chest_pain = st.slider("Chest Pain", 1, 5, 3)
        coughing_blood = st.slider("Coughing Blood", 1, 5, 3)
        fatigue = st.slider("Fatigue", 1, 5, 3)
        weight_loss = st.slider("Weight Loss", 1, 5, 3)
        shortness_of_breath = st.slider("Shortness of Breath", 1, 5, 3)
        wheezing = st.slider("Wheezing", 1, 5, 3)
        swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 5, 3)
        swallowing_difficulty_1 = st.slider("Swallowing Difficulty (Severe)", 1, 5, 3)
        clubbing_fingers = st.slider("Clubbing of Fingers", 1, 5, 3)
        frequent_cold = st.slider("Frequent Cold", 1, 5, 3)
        dry_cough = st.slider("Dry Cough", 1, 5, 3)
        snoring = st.slider("Snoring", 1, 5, 3)
        
        submit = st.form_submit_button("Predict")

    if submit:
        # Creating dataframe for prediction
        columns = ['Patient Id', 'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
                'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
                'Balanced Diet', 'Obesity', 'Smoking', 'Smoking.1', 'Passive Smoker',
                'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                'Swallowing Difficulty.1', 'Clubbing of Finger Nails', 'Frequent Cold',
                'Dry Cough', 'Snoring']
        
        values = [patient_id, age, gender, air_pollution, alcohol_use, dust_allergy, occupational_hazards,
                genetic_risk, chronic_lung_disease, balanced_diet, obesity, smoking,
                smoking_1, passive_smoker, chest_pain, coughing_blood, fatigue, weight_loss,
                shortness_of_breath, wheezing, swallowing_difficulty, swallowing_difficulty_1,
                clubbing_fingers, frequent_cold, dry_cough, snoring]

        df = pd.DataFrame([values], columns=columns)
        st.write("Feature count:", df.shape[1])  # Debugging
        prediction = tabular_model.predict(df)
        
        st.subheader("Prediction Result:")
        if prediction[0] == 2:  # Assuming 2 means lung disease
            st.error("The model predicts *Lung Disease*")
        else:
            st.success("The model predicts *No Lung Disease*")
            
            
            
            
elif option == "Image-Based Prediction":
    uploaded_file = st.file_uploader("Upload Lung X-ray Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # st.image(image, caption="Uploaded Image", use_container_width=True)
        st.image(image, caption="Uploaded Image")


        processed_image = preprocess_image(image)
        prediction = cnn_model.predict(processed_image)
        print(prediction)

        labels = {
            0: "Bacterial Pneumonia",
            1: "Corona Virus Disease",
            2: "Normal",
            3: "Tuberculosis",
            4: "Viral Pneumonia"
        }
        
        max_index = np.argmax(prediction)
        
        predicted_class = labels[max_index]
        
        print("Expected Model Input Shape:", cnn_model.input_shape)
        print("Processed Image Shape:", processed_image.shape)


        st.subheader("Prediction Result:")

        if max_index == 2:
            st.success("The model predicts *Healthy Lungs*")
        else:
            st.error(f"The model predicts *Lung Disease* {predicted_class}")
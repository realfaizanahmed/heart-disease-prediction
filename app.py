import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Streamlit App
st.title("Heart Disease Prediction")

# Display model performance
st.write(f"### Model Accuracy")
st.write(f"- **Training Accuracy**: {training_data_accuracy:.2f}")
st.write(f"- **Test Accuracy**: {test_data_accuracy:.2f}")

# User Input Form
st.sidebar.header("User Input")

def get_user_input():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}[x])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}[x])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}[x])
    
    user_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    return np.asarray(user_data).reshape(1, -1)

# Get user input
input_data = get_user_input()

# Make a prediction
prediction = model.predict(input_data)

# Display the prediction result
st.write("### Prediction Result")
if prediction[0] == 0:
    st.write("**This System Predicts that This Person does not have Heart Disease**")
else:
    st.write("**This System Predicts that The Person may have a Heart Disease. Consult your Doctor/Physician Immediately !!!**")
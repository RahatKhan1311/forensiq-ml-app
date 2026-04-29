# ForensiQ - Crime Prediction System

An interactive ML-powered web app that predicts crime types 
based on case details like location, evidence type and case status.

## 🚀 Live Demo
👉 [Try the app here](https://forensiq-ml-app-by-rahatkhan.streamlit.app/)

## How It Works
1. Select location, evidence type and case status
2. Set month and weekday
3. Click Predict — get instant crime type prediction
4. View model confidence chart for all crime types

## Model Details
- Algorithm: Random Forest Classifier
- Dataset: 1000 synthetic crime cases with realistic patterns
- Accuracy: 58% | Cross Validation Mean: 56.9%
- Pipeline: OneHotEncoder + Random Forest

## Key Features
- Real-time crime type prediction
- Probability confidence chart (Plotly)
- Clean two-column UI layout
- Deployed on Streamlit Cloud

## Technologies Used
- Python
- Scikit-learn (Random Forest, Pipeline, ColumnTransformer)
- Streamlit
- Plotly
- Joblib

## Author
Rahat Khan
LinkedIn: www.linkedin.com/in/rahatkhan1305
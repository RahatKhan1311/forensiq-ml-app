import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# CONFIG
st.set_page_config(page_title="ForensiQ", layout="wide")

# LOAD MODEL
model = joblib.load("crime_model.pkl")

# UI HEADER
st.title("🚓 ForensiQ - Crime Prediction System")
st.markdown("Predict likely crime type based on case details")

# INPUT SECTION
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", ['Mumbai','Delhi','Pune','Bangalore','Chennai'])
    evidence = st.selectbox("🧾 Evidence Type", ['CCTV','Logs','Witness','Documents','Forensic'])
    status = st.selectbox("📂 Case Status", ['Open','Closed','Pending'])

with col2:
    month = st.slider("📅 Month", 1, 12)
    weekday = st.slider("📆 Weekday (0=Mon)", 0, 6)

st.markdown("---")

# PREDICTION
if st.button("🔍 Predict Crime Type"):

    input_data = pd.DataFrame([{
        'location': location,
        'evidence_type': evidence,
        'status': status,
        'month': month,
        'weekday': weekday
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"🚨 Predicted Crime Type: **{prediction}**")

    # PROBABILITY INSIGHTS (Dynamic)
    st.markdown("## 📊 Prediction Confidence")

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    prob_df = pd.DataFrame({
        "Crime Type": classes,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)

    st.info("Higher probability = more likely prediction")

    fig = px.bar(
        prob_df,
        x="Probability",
        y="Crime Type",
        orientation="h",
        color="Probability",
        color_continuous_scale="Blues",
        title="Model Confidence"
    )

    fig.update_layout(height=350)

    st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("""
### 📌 About ForensiQ
This system predicts crime types using machine learning based on:
- Location
- Evidence type
- Case status
- Time factors  

⚠️ Note: Uses synthetic data for demonstration.
""")
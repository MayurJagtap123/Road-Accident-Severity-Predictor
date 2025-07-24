# app.py

import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Load model and column list
model = joblib.load("xgboost_classification_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Severity mapping from prediction integer to label
severity_mapping = {
    0: "Slight",
    1: "Serious",
    2: "Fetal"
}

# Streamlit UI
st.set_page_config(page_title="Accident Severity Predictor", layout="centered")
st.title("🚦 Road Accident Severity Predictor")

st.markdown("### Enter Road and Environmental Conditions")

# Input form
with st.form("prediction_form"):
    junction_control = st.selectbox("Junction Control", ['Authorised person', 'Auto traffic signal', 'Give way or uncontrolled', 'Not at junction or within 20 metres', 'Stop sign'])
    junction_detail = st.selectbox("Junction Detail", ['Crossroads', 'Mini-roundabout', 'More than 4 arms (not roundabout)', 'Private drive or entrance', 'Roundabout', 'Slip road', 'T or staggered junction'])
    light_conditions = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lighting unknown', 'Darkness - lights unlit'])
    road_surface = st.selectbox("Road Surface Conditions", ['Dry', 'Wet or damp', 'Snow', 'Frost or ice', 'Flood over 3cm. deep', 'Oil or diesel', 'Mud'])
    road_type = st.selectbox("Road Type", ['Single carriageway', 'Dual carriageway', 'Roundabout', 'One way street', 'Slip road', 'Unknown'])
    weather = st.selectbox("Weather Conditions", ['Fine no high winds', 'Raining no high winds', 'Snowing no high winds', 'Fine + high winds', 'Raining + high winds', 'Fog or mist', 'Other', 'Unknown'])
    vehicle = st.selectbox("Vehicle Type", ['Car', 'Van', 'Motorcycle', 'Bus', 'Bicycle', 'Other'])

    submit = st.form_submit_button("Predict Severity")

# On predict
if submit:
    input_df = pd.DataFrame([{
        'Junction_Control': junction_control,
        'Junction_Detail': junction_detail,
        'Light_Conditions': light_conditions,
        'Road_Surface_Conditions': road_surface,
        'Road_Type': road_type,
        'Weather_Conditions': weather,
        'Vehicle_Type': vehicle
    }])

    # One-hot encode and align columns
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    severity = severity_mapping.get(prediction, "Unknown")

    st.markdown("---")
    st.success(f"🧠 **Predicted Accident Severity:** {severity}")
    st.markdown("---")
# Streamlit-based Predictive Web App

import streamlit as st
import numpy as np
import pickle

# Load the saved model
model_path = r"C:/Users/Anudip/OneDrive/Desktop/minor project/student_model.sav"  # Adjust path accordingly
model = pickle.load(open(model_path, "rb"))

# Set up the Streamlit app
st.title("Student Performance Prediction")
st.markdown("### Enter the following details to predict student performance:")

# Input fields
hours_studied = st.slider("Hours Studied", min_value=0, max_value=15, value=7)
previous_scores = st.number_input("Previous Scores (%)", min_value=0, max_value=100, value=85)
extracurricular_activities = st.selectbox(
    "Participation in Extracurricular Activities", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
sleep_hours = st.slider("Sleep Hours", min_value=0, max_value=12, value=8)
sample_question_papers_practiced = st.slider("Sample Question Papers Practiced", min_value=0, max_value=20, value=3)

# Collect input data
input_data = np.array([
    hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced
]).reshape(1, -1)

# Predict button
if st.button("Predict"):
    # Make the prediction
    prediction = model.predict(input_data)
    performance_index = prediction[0]

    # Display the prediction result
    st.subheader("Predicted Performance Index:")
    st.write(f"**{performance_index}**")

    # Interpret the result
    if performance_index >= 90:
        st.success("Excellent performance predicted!")
    elif performance_index >= 75:
        st.info("Good performance predicted!")
    elif performance_index >= 50:
        st.warning("Average performance predicted.")
    else:
        st.error("Below average performance predicted. Additional support recommended.")



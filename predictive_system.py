import pandas as pd
import pickle

# Load the trained model
student_model = pickle.load(open(r"C:/Users/Anudip/OneDrive/Desktop/minor project/student_model.sav", 'rb'))

# Input feature names (exactly as provided)
feature_names = ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']

# Input data
input_data = (7, 85, 1, 8, 3)  # Example input values

# Convert input data into a DataFrame with appropriate column names
input_data_df = pd.DataFrame([input_data], columns=feature_names)

# Predict using the model
predicted_performance = student_model.predict(input_data_df)

# Display the prediction result
print(f"Predicted Performance Index: {predicted_performance[0]}")

# Optionally, interpret the prediction (e.g., based on performance ranges)
if predicted_performance[0] >= 90:
    print("Excellent performance predicted!")
elif predicted_performance[0] >= 75:
    print("Good performance predicted!")
elif predicted_performance[0] >= 50:
    print("Average performance predicted.")
else:
    print("Below average performance predicted. Additional support recommended.")

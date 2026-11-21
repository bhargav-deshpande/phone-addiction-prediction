import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv("teen_phone_addiction_dataset.csv")
model = joblib.load("addiction_level.pkl")
columns = joblib.load("columns.pkl")

st.title("📱 Teen Phone Addiction Predictor")
data = {}

numeric = [
    'Age', 'Daily_Usage_Hours', 'Sleep_Hours',
    'Phone_Checks_Per_Day', 'Apps_Used_Daily',
    'Time_on_Social_Media', 'Time_on_Gaming',
    'Time_on_Education', 'Weekend_Usage_Hours',
    'Screen_Time_Before_Bed', 'Exercise_Hours',
    'Social_Interactions', 'Academic_Performance',
    'Family_Communication', 'Anxiety_Level',
    'Depression_Level', 'Self_Esteem', 'Parental_Control'
]

for col in numeric:
    data[col] = st.number_input(col, value=0.0)

data['Gender'] = st.selectbox("Gender", df['Gender'].unique())
data['School_Grade'] = st.selectbox("School Grade", df['School_Grade'].unique())
data['Phone_Usage_Purpose'] = st.selectbox(
    "Phone Usage Purpose",
    df['Phone_Usage_Purpose'].unique()
)

if st.button("🔮 Predict"):
    input_data = pd.DataFrame([data])

    input_data = pd.get_dummies(
        input_data,
        columns=['Gender', 'School_Grade', 'Phone_Usage_Purpose'],
        drop_first=True,
        dtype=int
    )

    input_data = input_data.reindex(columns=columns, fill_value=0)
    st.write("Input data:")
    st.write(input_data)

    st.write("Prediction:")
    st.write(model.predict(input_data))
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Addiction Level: {prediction:.2f}")
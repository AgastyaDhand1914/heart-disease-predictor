import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "heart_final.pkl"))

st.title("Heart Disease Predictor")
st.markdown("Provide the following information to predict the risk of heart disease")

age = st.slider("Age", min_value=18, max_value=100, value=18)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox('Chest Pain Type', ['Atypical Angina (ATA)', 'Non-Anginal Pain (NAP)', 'Typical Angina (TA)', 'Asymptomatic (ASY)'])
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=70, max_value=200, value=120)
cholesterol = st.slider('Cholesterol (mg/dL)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dL ?', ['Yes', 'No'])
resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
max_hr =st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if fasting_bs == 'Yes':
    fasting_bs = 1
else:
    fasting_bs = 0

start = chest_pain.find('(') + 1
end = chest_pain.find(')')
chest_pain_type = chest_pain[start:end]

exercise_angina_type = exercise_angina[:1]

sex = sex[:1]

if st.button("Predict"):
    input = {
        'Age' : age,
        'Sex_' + sex : 1,
        'ChestPainType_' + chest_pain_type : 1,
        'RestingBP' : resting_bp,
        'Cholesterol' : cholesterol,
        'FastingBS' : fasting_bs,
        'RestingECG_' + resting_ecg : 1,
        'MaxHR': max_hr,
        'ExerciseAngina_' + exercise_angina_type.strip() : 1,
        'Oldpeak' : oldpeak,
        'ST_Slope_' + st_slope : 1
    }

    df = pd.DataFrame([input])

    for col in columns:
        if col not in df.columns:
            df[col] = 0
    

    df = df[columns]
    df = df[[col for col in df.columns if col in columns]]

    scaled_df = scaler.transform(df)
    prediction = model.predict(scaled_df)[0]
    
    if prediction == 1:
        st.error("You have a high risk of heart disease. Please consult a Medical Professional.")
    else:
        st.success("You have a low risk of heart disease. Continue improving your lifestyle.")

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and preprocessors
model = load_model('university_recommendation_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_course.pkl', 'rb') as f:
    label_encoder_course = pickle.load(f)
with open('label_encoder_uni.pkl', 'rb') as f:
    label_encoder_uni = pickle.load(f)

# Function to predict the university, recommended course, and advice based on input marks
def predict_university_and_advice(science_marks, maths_marks, history_marks, english_marks, gre_marks):
    input_data = np.array([[science_marks, maths_marks, history_marks, english_marks, gre_marks]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    predicted_uni_index = np.argmax(prediction)
    predicted_uni = label_encoder_uni.inverse_transform([predicted_uni_index])[0]

    # Determine the recommended course based on the highest marks
    marks_dict = {
        "Science": science_marks,
        "Maths": maths_marks,
        "History": history_marks,
        "English": english_marks,
        "GRE": gre_marks
    }
    highest_mark_subject = max(marks_dict, key=marks_dict.get)

    if highest_mark_subject == "Science":
        recommended_course = "MBBS"
    elif highest_mark_subject == "Maths":
        recommended_course = "Engineering"
    elif highest_mark_subject == "History":
        recommended_course = "History"
    elif highest_mark_subject == "English":
        recommended_course = "Literature"
    elif highest_mark_subject == "GRE":
        recommended_course = "Business Administration"

    # Personalized advice
    advice = []

    if science_marks < 50:
        advice.append("Focus on improving your Science marks.")
    if maths_marks < 50:
        advice.append("Focus on improving your Maths marks.")
    if history_marks < 50:
        advice.append("Focus on improving your History marks.")
    if english_marks < 50:
        advice.append("Focus on improving your English marks.")
    if gre_marks < 300:
        advice.append("Consider retaking the GRE to improve your score.")

    if recommended_course == "Engineering":
        advice.append("Enhance your programming skills by taking online courses.")
    elif recommended_course == "Computer Science":
        advice.append("Work on projects and internships related to software development.")
    elif recommended_course == "Physics":
        advice.append("Participate in research projects and science fairs.")
    elif recommended_course == "History":
        advice.append("Read extensively and engage in historical research projects.")
    elif recommended_course == "Chemistry":
        advice.append("Gain hands-on experience in laboratories and participate in chemistry competitions.")
    elif recommended_course == "MBBS":
        advice.append("Gain practical experience by volunteering at clinics or hospitals.")
    elif recommended_course == "Literature":
        advice.append("Engage in extensive reading and writing practice.")
    elif recommended_course == "Business Administration":
        advice.append("Develop leadership and management skills through relevant courses and activities.")

    return predicted_uni, recommended_course, advice

# Streamlit app layout
st.title('University Recommendation System')

science_marks = st.number_input('Science Marks', min_value=0.0, max_value=100.0)
maths_marks = st.number_input('Maths Marks', min_value=0.0, max_value=100.0)
history_marks = st.number_input('History Marks', min_value=0.0, max_value=100.0)
english_marks = st.number_input('English Marks', min_value=0.0, max_value=100.0)
gre_marks = st.number_input('GRE Marks', min_value=0.0, max_value=340.0)

if st.button('Submit'):
    university, course, advice = predict_university_and_advice(science_marks, maths_marks, history_marks, english_marks, gre_marks)
    st.write(f"Recommended University: {university}")
    st.write(f"Recommended Course: {course}")
    st.write("Personal Advice:")
    for item in advice:
        st.write(f"- {item}")

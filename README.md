# University Recommendation System

This app is created to predict the best university according to your marks in various subjects. It also includes personal advice for each individual, which helps guide you towards your university goals.

## Features

- Predicts the best university based on your marks in Science, Maths, History, English, and GRE.
- Provides personalized course recommendations.
- Offers personalized advice to help improve your chances of admission.

## Usage

To use the app, simply input your marks in the provided fields and click "Submit". The app will display the recommended university, course, and personal advice.

## Deployment

Follow these steps to deploy the app on Streamlit Community Cloud:

1. **Ensure Your Project Directory Contains the Following Files:**
   - `app.py`: Your Streamlit app script.
   - `requirements.txt`: List of dependencies.
   - `university_recommendation_model.h5`: Your saved model.
   - `scaler.pkl`: Your saved scaler.
   - `label_encoder_course.pkl`: Your saved course label encoder.
   - `label_encoder_uni.pkl`: Your saved university label encoder.

2. **Create a GitHub Repository:**
   - Go to [GitHub](https://github.com) and sign in.
   - Click on the "New" button to create a new repository.
   - Name your repository (e.g., `UniversityRecommendationApp`).
   - Optionally, add a description, choose public or private, and initialize with a README.
   - Click "Create repository".

3. **Push Your Project to GitHub:**
   Open your terminal or command prompt and navigate to your project directory. Then follow these steps:

   ```sh
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/UniversityRecommendationApp.git
   git push -u origin main


import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# LOAD TRAINED MODEL

model = joblib.load("student_performance_model.pkl")

# load the features names used when training
features = joblib.load("student_features.pkl")

# TITLE

st.title("🎓 student Performance Prediction")

st.write(
    "This application uses Linear Regression "
    "to predict a student's overall score."
)

st.divider()

# STUDENT INPUT

st.subheader("Enter Student Information")


# study hours
study_hours = st.number_input(
    "Study Hours",
    min_value=0,
    max_value=24,
    value=0
)


# attendance percentage
attendance_percentage = st.number_input(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=75,
    step=0.1
)


# internet access
internet_access = st.selectbox(
    "Internet Access",
    options=["Yes", "No"]
)


# travel time
travel_time = st.selectbox(
    "Travel Time",
    [
        "<15 min",
        "15-30 min",
        "30-60 min",
        ">60 min"
    ]
)


# extra activities
extra_activities = st.selectbox(
    "Extra Activities",
    options=["Yes", "No"]
)


# PREDICTION BUTTON

if st.button("💡 Predict overall score"):

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance_percentage": [attendance_percentage],
        "internet_access": [internet_access],
        "travel_time": [travel_time],
        "extra_activities": [extra_activities]
    })


    # ENCODE CATEGORICAL VARIABLES

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "internet_access",
            "travel_time",
            "extra_activities"
        ],
        drop_first=True
    )


    # MATCH TRAINING COLUMNS

    input_data = input_data.reindex(
        columns=features,
        fill_value=0
    )


    # MAKE PREDICTION

    prediction = model.predict(input_data)[0]


    # keep prediction between 0 and 100
    prediction = max(0, min(100, prediction))


    # DISPLAY RESULT

    st.success(
        f"Prediction Overall Score: {prediction:.2f}"
    )


    # PERFORMANCE LEVEL

    if prediction >= 70:

        st.write("🌟 performance level: Excellent")

    elif prediction >= 50:

        st.write("👍 performance level: Average")

    else:

        st.write("📚 performance level: Needs Improvement")



    # DISPLAY INPUT

    st.subheader("Student information")

    st.dataframe(
        {
            "Study Hours": [study_hours],
            "Attendance (%)": [attendance_percentage],
            "Internet Access": [internet_access],
            "Travel Time": [travel_time],
            "Extra Activities": [extra_activities]
        }
    )


# ABOUT PROJECT

st.divider()

st.subheader("About This Project")

st.write(
    "This project uses Linear Regression to predict "
    "a student's overall score based on study hours "
    "attendance percentage, internet access, travel time, "
    "and participation in extra activities."
)

st.write(
    "Linear Regression was selected because the target"
    "variable, overall_score, is continuous and numerical."
)


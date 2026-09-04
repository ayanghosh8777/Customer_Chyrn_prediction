
import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pickle
import numpy as np


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model("model.h5")


# ==========================================
# LOAD SCALER AND ENCODERS
# ==========================================

with open("level.pkl", "rb") as file:
    level = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("one.pkl", "rb") as file:
    one = pickle.load(file)


# ==========================================
# STREAMLIT APP
# ==========================================

st.title("Customer Churn Prediction")

st.write("Enter customer details below:")


# ==========================================
# USER INPUT
# ==========================================

credit_score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    level.classes_
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    max_value=250000.0,
    value=50000.0
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.checkbox(
    "Has Credit Card"
)

is_active_member = st.checkbox(
    "Is Active Member"
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    max_value=200000.0,
    value=50000.0
)


# ==========================================
# PREPARE INPUT DATA
# ==========================================

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Balance": balance,
    "Tenure": tenure,
    "NumOfProducts": num_of_products,
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active_member),
    "EstimatedSalary": estimated_salary
}

input_data = pd.DataFrame([input_data])


# ==========================================
# ENCODE GEOGRAPHY
# ==========================================

geo_encoded = one.transform(
    input_data[["Geography"]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one.get_feature_names_out(["Geography"]),
    index=input_data.index
)


# Remove original Geography and add encoded columns
input_data = pd.concat(
    [
        input_data.drop("Geography", axis=1),
        geo_encoded_df
    ],
    axis=1
)


# ==========================================
# ENCODE GENDER
# ==========================================

input_data["Gender"] = level.transform(
    input_data["Gender"]
)


# ==========================================
# MATCH SCALER FEATURE ORDER
# ==========================================

try:
    input_data = input_data[scaler.feature_names_in_]
except AttributeError:
    pass


# ==========================================
# SCALE INPUT
# ==========================================

input_data_scaled = scaler.transform(input_data)


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Churn"):

    prediction = model.predict(input_data_scaled, verbose=0)

    prediction_proba = prediction[0][0]

    st.write(
        f"Prediction probability of churn: {prediction_proba:.2f}"
    )

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")


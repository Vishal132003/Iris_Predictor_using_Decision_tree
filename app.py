import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Species Prediction App", layout="centered")

# Load model
try:
    with open('iris.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: iris.pkl file not found.")
    st.stop()

# Load dataset
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# Create dataframe for min/max values
df = pd.DataFrame(iris.data, columns=feature_names)

st.title("Iris Species Prediction App")
st.write("This app predicts the Iris species based on sepal and petal measurements.")

st.sidebar.header("Input Features")

# User input
def user_input_features():
    sepal_length = st.sidebar.slider(
        "Sepal Length (cm)",
        float(df['sepal length (cm)'].min()),
        float(df['sepal length (cm)'].max()),
        float(df['sepal length (cm)'].mean())
    )

    sepal_width = st.sidebar.slider(
        "Sepal Width (cm)",
        float(df['sepal width (cm)'].min()),
        float(df['sepal width (cm)'].max()),
        float(df['sepal width (cm)'].mean())
    )

    petal_length = st.sidebar.slider(
        "Petal Length (cm)",
        float(df['petal length (cm)'].min()),
        float(df['petal length (cm)'].max()),
        float(df['petal length (cm)'].mean())
    )

    petal_width = st.sidebar.slider(
        "Petal Width (cm)",
        float(df['petal width (cm)'].min()),
        float(df['petal width (cm)'].max()),
        float(df['petal width (cm)'].mean())
    )

    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write(target_names[prediction[0]])

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=target_names))

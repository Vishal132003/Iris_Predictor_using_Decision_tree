import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# Load the saved model
# Make sure 'iris.pkl' is in the same directory as this Streamlit app.
try:
    with open('iris.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'iris.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Load iris dataset for feature names and target names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.set_page_config(page_title="Iris Species Prediction App", layout="centered")
st.title('Iris Species Prediction')
st.write('This app predicts the Iris species based on sepal and petal measurements.')

st.sidebar.header('Input Features')

# Function to get user input
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('Sepal Width (cm)', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('Petal Length (cm)', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('Petal Width (cm)', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input features')
st.write(df_input)

# Make prediction
prediction = model.predict(df_input)
prediction_proba = model.predict_proba(df_input)

st.subheader('Prediction')
st.write(target_names[prediction[0]])

st.subheader('Prediction Probability (per class)')
st.write(pd.DataFrame(prediction_proba, columns=target_names))

st.markdown("""
**How to run this app:**
1. Save the code above as a Python file (e.g., `iris_app.py`).
2. Make sure the `iris.pkl` model file is in the same directory.
3. Open your terminal or command prompt.
4. Navigate to the directory where you saved the file.
5. Run the command: `streamlit run iris_app.py`
""")

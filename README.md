ris Species Classifier
Project Description
This project implements a Decision Tree Classifier to predict the species of Iris flowers based on their sepal and petal measurements. The project includes data loading, exploratory data analysis (EDA), model training, evaluation, and visualization of the decision tree and feature importances. Additionally, a Streamlit application is provided to deploy the trained model for interactive predictions.

Table of Contents
Project Structure
Setup Instructions
Running the Streamlit App
Requirements
Project Structure
iris_classification_notebook.ipynb: Jupyter/Colab notebook containing the entire workflow from data loading to model saving.
iris.pkl: The trained Decision Tree Classifier model, saved using pickle.
app.py: Streamlit application code for deploying the model.
requirements.txt: List of Python dependencies.
Setup Instructions
To set up and run this project locally, follow these steps:

Clone the repository (if applicable) or download the files:

git clone <repository_url>
cd <project_directory>
Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
Install the required dependencies:

pip install -r requirements.txt
Ensure the model file exists: Run the iris_classification_notebook.ipynb notebook completely to generate the iris.pkl model file.

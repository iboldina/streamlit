import streamlit as st
import joblib
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit.components.v1 as components
import  streamlit.web.cli as stcli
import sys

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "mystrapp.py"]
    sys.exit(stcli.main())
st.title('Modèle de scoring')
st.subheader("Prédiction sur la probabilité de faillite d'un client")
st.write(("Dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle"))

# Import our data
data = pd.read_csv('test_production.csv')
st.write(data.head())

# Ask the user to input an ID
id = st.number_input('Please enter an ID', min_value=1, step=1)

def predict(id: int):
    # Select the row where SK_ID_CURR equals the provided id
    row = data[data['SK_ID_CURR'] == id]
    url = f"https://web2projet7.azurewebsites.net/predict/{id}"  # Include the ID in the URL
    response = requests.post(url, json=row.to_dict())
    if response.status_code == 200:
        prediction = response.json()['prediction']
        log_proba = response.json()['log_proba']
        # Get the maximum probability
        max_log_proba = np.max(log_proba)
        # Transform the result into words
        result = 'Prêt peut être accordé' if prediction[0] == 0 else 'Prêt ne doit pas être accordé'
    else:
        st.write("Error calling the API")
        st.write("Error message: ", response.text)  # Print the actual error message
        result = "Error"
        max_log_proba = 0
    return result, max_log_proba

# Call the predict function when the user inputs an ID
if st.button('Predict'):
    result, max_log_proba = predict(id)
    st.write(result)
    st.write("The maximum predicted probability is: {:.2f}".format(max_log_proba))

# Ask the user to input a column name
column = st.selectbox('Please select a column', data.columns[1:])

def plot_distribution(column: str, id: int):
    # Plot the distribution of the selected column
    plt.hist(data[column], bins=30, edgecolor='black')
    plt.title('Distribution of ' + column)
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Add a vertical line at the value of the selected ID
    plt.axvline(x=data.loc[data['SK_ID_CURR'] == id, column].values[0], color='r', linestyle='--')

# Call the plot_distribution function when the user selects a column
if st.button('Show Distribution'):
    plot_distribution(column, id)
    st.pyplot(plt)

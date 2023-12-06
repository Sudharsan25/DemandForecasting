import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('catboost_model.pkl', 'rb') as file:
    catboost_regressor = pickle.load(file)

def preprocess(sales_price, base_price, featured, on_display, scaler):
    input_data = pd.DataFrame({
        'Sales_Price': [sales_price],
        'Base_Price': [base_price],
        'Featured_Item_Of_Week_Featured': [featured],
        'Displayed_Prominently_On Display': [on_display]
    })

    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

def home_page():
    st.title("Demand Forecasting Application")
    st.write(
        """
        Welcome to the Demand Forecasting App! This app predicts the number of units sold based on input parameters.
        Use the navigation bar on the left to explore the app. 

        Demand forecasting can be useful for many important sectors like health, finance, food, and many more. It is important for the
        industries to predict the correct demand of supplies, which will enable them to produce the right amount of supplies in time
        of need. This helps to avoid wastage, and at the same time meet the requirements of the consumer.

        To perform Demand forecasting, open source data from a particular store was used.
        """
    )

def predict_page():
    st.title("Demand Forecasting")
    sales_price = st.number_input("Enter Sales Price:")
    base_price = st.number_input("Enter Base Price:")
    featured = st.checkbox("Featured Item of the Week (1 for Yes, 0 for No)")
    on_display = st.checkbox("Displayed Prominently (1 for Yes, 0 for No)")

    if st.button("Predict"):
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        input_data_scaled = preprocess(sales_price, base_price, featured, on_display, scaler)
        prediction = catboost_regressor.predict(input_data_scaled)
        st.success(f"Predicted Number of Units Sold: {prediction}")

def main():
    st.sidebar.title("Navigation")
    pages = ["Home", "Predict"]
    selection = st.sidebar.radio("Go to", pages)

    if selection == "Home":
        home_page()
    elif selection == "Predict":
        predict_page()

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from regression_ml import CompareRegressionModels

st.title('AirBnB Price Predictor')

def load_define_data_features() -> pd.DataFrame:
    airbnb = pd.read_csv(f'{data_loc}/airbnb_nyc.csv')
    feature_cols = [
        'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 
        'availability_365', 'Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Entire home/apt',
        'Private room', 'Shared room'
    ]
    target_col = 'price'
    airbnb = pd.concat([airbnb, pd.get_dummies(airbnb['neighbourhood_group'])], axis=1)
    airbnb = pd.concat([airbnb, pd.get_dummies(airbnb['room_type'])], axis=1)
    return airbnb, feature_cols, target_col

@st.cache()
def train_and_find_best_model(dataframe, feature_cols, target_col):
    cm = CompareRegressionModels(dataframe, feature_cols, target_col)
    
    best_mae = 0
    best_model = ''
    for idx, row in cm.mae:
        if row['mean_value'] > best_mae:
            best_mae = row['mean_value']
            best_model = row['model']

    bm = cm.fitted_models[best_model]
    return bm

def gather_user_input():
    supported_regions = ('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island')
    supported_listing = ('Entire home/apt', 'Private room', 'Shared room')

    with st.form('user_input'):
        col1, col2, col3, col4 = st.columns(3)
        col5, col6 = st.columns(2)
        
        with col1:
            min_nights = st.number_input('Minimum number of nights', value=1, help="Defined by host listing")
            
        with col2:
            num_reviews = st.number_input('Number of reviews', value=10, help="Number of reviews on the listing page")
        
        with col3:
            num_host_listings = st.number_input("Number of listings host has", value=1, help="Number of listings on AirBnB, found in host profile")
        
        with col4:
            availability = st.number_input("Number of days listing is available per year", value=365, help="How many days the listing is available for renting per year")
        
        with col5:
            region = st.selectbox('Which area is the AirBnB in?', options=supported_regions)

        with col6:
            listing_type = st.selectbox('What is being rented?', options=supported_listing)
        submit = st.form_submit_button('Submit')

    if not submit:
        st.info('Review the options above then click Submit when ready')
        st.stop()
    
    region_setting = define_binary_settings(supported_regions, region)
    listing_setting = define_binary_settings(supported_listings, listing_type)
    
    user_input = [min_nights, num_reviews, num_host_listings, availability]
    user_input.append(region_setting)
    user_input.append(listing_setting)
    
    return user_input
    
def define_binary_settings(supported_setting: list, user_input: str) -> list:
    binary_setting = []
    for i in range(len(supported_setting)):
        if supported_setting[i] == user_input:
            binary_setting.append(1)  # ML needs 1 or 0, only one 1 allowed
        else:
            binary_setting.append(0)
    return binary_setting
    
def predict_using_user_input(best_model, user_input) -> float:
    predicted_price = best_model.predict([user_input]) # .predict requires 2D datapoint
    return predicted_price

def app():
    airbnb_df, feature_cols, target_col = load_define_data_features()
    model = train_and_find_best_model(airbnb_df, feature_cols, target_col)
    user_input = gather_user_input()
    prediction = predict_using_user_input(model, user_input)

    st.success(f"Predicted price of an AirBnB with selected features is __${prediction: .2f}__")

app()
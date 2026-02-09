import pickle
import streamlit as st
import pandas as pd

#load model
with open("Model/model.pkl", "rb")as file:
    model=pickle.load(file)
    
st.title("🏠California housing price prediction")
st.subheader("Enter the housing features below")

#Input fields(same order as datasets)
col1, col2, col3, col4 =st.columns(4)

with col1:
    MedInc = st.number_input("Median Income (10k $)", value=5.0, min_value=1.0, max_value=15.0, step=0.1)
    HouseAge = st.number_input("House Age (years)", value=20.0, min_value=1.0, max_value=52.0, step=1.0)

with col2:
    AveRooms = st.number_input("Average Rooms", value=6.0, min_value=2.0, max_value=10.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms", value=2.0, min_value=1.0, max_value=5.0, step=0.1)

with col3:     
    Population = st.number_input("Population", value=1000, min_value=3, max_value=5000, step=1)
    AveOccup = st.number_input("Average Occupancy", value=3.0, min_value=1.0, max_value=5.0, step=0.1)
    
with col4:    
    Latitude = st.number_input("Latitude(⁰)", value=34.0, min_value=32.0, max_value=42.0, step=0.01)
    Longitude = st.number_input("Longitude(⁰W)", value=-118.0, min_value=-124.0, max_value=-114.0, step=0.01)

if st.button("predict house price"):
  predict_value=model.predict(pd.DataFrame([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]],
  columns=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]))
  
  st.success(f"predicted house value:${predict_value[0]*100000:,.2f}")
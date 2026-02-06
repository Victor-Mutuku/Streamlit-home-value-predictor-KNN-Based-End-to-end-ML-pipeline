import pickle
import streamlit as st
import pandas as pd

#load model
with open("Model\model.pkl", "rb")as file:
    model=pickle.load(file)
    
st.title("California housing price prediction")
st.write("Enter the housing features below")

#Input fields(same order as datasets)
Medinc=st.number_input("median income",value=3.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

if st.button("predict price"):
  input_df=pd.DataFrame([[Medinc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]],
  columns=["Medinc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"])
  
  prediction=model.predict(input_df)
  
  st.success(f"predicted house value:$prediction[0]*100000:2f")
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('lr.pkl','rb'))

st.title("Salary Predictor")
yearofexp = st.text_input("enter experience")

if yearofexp:
    yearofexp = np.array(yearofexp,dtype=np.float64).reshape(-1,1)
    output = model.predict(yearofexp)
    st.write(f"predicted salary :  {output}")

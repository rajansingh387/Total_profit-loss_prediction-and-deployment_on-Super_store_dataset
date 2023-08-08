import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import pickle

import streamlit as st
df = pd.read_pickle('ordermain.pkl')
model = pd.read_pickle('piperf.pkl')


st.title('Sales Profit predictioon')
st.header("choose what suits best")

Discount = st.slider("discount in %:", min_value=0, max_value=100, value=20, step=1)
Sales = st.slider("total Sales:", min_value=0, max_value=10000, value=20, step=5)
Quantity = st.slider("quantity:", min_value=0, max_value=50, value=20, step=1)
Days_to_Ship = st.slider("days to ship:", min_value=0, max_value=365, value=20, step=1)
Category = st.selectbox('Choose product category', df['Category'].unique().astype(str))
Sub_Category = st.selectbox('Choose product sub-category', df['Sub-Category'].unique().astype(str))
Country = st.selectbox('Choose country', df['Country'].unique().astype(str))
Region = st.selectbox('Choose region', df['Region'].unique().astype(str))
Segment = st.selectbox('Choose segment', df['Segment'].unique().astype(str))
Ship_Mode = st.selectbox('Choose ship mode', df['Ship Mode'].unique().astype(str))
State = st.selectbox('Choose state', df['State'].unique().astype(str))
Company = st.selectbox('Choose product', df['Company'].unique().astype(str))


    
    
if st.button("Guess Profits"):
    if Discount >0:
        Discount= Discount/100
        d= {'Discount':Discount,'Sales':Sales,'Quantity':Quantity,'Days to Ship':Days_to_Ship,
       'Category':Category,'Sub-Category':Sub_Category,'Country':Country,
       'Region':Region,'Segment':Segment,'Ship Mode':Ship_Mode,'State':State,'Company':Company}
        test=pd.DataFrame(data=d,index=[0])
        
        predicted_price = model.predict(test)
        st.success(predicted_price)





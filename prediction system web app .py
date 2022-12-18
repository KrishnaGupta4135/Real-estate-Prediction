# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:09:04 2022

@author: welcome
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st


df13=pd.read_csv(r'C:/Users/welcome/bhk1/bhp1.csv')
y = df13.price
y.head()

df13.drop(['price'],axis=1,inplace=True)
x=df13
x.head()



#spliting data into test and train data 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)


# loading the saved model

loaded_model = pickle.load(open('C:/Users/welcome/bhk1/trained_model_bhk.sav', 'rb'))



def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]
    z = np.zeros(len(x.columns))
    z[0] = sqft
    z[1] = bath
    z[2] = bhk
    if loc_index >= 0:
        z[loc_index] = 1
        return loaded_model.predict([z])[0]
      
def main():
    
    
    # giving a title
    st.title('Real-estae Price Predication Web App')
    
    
    # getting the input data from the user
    
    
    location = st.text_input('Enter the Location')
    sqft = st.text_input('Enter the Area in Square feet ')
    bath= st.text_input('Enter the Number of bathroom')
    bhk = st.text_input('Enter the Number of Room')
    
    
    # code for Prediction
    
    # creating a button for Prediction
    
    if st.button('House price'):
        house_price = predict_price(location,sqft,bath,bhk)
    st.write(house_price)
    
    
    
    
    
if __name__ == '__main__':
    main()
    

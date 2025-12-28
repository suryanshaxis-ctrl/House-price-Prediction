from re import X
import streamlit as st
import time
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.datasets import fetch_california_housing

st.title('🏡 HOUSE PRICE PREDICTION USING ML')

st.image('https://media1.giphy.com/media/v1.Y2lkPTZjMDliOTUyaXhmZjBqdzcxZ2NmY3R5djV3YTI1M2pobDVrNGlxMmhlOHV3ZmpmaCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/l0Iy7ez9MiA0ChmbC/200w.gif')

df = pd.read_csv('house_data.csv')
x = df.iloc[:,:-3]
y = df.iloc[:,-1]

final_X = X
scaler = StandardScaler()
scaled_x = scaler.fit_transform(final_x)

st.sidebar.title('Select House Features:')



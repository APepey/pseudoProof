import streamlit as st
import pandas as pd

st.write('''
         "My first app
         Hello *world!*
         ''')

df = pd.read_csv('data.csv')
st.line_chart(df)

import streamlit as st
import pandas as pd

st.write('''
         "My first app
         Hello *world!*
         ''')

df = pd.read_csv('./raw_data/datasets/complete_dataset_true_fake.csv')
st.line_chart(df)

import streamlit as st

import datetime
import requests


## requirements
api_url = "todo/predict"
response = requests.get(api_url)

## all results
result = response.json()

## dfs
df = result[prediction_df]

## percentages
percent = result[pred_percent]

# example:
# display total % of fabricated data
st.header(
    f"According to this knn model, {percent['knn']}% of the uploaded dataframe as been fabricated."
)
# display df with fabricated rows being highlighted

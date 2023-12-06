import streamlit as st
from io import BytesIO, StringIO
import pandas as pd
import requests
import os

# Set page tab display
st.set_page_config(
    page_title="PseudoProof",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

url = "https://pseudoproofimage-rmkp3lpc6a-ew.a.run.app"


# functions
@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


# App title and description
st.header("PseudoProof")
st.markdown(
    """
A random forest machine learning model built to identify fabricated entries within datasets.\n
"""
)

st.markdown("---")

### Create a native Streamlit file upload input
# st.markdown("###")
csv_file_buffer = st.file_uploader(
    label="Upload your dataset as a .csv file:", type="csv"
)

if csv_file_buffer is not None:
    with st.spinner("Wait for it..."):
        ### Get bytes from the file buffer
        csv_bytes = csv_file_buffer.getvalue()

        ### Make request to  API (stream=True to stream response as bytes)
        res = requests.post(url + "/predict_RF", files={"csv": csv_bytes})

        if res.status_code == 200:
            # fetching data from response
            pred_bytes = res.content
            # transforming it into usable data
            byte_string = str(pred_bytes, "utf-8")
            data = StringIO(byte_string)
            # percentage of fake rows
            df_percent = eval(data.getvalue())[1]
            # showing percentage
            st.markdown("---")
            st.markdown(
                f"""
                ### We estimated that {df_percent}% of the dataset rows have been fabricated.\n
                Please see below the details of that prediction:
                """
            )
            # original df + corresponding prediction
            # creating the df
            df_res = pd.DataFrame(eval(data.getvalue())[0])
            # showing df
            st.dataframe(df_res)
            # downloading
            st.download_button(
                label="Download prediction data as CSV",
                data=convert_df_to_csv(df_res),
                file_name="PseudoProof_prediction.csv",
                mime="text/csv",
            )

        else:
            st.markdown("**Oops**, something went wrong. Please try again.")

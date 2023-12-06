import streamlit as st
from io import BytesIO, StringIO
import pandas as pd
import requests
import os

# Set page tab display
st.set_page_config(
    page_title="About",
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
st.header("Under the hood")
st.markdown(
    """
During the development of PseudoProof, we trained six different models to find the best performing one:\n
- Gradient Boosting Classifier\n
- K-Nearest Neighbours (KNN)\n
- Multi-Layer Perceptron (MLP)\n
- Naïve Bayes\n
- Random Forest\n
- Support Vector Machine (SVM)\n
While we recommend using our random forest model (available on the homepage), you can still have a look below at our other candidates and their predictions!
"""
)
st.markdown("---")

# some graphs/table/etc with the accuracy of these models?

# create a drop-down option with model names to choose from: one option per model + all models at once?

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
        res = requests.post(url + "/predict", files={"csv": csv_bytes})

        if res.status_code == 200:
            # fetching data from response
            pred_bytes = res.content
            # transforming it into usable data
            byte_string = str(pred_bytes, "utf-8")
            data = StringIO(byte_string)
            # percentage of fake rows
            # creating the df
            df_percent = pd.DataFrame(eval(data.getvalue())[1], index=[0])
            df_percent_clean = df_percent.reset_index(drop=True)
            # showing df
            st.markdown("---")
            st.markdown(
                """
Percentage of fabricated data in this dataset, according to each model:
"""
            )
            st.dataframe(df_percent_clean)
            # downloading
            st.download_button(
                label="Download percentage table as CSV",
                data=convert_df_to_csv(df_percent),
                file_name="PseudoProof_prediction.csv",
                mime="text/csv",
            )
            # original df + corresponding prediction
            # creating the df
            df_res = pd.DataFrame(eval(data.getvalue())[0])
            # showing df
            st.markdown("---")
            st.markdown(
                """
Your dataset completed with the prediction of each model:
"""
            )
            st.dataframe(df_res)
            # downloading
            st.download_button(
                label="Download prediction data table as CSV",
                data=convert_df_to_csv(df_res),
                file_name="PseudoProof_prediction.csv",
                mime="text/csv",
            )

        else:
            st.markdown("**Oops**, something went wrong. Please try again.")

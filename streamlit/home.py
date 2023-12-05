import streamlit as st
from io import BytesIO, StringIO
import pandas as pd


# from PIL import Image
import requests
from dotenv import load_dotenv
import os

# Set page tab display
st.set_page_config(
    page_title="PseudoProof",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Example local Docker container URL
# url = 'http://api:8000'
# Example localhost development URL
# url = 'http://localhost:8000'
load_dotenv()
url = os.getenv("API_URL")


# App title and description
st.header("PseudoProof")
st.markdown(
    """
            Welcome to PseudoProof!\n
            Our carefully crafted machine learning models will identify fabricated rows within datasets.\n
            More information on models selection and their parameters in the 'About' tab.
            """
)

st.markdown("---")

### Create a native Streamlit file upload input
# st.markdown("###")
csv_file_buffer = st.file_uploader(
    label="Upload your dataset as a .csv file below", type="csv"
)

if csv_file_buffer is not None:
    col1, col2 = st.columns(2)

    # with col1:
    ### Display the image user uploaded
    # open(csv_file_buffer)  # , caption="Here's the dataset you uploaded ☝️"

    with col2:
        with st.spinner("Wait for it..."):
            ### Get bytes from the file buffer
            csv_bytes = csv_file_buffer.getvalue()

            ### Make request to  API (stream=True to stream response as bytes)
            res = requests.post(url + "/predict", files={"csv": csv_bytes})

            if res.status_code == 200:
                ### Display the data returned by the API
                # st.image(res.content, caption="Dataset returned from API ☝️")
                # percent = res[pred_percent]

                pred_bytes = res.content

                byte_string = str(pred_bytes, "utf-8")
                data = StringIO(byte_string)

                df_res = pd.DataFrame(eval(data.getvalue())[0])

                df_percent = pd.DataFrame(eval(data.getvalue())[1], index=[0])

                st.dataframe(df_res)
                st.dataframe(df_percent)
                # st.header(
                #     f"According to this knn model, {percent['knn']}% of the uploaded dataframe as been fabricated."
                # )
            else:
                st.markdown("**Oops**, something went wrong. Please try again.")
                print(res.status_code, res.content)


# display df with fabricated rows being highlighted

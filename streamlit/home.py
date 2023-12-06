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


def highlight_row(s):
    if s.iloc[-1] == 1:  # Check the last column's value
        return ["background-color: yellow"] * len(s)
    else:
        return [""] * len(s)


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
st.markdown("---")

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
            # displaying some stuff to appreciate the df the results
            if df_percent == 0:
                st.markdown(
                    f"""
                ## Congratulations!\n
                Your dataset is fraud-free!
                """
                )
            else:
                st.markdown(
                    f"""
                ## This dataset might not be reliable...
                """
                )
            # original df + corresponding prediction
            # creating the df
            df_res = pd.DataFrame(eval(data.getvalue())[0])

            # showing percentage
            st.markdown(
                f"""
                ### We estimated that {df_percent}% of the dataset has been fabricated.\n
                Please see below the details of that prediction:
                """
            )
            # downloading
            st.download_button(
                label="Download prediction data as CSV",
                data=convert_df_to_csv(df_res),
                file_name="PseudoProof_prediction.csv",
                mime="text/csv",
            )

            # showing df
            # st.dataframe(df_res)
            # showing highlighted df
            df_res_highlight = df_res.style.apply(highlight_row, axis=1)
            st.dataframe(df_res_highlight)
            # some fun
            st.markdown(
                "![Alt Text](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.tenor.com%2Fimages%2F43a490c0b385d2cbb2795912be677de4%2Ftenor.gif&f=1&nofb=1&ipt=7173e99eb84864dfd355c2aa730ff368e6abf89f6ebf31f62387b5e6304b8512&ipo=images)"
            )

        else:
            st.markdown("**Oops**, something went wrong. Please try again.")

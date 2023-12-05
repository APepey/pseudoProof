import streamlit as st
from io import BytesIO, StringIO
import pandas as pd


# from PIL import Image
import requests
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd




#https://streamlit.lewagon.ai/

st.write('''
         "Fabrication Detection" is a web app that allows you to upload a CSV file to check if the data has been fabricated or not. Below are some of the features of the app.
         ''')

# from PIL import Image
# image = Image.open('images/wagon.png')
# st.image(image, caption='Le Wagon', use_column_width=False)


#create a button
if st.checkbox('Test my csv file'):
    st.write('''
        This code will only be executed when the check box is checked

        Streamlit elements injected inside of this block of code will \
        not get displayed unless it is checked
        ''')


    #upload file
    #st.set_option('deprecation.showfileUploaderEncoding', False)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)




        #show progress bar
        if st.checkbox('Show progress bar'):
            import time

            'Starting a long computation...'

            # Add a placeholder
            latest_iteration = st.empty()
            bar = st.progress(0)

            for i in range(100):
                # Update the progress bar with each iteration.
                latest_iteration.text(f'Iteration {i+1}')
                bar.progress(i + 1)
                time.sleep(0.1)

            '...and now we\'re done!'





        #show success
        st.success('This is a success!')
        st.info('This is an info')
        st.warning('This is a semi success')
        st.error('Let\'s keep positive, this might be pretty close to a success!')


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
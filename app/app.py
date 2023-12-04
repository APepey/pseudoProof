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

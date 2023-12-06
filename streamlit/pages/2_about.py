import streamlit as st

# Set page tab display
st.set_page_config(
    page_title="About",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("The story behind PseudoProof")
st.markdown("---")
st.markdown(
    """
    Crafted with :heart: & :coffee:\n
    by Anaïs Pepey, Despoina Kotsopoulou and Mariano Rubio\n
    during project week @ Le Wagon, Barcelona
    """
)
st.markdown("---")
st.markdown(
    """
    This project was greatly influenced by Michael S. Bradshaw and Samuel H. Payne's paper
    [Detecting fabrication in large-scale molecular omics data](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260395).
    """
)
st.markdown("---")
st.markdown(
    """
    Preprocessing, models parameters and all code related to this project is freely available on [GitHub](https://github.com/APepey/pseudoProof).
    """
)

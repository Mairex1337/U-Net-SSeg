import streamlit as st
from utils import configure_layout


configure_layout()
st.markdown(open("streamlit/Instructions.md").read())

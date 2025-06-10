import streamlit as st
from dotenv import load_dotenv
from utils import configure_layout

load_dotenv()
configure_layout()
st.markdown(open("streamlit/Instructions.md").read())

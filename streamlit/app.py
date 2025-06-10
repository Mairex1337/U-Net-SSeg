import streamlit as st
from dotenv import load_dotenv
from utils import configure_layout

load_dotenv()
configure_layout()
with open("streamlit/Instructions.md", "r", encoding="utf-8") as f:
    instructions = f.read()

st.markdown(instructions)

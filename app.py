import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG 
# ---------------------------------------------------------------------------
HF_REPO_ID = "Abasiofon001/Concrete_Crack_Screening"   
HF_FILENAME = "best_model .keras"
IMG_SIZE = (224, 224)               
LABEL_MAP = {0: "Non-cracked", 1: "Cracked"}   
DEFAULT_THRESHOLD = 0.5  

st.set_page_config(page_title="Bridge Deck Crack Detector", page_icon="🌉", layout="wide")


st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

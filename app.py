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
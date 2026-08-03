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

# ---------------------------------------------------------------------------
# Model loading — cached so it only downloads/loads once per session, not
# on every rerun
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    except Exception as e:
        st.error(
            f"Could not download model from '{HF_REPO_ID}/{HF_FILENAME}'. "
            f"Check the repo id, filename, and that the repo is public "
            f"(or that you've set HF_TOKEN if it's private). Error: {e}"
        )
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(
            f"Model file downloaded but failed to load. This is usually a "
            f"TensorFlow/Keras version mismatch between training and this "
            f"environment — check requirements.txt pins the same major TF "
            f"version you trained with. Error: {e}"
        )
        st.stop()

    return model

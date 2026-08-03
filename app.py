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

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Must exactly mirror the training pipeline's preprocessing.
    Training used: decode -> resize -> cast float32 -> /255.0
    (EfficientNetV2 preprocess_input is applied INSIDE the model itself,
    so do NOT apply it again here or you'll double-preprocess.)"""
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  


with st.sidebar:
    st.header(" Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01,
        help=(
            "Probability of 'Cracked' above which it's flagged. Lower this if "
            "you'd rather over-flag and manually review than miss a real crack. "
            "Don't leave this at the default without checking your own "
            "validation PR curve for the threshold that maximizes recall on "
            "the Cracked class."
        ),
    )

    st.divider()
    st.header("ℹ Model card")
    st.markdown(
        "- **Architecture:** EfficientNetV2-B0 (transfer learning)\n"
        "- **Task:** Cracked vs Non-cracked, bridge deck patches\n"
        "- **Training data:** SDNET2018 Deck subset, 2:1 undersampled,\n"
        "  group-leakage-checked split\n"
    )


st.title(" Concrete Bridge Deck Crack Detector")

tab_predict, tab_about = st.tabs([" Predict", " About this model"])

with tab_predict:
    with st.spinner("Loading model..."):
        model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a bridge deck image", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running inference..."):
            x = preprocess_image(pil_img)
            prob_cracked = float(model.predict(x, verbose=0).ravel()[0])

        pred_label = LABEL_MAP[1] if prob_cracked >= threshold else LABEL_MAP[0]

        with col2:
            st.metric("Prediction", pred_label)
            st.metric("P(Cracked)", f"{prob_cracked:.3f}")
            st.progress(prob_cracked)
            st.caption(f"Threshold in use: {threshold:.2f}")

            if pred_label == "Cracked":
                st.error(" Flagged as Cracked")
            else:
                st.success(" Flagged as Non-cracked")

            if abs(prob_cracked - threshold) < 0.1:
                st.warning(
                    "This prediction is close to the decision boundary — "
                    "treat it as borderline, not confident, and consider manual review."
                )
    else:
        st.info("Upload an image above to run the model.")

with tab_about:
    st.subheader("Known limitations")

    st.subheader("How predictions are generated")
    st.markdown(
        "Image → resize to 224×224 → scale to [0,1] → EfficientNetV2-B0 "
        "(preprocessing applied internally by the model) → sigmoid output → "
        "thresholded at the value set in the sidebar."
    )

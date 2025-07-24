"""
Interactive demo for the Garbage Classification model — Streamlit version.

Run locally (from the repo root):
    streamlit run streamlit_demo/app.py

Deploy for free on Streamlit Community Cloud (share.streamlit.io):
Main file path = streamlit_demo/app.py
(no need to configure a requirements file manually — the requirements.txt
next to this file is picked up automatically)
"""

import os
import sys

# Add the repo root (one level up from this file) to the path so `from src import ...` resolves.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image

from src import config, predict

st.set_page_config(page_title="Garbage Classifier", page_icon="♻️", layout="centered")


@st.cache_resource
def get_model():
    return predict.load_trained_model()


st.title("♻️ Garbage Classifier")
st.write(
    "Upload a photo of an item and the model will classify it into one of "
    f"**{len(config.CLASS_NAMES)}** categories ({', '.join(config.CLASS_NAMES)}), "
    "tell you whether it's recyclable, and give you a disposal tip."
)

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded photo", use_container_width=True)

    with st.spinner("Classifying..."):
        model = get_model()
        result = predict.predict(model, image)

    material = result["material"]
    confidence = result["confidence"]

    if result["is_recyclable"]:
        st.success(f"**{material.upper()}** — {confidence:.1%} confidence · ♻️ Recyclable")
    else:
        st.warning(f"**{material.upper()}** — {confidence:.1%} confidence · 🗑️ General waste")

    st.info(f"💡 {result['disposal_tip']}")

    if result["low_confidence"]:
        st.error("⚠️ Low confidence prediction — try a clearer, closer photo.")

    st.subheader("Confidence per class")
    st.bar_chart(result["all_probabilities"])
else:
    st.caption("👆 Upload a photo to get started.")

st.divider()
st.caption(
    "Model: MobileNetV2 transfer learning, 82.9% test accuracy. "
    "[View the full project on GitHub](https://github.com/shahndaa/Garbage-Image-Classification)"
)

"""
Interactive demo for the Garbage Classification model.

Run locally:
    python app/app.py

Or deploy for free on Hugging Face Spaces (recommended for a portfolio link
recruiters can click and try — see README for the 2-minute setup).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from src import config, predict

_model = None


def get_model():
    global _model
    if _model is None:
        _model = predict.load_trained_model()
    return _model


def classify(image):
    if image is None:
        return None, "Upload a photo of an item to classify it."

    model = get_model()
    result = predict.predict(model, image)

    label_scores = result["all_probabilities"]

    verdict = "♻️ Recyclable" if result["is_recyclable"] else "🗑️ General waste"
    message = (
        f"**{result['material'].upper()}** — {result['confidence']:.1%} confidence\n\n"
        f"**{verdict}**\n\n"
        f"💡 {result['disposal_tip']}"
    )
    if result["low_confidence"]:
        message += "\n\n⚠️ Low confidence — try a clearer, closer photo."

    return label_scores, message


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a photo of an item"),
    outputs=[
        gr.Label(num_top_classes=6, label="Predicted material"),
        gr.Markdown(label="Result"),
    ],
    title="♻️ Garbage Classifier",
    description=(
        "Upload a photo of a piece of waste and the model will classify it into one of "
        f"{len(config.CLASS_NAMES)} categories ({', '.join(config.CLASS_NAMES)}) and tell you "
        "whether it's recyclable, with a disposal tip."
    ),
    examples=None,
)

if __name__ == "__main__":
    demo.launch()

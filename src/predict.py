"""
Inference utilities. Uses the SAME preprocessing function as training
(tf.keras.applications.mobilenet_v2.preprocess_input) so predictions match
what the model actually learned — this was the main bug in the original code.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

from src import config


def load_trained_model(model_path=config.MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image, confidence_threshold=0.5):
    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    material = config.CLASS_NAMES[class_idx]

    result = {
        "material": material,
        "confidence": confidence,
        "is_recyclable": material != "trash",
        "disposal_tip": config.DISPOSAL_GUIDE[material],
        "all_probabilities": {name: float(p) for name, p in zip(config.CLASS_NAMES, probs)},
        "low_confidence": confidence < confidence_threshold,
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--model", default=config.MODEL_PATH)
    args = parser.parse_args()

    m = load_trained_model(args.model)
    img = Image.open(args.image_path)
    result = predict(m, img)

    print(f"Material   : {result['material'].upper()}")
    print(f"Confidence : {result['confidence']:.1%}")
    print(f"Recyclable : {'Yes' if result['is_recyclable'] else 'No'}")
    print(f"Tip        : {result['disposal_tip']}")
    if result["low_confidence"]:
        print("\n[!] Low confidence prediction — consider a clearer photo.")

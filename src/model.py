"""Model definition: MobileNetV2 backbone + a small classification head."""

import os

import tensorflow as tf
from tensorflow.keras import layers

from src import config


def _load_imagenet_backbone():
    """Build MobileNetV2 with ImageNet weights.

    Normally `weights="imagenet"` just works (Colab, most local setups).
    Some locked-down/offline environments can't reach
    storage.googleapis.com though — in that case set the
    GARBAGE_IMAGENET_WEIGHTS env var to a local .h5 file
    (e.g. downloaded once from a mirror) and it will be used instead.
    """
    weights_source = os.environ.get("GARBAGE_IMAGENET_WEIGHTS", "imagenet")
    return tf.keras.applications.MobileNetV2(
        input_shape=(*config.IMG_SIZE, 3),
        include_top=False,
        weights=weights_source,
        pooling="avg",
    )


def build_model(num_classes=len(config.CLASS_NAMES)):
    base_model = _load_imagenet_backbone()
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="garbage_classifier")
    return model, base_model


def unfreeze_for_fine_tuning(base_model, fine_tune_at=config.FINE_TUNE_AT_LAYER):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return base_model

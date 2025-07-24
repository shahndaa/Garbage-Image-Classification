"""
Dataset loading utilities.

Fixes vs. the original notebooks:
- A single, consistent preprocessing function (MobileNetV2's own
  `preprocess_input`) is used everywhere: during training, validation,
  testing AND inference. The original code trained with `rescale=1./255`
  but predicted with `preprocess_input` (which maps pixels to [-1, 1]) —
  those are two different input distributions, which is why predictions
  looked broken.
- A real 3-way split (train/val/test) instead of just train/val, so the
  reported accuracy is on data the model never influenced early stopping
  decisions with.
- Class weights are computed from the ACTUAL split, not hard-coded numbers
  copy-pasted from a previous run.
"""

import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src import config


def list_image_paths_and_labels(data_dir=config.DATA_DIR):
    """Walk the flat dataset folder (class_name/*.jpg) and return
    (filepaths, labels) as parallel lists. Labels are integer indices
    into config.CLASS_NAMES.
    """
    filepaths, labels = [], []
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {class_dir}\n"
                f"Make sure {data_dir} directly contains: {config.CLASS_NAMES}"
            )
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)

    if not filepaths:
        raise RuntimeError(f"No images found under {data_dir}")

    return np.array(filepaths), np.array(labels)


def stratified_splits(data_dir=config.DATA_DIR):
    """Return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)."""
    paths, labels = list_image_paths_and_labels(data_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(1 - config.TRAIN_SPLIT),
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )

    # split the remaining (val + test) chunk in half between val and test
    relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=config.RANDOM_SEED,
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def compute_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(config.CLASS_NAMES)
    return {cls_idx: total / (n_classes * count) for cls_idx, count in counts.items()}


def _load_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, config.IMG_SIZE)
    return img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label


def _preprocess(img, label):
    # Single source of truth for preprocessing: used at train, val, test AND
    # inference time (see src/predict.py) so the model always sees inputs
    # from the same distribution it was trained on.
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def make_dataset(paths, labels, training, batch_size=config.BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_datasets(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = stratified_splits(data_dir)

    train_ds = make_dataset(train_p, train_l, training=True, batch_size=batch_size)
    val_ds = make_dataset(val_p, val_l, training=False, batch_size=batch_size)
    test_ds = make_dataset(test_p, test_l, training=False, batch_size=batch_size)

    class_weights = compute_class_weights(train_l)

    info = {
        "train_count": len(train_p),
        "val_count": len(val_p),
        "test_count": len(test_p),
        "class_weights": class_weights,
        "test_paths": test_p,
        "test_labels": test_l,
    }
    return train_ds, val_ds, test_ds, info

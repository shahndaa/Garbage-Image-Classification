"""
Train the garbage classifier end to end:
  1. frozen-backbone warm-up (train the head only)
  2. fine-tuning (unfreeze the top of MobileNetV2 at a low learning rate)
  3. evaluation on a held-out test set (accuracy, precision/recall, confusion matrix)
  4. saves the model + training curves + confusion matrix under models/ and assets/

Usage:
    python -m src.train
    python -m src.train --skip-fine-tune   # faster, head-only training
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config, data, model as model_lib


def plot_history(histories, out_path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    if len(histories) > 1:
        plt.axvline(x=len(histories[0].history["accuracy"]) - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    if len(histories) > 1:
        plt.axvline(x=len(histories[0].history["loss"]) - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_on_test(model, test_ds, test_labels):
    probs = model.predict(test_ds, verbose=0)
    preds = np.argmax(probs, axis=1)

    report = classification_report(
        test_labels, preds, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(test_labels, preds)
    return report, cm, preds


def main(skip_fine_tune=False, head_epochs=None, fine_tune_epochs=None):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.ASSETS_DIR, exist_ok=True)

    head_epochs = head_epochs or config.HEAD_EPOCHS
    fine_tune_epochs = fine_tune_epochs or config.FINE_TUNE_EPOCHS

    print(f"Loading dataset from: {config.DATA_DIR}")
    train_ds, val_ds, test_ds, info = data.load_datasets()
    print(f"Train: {info['train_count']}  Val: {info['val_count']}  Test: {info['test_count']}")
    print(f"Class weights: {info['class_weights']}")

    model, base_model = model_lib.build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.HEAD_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, monitor="val_loss"),
    ]

    print("\n=== Stage 1: training classification head (backbone frozen) ===")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=info["class_weights"],
        callbacks=callbacks,
    )

    histories = [history_head]

    if not skip_fine_tune:
        print("\n=== Stage 2: fine-tuning top layers of MobileNetV2 ===")
        model_lib.unfreeze_for_fine_tuning(base_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_fine = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            class_weight=info["class_weights"],
            callbacks=callbacks,
        )
        histories.append(history_fine)

    plot_history(histories, os.path.join(config.ASSETS_DIR, "training_curves.png"))

    print("\n=== Evaluating on held-out test set ===")
    report, cm, preds = evaluate_on_test(model, test_ds, info["test_labels"])
    plot_confusion_matrix(cm, config.CLASS_NAMES, os.path.join(config.ASSETS_DIR, "confusion_matrix.png"))

    with open(os.path.join(config.ASSETS_DIR, "test_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")

    model.save(config.MODEL_PATH)
    print(f"\nModel saved to {config.MODEL_PATH}")
    print(f"Plots saved to {config.ASSETS_DIR}/")

    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fine-tune", action="store_true", help="Only train the classification head (faster, useful on CPU).")
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    args = parser.parse_args()
    main(skip_fine_tune=args.skip_fine_tune, head_epochs=args.head_epochs, fine_tune_epochs=args.fine_tune_epochs)

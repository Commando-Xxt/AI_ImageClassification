import os, argparse, json
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from utils import build_datasets, build_model

def main(args):
    img_h, img_w = map(int, args.img_size.split("x"))
    train_ds, val_ds, class_names = build_datasets(args.data_dir, (img_h, img_w), args.batch_size)
    num_classes = len(class_names)

    model = build_model(num_classes, (img_h, img_w, 3), base_trainable=args.finetune)

    ckpt_dir = os.path.join("models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "best.keras"),
        monitor="val_accuracy", save_best_only=True, mode="max"
    )
    early_cb = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt_cb, early_cb],
        verbose=1
    )

    export_path = os.path.join("models", "week7_mobilenetv2.keras")
    os.makedirs("models", exist_ok=True)
    model.save(export_path)

    y_true, y_pred = [], []
    for x, y in val_ds:
        p = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "val_accuracy": float(report["accuracy"]),
        "per_class": {cls: report[cls] for cls in class_names},
        "confusion_matrix": cm,
        "classes": class_names
    }
    with open(os.path.join("models", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {export_path}")
    print(f"Classes: {class_names}")
    print(f"Val acc: {metrics['val_accuracy']:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--img_size", type=str, default="224x224")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--finetune", action="store_true")
    args = p.parse_args()
    main(args)

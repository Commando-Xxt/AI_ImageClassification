import argparse, os
import numpy as np
import tensorflow as tf

def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess(img_path, img_size):
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def main(args):
    h, w = map(int, args.img_size.split("x"))
    model = load_model(args.model_path)
    x = preprocess(args.image, (h, w))
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    print(f"class_index={i}")
    print(f"prob={float(probs[i]):.6f}")
    print("probs=", [float(p) for p in probs])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/week7_mobilenetv2.keras")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--img_size", type=str, default="224x224")
    args = p.parse_args()
    main(args)

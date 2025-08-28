#!/usr/bin/env python3
# Teacher: run a .tflite ViT classifier on a dataset and save soft labels (.npz)
# Usage example:
# python teacher_softlabel_tflite.py --val_dir data/val --teacher_tflite teacher.tflite \
#   --batch 32 --temperature 2.0 --softlabel_output soft_labels.npz

import os
import argparse
import numpy as np
import tensorflow as tf

# ===== defaults (overwritten by argparse) =====
EPOCHS = 1
BATCH_SIZE = 32
STUDENT_FEATURE_DIM = 768
#TRAIN_DIR = "data/train"
#VAL_DIR = "data/val"
TRAIN_DIR = "../../../../dataset/sprout_y_n_data3/train"  # expected: data/train/class_x/xxx.jpg
VAL_DIR   = "../../../../dataset/sprout_y_n_data3/val"

TEACHER_TFLITE = "teacher.tflite"
SOFTLABEL_OUTPUT = "soft_labels.npz"
TEACHER_IMG_SIZE = (224, 224)  # typical for ViT
TEMPERATURE = 2.0

def load_image_dataset(dirpath, img_size=(224,224), batch_size=32, shuffle=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        dirpath, labels="inferred", label_mode="int",
        image_size=img_size, batch_size=batch_size, shuffle=shuffle
    )
    # keep pixel values 0..255 (we will convert depending on interpreter input dtype)
    return ds

def build_interpreter(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def prepare_input_batch(batch_images, input_detail):
    """
    batch_images: numpy array float32 in [0,255] (from TF ds)
    input_detail: dict from interpreter.get_input_details()[0]
    Return properly-typed numpy array shaped for interpreter.
    """
    shape = input_detail["shape"]
    dtype = input_detail["dtype"]
    # interpreter may expect shape (1,H,W,C) or (N,H,W,C) where N == batch size
    # ensure batch dimension matches
    # if model expects fixed batch 1, we'll iterate per example
    # We'll attempt to batch when possible.
    # Handle quantized vs float:
    if np.issubdtype(dtype, np.floating):
        # convert to float32 and scale to [0,1] if expected? Many ViT TFLite expect float32 [0,1]
        # We assume teacher was exported for 0..1 float; but if model input range is 0..255, user should adapt.
        # We'll map to [0,1] here.
        prepared = (batch_images / 255.0).astype(np.float32)
    else:
        # int8/uint8 quantized input: cast to that dtype; values should be 0..255
        prepared = np.clip(batch_images, 0, 255).astype(dtype)
    # If shape has fixed batch size (e.g., [1,224,224,3]) and prepared has larger batch,
    # caller should send one sample at a time. We rely on caller to handle that.
    return prepared

def dequantize_output(out_array, out_detail):
    """
    If output is quantized (dtype int8/uint8), convert to float using scale & zero_point
    """
    dtype = out_detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = out_detail.get("quantization", (0.0, 0))
        if isinstance(scale, (list, tuple)):
            scale = scale[0]
        if isinstance(zero_point, (list, tuple)):
            zero_point = zero_point[0]
        return (out_array.astype(np.float32) - zero_point) * scale
    else:
        return out_array.astype(np.float32)

def infer_tflite_on_dataset(interpreter, input_details, output_details, dataset, temperature, batch_size):
    """
    Runs interpreter on dataset (tf.data dataset) and returns:
      logits_all: numpy array shape (N, num_classes) - dequantized logits
      hard_labels: numpy array shape (N,)
      soft_probs_T: softmax(logits / T)
    This function will try to run batches; if model input has batch dim == 1 fixed, it will run per sample.
    """
    logits_list = []
    labels_list = []
    num_out_classes = None

    in_detail = input_details[0]
    out_detail = output_details[0]

    expected_batch = in_detail["shape"][0]  # could be 1 or -1/None; tflite gives int
    dynamic_batch = (expected_batch ==  -1) or (expected_batch == 0) or (expected_batch > 1 and expected_batch != batch_size)

    for batch_images, batch_labels in dataset:
        # batch_images: tf.Tensor shape (B,H,W,C) dtype uint8 or float32 (TF dataset loaded as uint8)
        imgs_np = batch_images.numpy()  # uint8 0..255
        B = imgs_np.shape[0]
        labels_list.append(batch_labels.numpy())

        # If interpreter expects fixed batch size 1, run per sample
        if in_detail["shape"][0] == 1:
            for i in range(B):
                arr = prepare_input_batch(imgs_np[i:i+1], in_detail)
                interpreter.set_tensor(in_detail["index"], arr)
                interpreter.invoke()
                out = interpreter.get_tensor(out_detail["index"])
                out = dequantize_output(out, out_detail)
                # out shape likely (1, C)
                logits_list.append(out.reshape(-1))
        else:
            # try to feed the whole batch if shapes compatible
            # some tflite models accept dynamic batch (shape[0]==-1) or fixed batch matching B
            in_shape = in_detail["shape"].copy()
            in_shape[0] = imgs_np.shape[0]
            # attempt to reshape interpreter input if needed
            try:
                # Some interpreters allow resize_tensor_input
                interpreter.resize_tensor_input(in_detail["index"], [imgs_np.shape[0]] + list(in_detail["shape"][1:]))
                interpreter.allocate_tensors()
                arr = prepare_input_batch(imgs_np, in_detail)
                interpreter.set_tensor(interpreter.get_input_details()[0]["index"], arr)
                interpreter.invoke()
                out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
                out = dequantize_output(out, out_detail)
                # out shape (B, C)
                for r in out:
                    logits_list.append(r.reshape(-1))
            except Exception:
                # fallback to per-sample
                for i in range(B):
                    arr = prepare_input_batch(imgs_np[i:i+1], in_detail)
                    interpreter.set_tensor(in_detail["index"], arr)
                    interpreter.invoke()
                    out = interpreter.get_tensor(out_detail["index"])
                    out = dequantize_output(out, out_detail)
                    logits_list.append(out.reshape(-1))

    logits_all = np.stack(logits_list, axis=0)
    hard_labels = np.concatenate(labels_list, axis=0)
    # compute soft probabilities with temperature
    # numerically stable softmax
    logits_T = logits_all / float(temperature)
    maxl = np.max(logits_T, axis=1, keepdims=True)
    exps = np.exp(logits_T - maxl)
    soft_probs = exps / np.sum(exps, axis=1, keepdims=True)
    return logits_all, hard_labels, soft_probs

def main(args):
    global EPOCHS, BATCH_SIZE, STUDENT_FEATURE_DIM, TRAIN_DIR, VAL_DIR, TEACHER_TFLITE, SOFTLABEL_OUTPUT, TEACHER_IMG_SIZE, TEMPERATURE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STUDENT_FEATURE_DIM = args.student_dim
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    TEACHER_TFLITE = args.teacher_tflite
    SOFTLABEL_OUTPUT = args.softlabel_output
    TEACHER_IMG_SIZE = (args.img_size[0], args.img_size[1])
    TEMPERATURE = args.temperature

    # load validation dataset (we generate soft labels on validation/public set)
    print("[INFO] Loading validation/public dataset from:", VAL_DIR)
    val_ds = load_image_dataset(VAL_DIR, img_size=TEACHER_IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Building TFLite interpreter for:", TEACHER_TFLITE)
    interpreter, input_details, output_details = build_interpreter(TEACHER_TFLITE)
    print("[INFO] Interpreter input detail:", input_details[0])
    print("[INFO] Interpreter output detail:", output_details[0])

    print("[INFO] Running inference to generate logits/soft labels...")
    logits_all, hard_labels, soft_probs = infer_tflite_on_dataset(
        interpreter, input_details, output_details, val_ds, TEMPERATURE, BATCH_SIZE
    )

    print(f"[INFO] Inferred {logits_all.shape[0]} samples, classes={logits_all.shape[1]}")

    print(f"[INFO] Saving soft labels to: {SOFTLABEL_OUTPUT}")
    # save logits, soft probabilities, hard labels
    np.savez_compressed(SOFTLABEL_OUTPUT,
                        logits=logits_all.astype(np.float32),
                        soft=soft_probs.astype(np.float32),
                        hard=hard_labels.astype(np.int32))
    print("[INFO] Done.")
#python teacher_softlabel_tflite.py --val_dir ./public_ds --teacher_tflite teacher_model.tflite \
#    --batch 32 --img_size 224 224 --temperature 2.0 --softlabel_output public_softlabels.npz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--student_dim", type=int, default=STUDENT_FEATURE_DIM)
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=VAL_DIR)
    parser.add_argument("--teacher_tflite", type=str, default=TEACHER_TFLITE, help=".tflite teacher model")
    parser.add_argument("--softlabel_output", type=str, default=SOFTLABEL_OUTPUT)
    parser.add_argument("--img_size", nargs=2, type=int, default=[224,224])
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()
    main(args)

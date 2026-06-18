import argparse
import json
import os
import zipfile
import struct

import tensorflow as tf
from tensorflow import keras
import numpy as np


def compression(args):
    with zipfile.ZipFile(f'{args.dataset}/mnist.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{args.dataset}/')

    print("====== Dataset loaded ======")


def train(args):
    with open(f'{args.dataset}/dataset/train-images.idx3-ubyte', 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        train_images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols, 1)

    with open(f'{args.dataset}/dataset/train-labels.idx1-ubyte', 'rb') as f:
        struct.unpack('>II', f.read(8))
        train_labels = np.fromfile(f, dtype=np.uint8)

    with open(f'{args.dataset}/dataset/t10k-images.idx3-ubyte', 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        test_images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols, 1)

    with open(f'{args.dataset}/dataset/t10k-labels.idx1-ubyte', 'rb') as f:
        struct.unpack('>II', f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)

    train_images = train_images.astype(np.float32) / 255.0
    test_images  = test_images.astype(np.float32)  / 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset  = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    train_dataset = train_dataset.shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset  = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    print("====== Model loaded ======")
    model = keras.Sequential([

        keras.layers.Flatten(input_shape=(28, 28, 1)),

        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=test_dataset,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")

    print("Finished Training.")
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    print("Saving the model...")
    model.save(os.path.join(os.environ["SM_MODEL_DIR"], "1"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers",    type=int,   default=2,     metavar="W")
    parser.add_argument("--epochs",     type=int,   default=10,    metavar="E")
    parser.add_argument("--batch_size", type=int,   default=64,    metavar="BS")
    parser.add_argument("--lr",         type=float, default=0.001, metavar="LR")
    parser.add_argument("--hosts",      type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument("--momentum",   type=float, default=0.9,   metavar="M")
    parser.add_argument("--dataset",    type=str,   default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model_dir",  type=str)

    args = parser.parse_args()
    compression(args)
    train(args)
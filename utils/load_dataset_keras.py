import os
import struct
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf



# ─────────────────────────────────────────────
#  1. idx-ubyte
# ─────────────────────────────────────────────

with open('train-images.idx3-ubyte', 'rb') as f:
    _, n, rows, cols = struct.unpack('>IIII', f.read(16))
    train_images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols, 1)  # NHWC

with open('train-labels.idx1-ubyte', 'rb') as f:
    struct.unpack('>II', f.read(8))
    train_labels = np.fromfile(f, dtype=np.uint8)

with open('t10k-images.idx3-ubyte', 'rb') as f:
    _, n, rows, cols = struct.unpack('>IIII', f.read(16))
    test_images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols, 1)

with open('t10k-labels.idx1-ubyte', 'rb') as f:
    struct.unpack('>II', f.read(8))
    test_labels = np.fromfile(f, dtype=np.uint8)

train_images = train_images.astype(np.float32) / 255.0
test_images  = test_images.astype(np.float32)  / 255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,  test_labels))



# ─────────────────────────────────────────────
#  2. pkl
# ─────────────────────────────────────────────

with open('train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('test.pkl', 'rb') as f:
    test_data = pickle.load(f)

train_data['images'] = train_data['images'].astype(np.float32) / 255.0
test_data['images']  = test_data['images'].astype(np.float32)  / 255.0

# print(type(train_data)) / print(train_data.keys()) 로 구조 확인 후 아래 수정
train_dataset = tf.data.Dataset.from_tensor_slices((train_data['images'], train_data['labels']))
test_dataset  = tf.data.Dataset.from_tensor_slices((test_data['images'],  test_data['labels']))



# ─────────────────────────────────────────────
#  3a. png — 폴더로 train/test 분리된 경우
#      data/train/<class>/, data/test/<class>/
# ─────────────────────────────────────────────

train_dataset = tf.keras.utils.image_dataset_from_directory(
    './data/train',
    image_size=(28, 28),
    batch_size=64
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    './data/test',
    image_size=(28, 28),
    batch_size=64
)



# ─────────────────────────────────────────────
#  3b. png — train/test 미분리, 클래스 폴더만 있는 경우
#      data/<class>/
# ─────────────────────────────────────────────

train_dataset = tf.keras.utils.image_dataset_from_directory(
    './data',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(28, 28),
    batch_size=64
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    './data',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(28, 28),
    batch_size=64
)



# ─────────────────────────────────────────────
#  3c. png — 클래스 폴더 없이 별도 라벨 파일이 있는 경우
#      labels.csv: filename, label, split
# ─────────────────────────────────────────────

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [28, 28])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

df       = pd.read_csv('labels.csv')  # columns: filename, label, split
train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

# split 컬럼이 없으면: from sklearn.model_selection import train_test_split

train_paths  = ('./data/' + train_df['filename']).tolist()
train_labels = train_df['label'].tolist()
test_paths   = ('./data/' + test_df['filename']).tolist()
test_labels  = test_df['label'].tolist()

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).map(load_image)
test_dataset  = tf.data.Dataset.from_tensor_slices((test_paths,  test_labels)).map(load_image)



# ─────────────────────────────────────────────
#  3d. png — CSV with pixel values
#      columns: label, pixel0, pixel1, ..., pixelN
# ─────────────────────────────────────────────

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

# (N, H, W, C) 로 reshape — 높이·너비·채널에 맞게 수정
train_images = train_df.drop(columns='label').values.astype(np.float32) / 255.0
train_labels = train_df['label'].values
train_images = train_images.reshape(-1, 28, 28, 1)

test_images  = test_df.drop(columns='label').values.astype(np.float32) / 255.0
test_labels  = test_df['label'].values
test_images  = test_images.reshape(-1, 28, 28, 1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset  = tf.data.Dataset.from_tensor_slices((test_images,  test_labels))



# ─────────────────────────────────────────────
#  tf.data pipeline
#  3a/3b 는 image_dataset_from_directory 에서 이미 배치 처리됨 → prefetch 만 추가
#  나머지는 아래처럼 shuffle/batch/prefetch 모두 적용
# ─────────────────────────────────────────────

train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

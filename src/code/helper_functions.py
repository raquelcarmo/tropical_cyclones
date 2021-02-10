import random
import glob
import os
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os.path
import time
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from google.colab.patches import cv2_imshow
import random
from shapely.geometry import Point
import re
import pickle
import scipy
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
from tensorflow.keras import Input
from tensorflow.keras.applications import resnet50, mobilenet_v2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import concatenate, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.set_printoptions(precision=4)

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_data(csv_path):
  # function to convert empty box to np.zeros(4)
  zero_pad = lambda y: np.zeros(4, dtype=int) if y==[] else y
  # read csv file
  df = pd.read_csv(csv_path, converters={'bbox_shape': eval}).dropna()
  images, labels = df["image"], df["label"]
  bbox_column = list(map(zero_pad, df['bbox_shape']))
  # convert from str to int
  boxes = [np.array(list(map(int, bbox))) for bbox in bbox_column]
  return images, labels, boxes


def load_from_df(df):
  # function to convert empty box to np.zeros(4)
  zero_pad = lambda y: np.zeros(4, dtype=int) if y==[] else y
  # extract info from df
  images, labels = df["image"], df["label"]
  bbox_column = list(map(zero_pad, df['bbox_shape']))
  # convert from str to int
  boxes = [np.array(list(map(int, bbox))) for bbox in bbox_column]
  return images, labels, boxes


def load_data_with_cat(csv_path):
  # function to convert empty box to np.zeros(4)
  zero_pad = lambda y: np.zeros(4, dtype=int) if y==[] else y
  # read csv file
  df = pd.read_csv(csv_path, converters={'bbox_shape': eval}).dropna()
  images = df["image"]
  bbox_column = list(map(zero_pad, df['bbox_shape']))
  # convert from str to int
  boxes = [np.array(list(map(int, bbox))) for bbox in bbox_column]

  labels = np.zeros(len(images), dtype=int)
  for index, row in df.iterrows():
    if df["label"][index] != 0:
      cat = df["image"][index].split('/')[1]
      labels[index] = int(cat[-1])
  
  # converts labels to one-hot encoded vectors
  labels = tf.keras.utils.to_categorical(labels, num_classes=6)
  return images, labels, boxes


def load_from_df_with_cat(df):
  # function to convert empty box to np.zeros(4)
  zero_pad = lambda y: np.zeros(4, dtype=int) if y==[] else y
  # extract info from df
  images = df["image"]
  bbox_column = list(map(zero_pad, df['bbox_shape']))
  # convert from str to int
  boxes = [np.array(list(map(int, bbox))) for bbox in bbox_column]

  labels = np.zeros(len(images), dtype=int)
  cnt = 0
  for index, row in df.iterrows():
    if df["label"][index] != 0:
      cat = df["image"][index].split('/')[1]
      labels[cnt] = int(cat[-1])
    cnt += 1
    
  labels = tf.keras.utils.to_categorical(labels, num_classes=6)
  return images, labels, boxes


def compute_class_weights(csv_path):
  # source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
  
  # read csv file
  df = pd.read_csv(csv_path, converters={'bbox_shape': eval}).dropna()
  labels = np.zeros(len(df), dtype=int)
  for index, row in df.iterrows():
    if df["label"][index] != 0:
      cat = df["image"][index].split('/')[1]
      labels[index] = int(cat[-1])

  # plot histograms of labels
  hist = pd.DataFrame({csv_path: labels}).hist()
  plt.show()
  #plt.savefig("SAR_swath_images_VV+VH+WS/csv/class_distribution.jpg")

  cat0, cat1, cat2, cat3, cat4, cat5 = np.bincount(labels)
  total = len(labels)

  # Scaling by total/2 helps keep the loss to a similar magnitude.
  # The sum of the weights of all examples stays the same.
  weight_for_0 = (1 / cat0)*(total)/6.0 
  weight_for_1 = (1 / cat1)*(total)/6.0
  weight_for_2 = (1 / cat2)*(total)/6.0 
  weight_for_3 = (1 / cat3)*(total)/6.0
  weight_for_4 = (1 / cat4)*(total)/6.0 
  weight_for_5 = (1 / cat5)*(total)/6.0

  print('Weights for class 0: {:.2f}, class 1: {:.2f}, class 2: {:.2f}, class 3: {:.2f}, class 4: {:.2f}, class 5: {:.2f}'.format(
      weight_for_0, weight_for_1, weight_for_2, weight_for_3, weight_for_4, weight_for_5)) 
  
  class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 
                  3: weight_for_3, 4: weight_for_4, 5: weight_for_5}
  return class_weight


def get_model_name(k):
  name = 'model_' + str(k) + '.h5'
  return name


#data_augmentation = tf.keras.Sequential([
#  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")
#])
def data_augmentation(image, label):
  image = tf.image.flip_left_right(image)
  image = tf.image.flip_up_down(image)
  return image, label

def vertical_flip(image, label):
  image = tf.image.flip_up_down(image)
  return image, label

def horizontal_flip(image, label):
  image = tf.image.flip_left_right(image)
  return image, label

def rot90(image, label):
  image = tf.image.rot90(image, k=1)
  return image, label


def configure_for_performance(ds, BUFFER_SIZE, BATCH_SIZE, shuffle = False, augment = False):
  ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(buffer_size = BUFFER_SIZE)
  # use data augmentation only on the training set
  if augment:
    #aug_ds = ds.map(data_augmentation, num_parallel_calls = AUTOTUNE)
    #ds = ds.concatenate(aug_ds)
    vert_ds = ds.map(vertical_flip, num_parallel_calls = AUTOTUNE)
    hor_ds = ds.map(horizontal_flip, num_parallel_calls = AUTOTUNE)
    rot_ds = ds.map(rot90, num_parallel_calls = AUTOTUNE)
    ds = ds.concatenate(vert_ds).concatenate(hor_ds).concatenate(rot_ds)
  # batch all datasets
  ds = ds.batch(batch_size = BATCH_SIZE)
  # use buffered prefecting on all datasets
  ds = ds.prefetch(buffer_size = AUTOTUNE)
  return ds


def prepare_dataset(processor, images, labels, bboxes):
  images_dataset = Dataset.from_tensor_slices((images, bboxes))
  labels_dataset = Dataset.from_tensor_slices(labels)
  #print(images_dataset.element_spec)

  # dataset transformation which applies the preprocessing function to each element
  processed_images_dataset = images_dataset.map(
      lambda x, y: tf.py_function(func = processor.preprocess_pipeline, inp=[x, y], Tout=[tf.float32, tf.float32]),
      num_parallel_calls = AUTOTUNE).map(
          # returns a dataset with only the processed images
          lambda x, y: x)
      
  # create a zipped dataset with the processed images and the labels
  ds = Dataset.zip((processed_images_dataset, labels_dataset))
  print(len(list(ds.as_numpy_iterator())))
  return ds


def create_dataset(processor, ncrops, images, labels, bboxes):
  processed_images_dataset = []
  labels_dataset = []

  for image_path, label, bbox in zip(images, labels, bboxes):
    # get bbox
    bbox_rotated = None if (bbox == 0).all() else bbox

    # read the image path
    im = cv2.imread(image_path).astype(np.float32) # loads images as BGR in float32
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB

    # pad images to the dimensions required
    padded_image, bbox_padded = processor.padding(image, bbox_rotated)
    height, width = padded_image.shape[:2]
    #print("Original:")
    #cv2_imshow(padded_image)
    
    if height > 288 or width > 288:
      #print('Cropped:')
      # randomly crop each image 5 times to augment the dataset
      for _ in range(ncrops):
        sized_image = processor.random_crop(padded_image, bbox_padded)
        processed_images_dataset.append(sized_image)
        labels_dataset.append(label)
        #cv2_imshow(sized_image)
    else:
      processed_images_dataset.append(padded_image)
      labels_dataset.append(label)      

  # create a dataset with the processed images and the labels
  ds = Dataset.from_tensor_slices((processed_images_dataset, labels_dataset))
  print(len(list(ds.as_numpy_iterator())))
  return ds


def get_stats(train_ds, val_ds):
  ds = train_ds.concatenate(val_ds)
  x = np.array([tfds.as_numpy(image) for image, label in ds])
  x[x == 0] = np.nan
  #mean = x.mean(axis=(0,1,2), keepdims=True)
  #std = x.std(axis=(0,1,2), keepdims=True)
  mean = np.nanmean(x, axis=(0,1,2))
  std = np.nanstd(x, axis=(0,1,2))
  print("mean: {0}, std: {1}".format(mean, std))
  return mean, std


def z_norm(train_ds, val_ds):
  mean, std = get_stats(train_ds, val_ds)

  def norm(image, label, mean = mean, std = std):
    image = (image - mean) / std
    return image, label

  train_ds = train_ds.map(norm, num_parallel_calls = AUTOTUNE)
  val_ds = val_ds.map(norm, num_parallel_calls = AUTOTUNE)
  return train_ds, val_ds

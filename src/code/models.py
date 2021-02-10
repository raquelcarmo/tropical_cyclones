import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications import resnet50, mobilenet_v2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall, TruePositives, FalsePositives, TrueNegatives, FalseNegatives

np.set_printoptions(precision=4)


def make_ResNet(IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE):

  resNet = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
  resNet_pool = GlobalAveragePooling2D()(resNet.output)

  z = Dense(6, activation="softmax")(resNet_pool)
  model = tf.keras.Model(inputs = resNet.input, outputs = z)

  # compile model
  model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                loss = "categorical_crossentropy",
                metrics = [CategoricalAccuracy(name="accuracy"), 
                           Precision(name="precision"), Recall(name="recall"), 
                           TruePositives(name='tp'), FalsePositives(name='fp'),
                           TrueNegatives(name='tn'), FalseNegatives(name='fn')])
  #model.summary()
  return model


def make_MobileNet(IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE):

  mobileNet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
  mobileNet_pool = GlobalAveragePooling2D()(mobileNet.output)

  z = Dense(6, activation="softmax")(mobileNet_pool)
  model = tf.keras.Model(inputs = mobileNet.input, outputs = z)

  # compile model
  model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                loss = "categorical_crossentropy",
                metrics = [CategoricalAccuracy(name="accuracy"), 
                           Precision(name="precision"), Recall(name="recall"), 
                           TruePositives(name='tp'), FalsePositives(name='fp'),
                           TrueNegatives(name='tn'), FalseNegatives(name='fn')])
  #model.summary()
  return model
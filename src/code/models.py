import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from keras import backend as K


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    

def make_ResNet(IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE, detection = False, eye_only = False):

    resNet = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    resNet_pool = GlobalAveragePooling2D()(resNet.output)

    if detection:
        z = Dense(1, activation="sigmoid")(resNet_pool)
        model = tf.keras.Model(inputs = resNet.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "binary_crossentropy",
                    metrics = [BinaryAccuracy(name="accuracy"), 
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])

    else: # categorisation
        classes = 6 if not eye_only else 5
        z = Dense(classes, activation="softmax")(resNet_pool)
        model = tf.keras.Model(inputs = resNet.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "categorical_crossentropy",
                    metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])
    #model.summary()
    return model


def make_MobileNet(IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE, detection = False, eye_only = False):

    mobileNet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    mobileNet_pool = GlobalAveragePooling2D()(mobileNet.output)

    if detection:
        z = Dense(1, activation="sigmoid")(mobileNet_pool)
        model = tf.keras.Model(inputs = mobileNet.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "binary_crossentropy",
                    metrics = [BinaryAccuracy(name="accuracy"), 
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])

    else: # categorisation
        classes = 6 if not eye_only else 5
        z = Dense(classes, activation="softmax")(mobileNet_pool)
        model = tf.keras.Model(inputs = mobileNet.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "categorical_crossentropy",
                    metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])
    #model.summary()
    return model
  
 
def make_VGG(IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE, detection = False, eye_only = False):

    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    vgg16_pool = GlobalAveragePooling2D()(vgg16.output)

    if detection:
        z = Dense(1, activation="sigmoid")(vgg16_pool)
        model = tf.keras.Model(inputs = vgg16.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "binary_crossentropy",
                    metrics = [BinaryAccuracy(name="accuracy"), 
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])

    else: # categorisation
        classes = 6 if not eye_only else 5
        z = Dense(classes, activation="softmax")(vgg16_pool)
        model = tf.keras.Model(inputs = vgg16.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "categorical_crossentropy",
                    metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])
    #model.summary()
    return model
  

def make_cnn(NETWORK, IMAGE_DIM_W, IMAGE_DIM_H, LEARNING_RATE, detection = False, eye_only = False, dropout = False, rate = 0):
    #sources:
    #- https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
    #- https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
    #- https://stackoverflow.com/questions/51705464/keras-tensorflow-combined-loss-function-for-single-output
    
    if NETWORK == "ResNet":
        base_cnn = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    elif NETWORK == "Mobile":
        base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    elif NETWORK == "VGG":
        base_cnn = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_DIM_W, IMAGE_DIM_H, 3))
    else:
        sys.exit("Incert valid network model. Options: Mobile, ResNet or VGG (case sensitive)")

    x = GlobalAveragePooling2D()(base_cnn.output)
    if dropout:
        x = Dropout(rate)(x)

    if detection:
        z = Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs = base_cnn.input, outputs = z)

        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), 
                    loss = "binary_crossentropy",
                    metrics = [BinaryAccuracy(name="accuracy"), 
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])

    else: # categorisation
        classes = 6 if not eye_only else 5
        z = Dense(classes, activation="softmax")(x)
        model = tf.keras.Model(inputs = base_cnn.input, outputs = z)
        #print("RMSE ON!")
        # compile model
        model.compile(optimizer = Adam(learning_rate = LEARNING_RATE),
                    loss = "categorical_crossentropy",
                    #loss = [RMSE, "categorical_crossentropy"],
                    #loss_weights = [1, 1],
                    metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                              Precision(name="precision"), Recall(name="recall"), 
                              TruePositives(name='tp'), FalsePositives(name='fp'),
                              TrueNegatives(name='tn'), FalseNegatives(name='fn')])
    #model.summary()
    return model
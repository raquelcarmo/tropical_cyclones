import os
import pandas as pd
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
from tensorflow.keras import Input
from tensorflow.keras.applications import resnet50, mobilenet_v2, vgg16
from tensorflow.keras.models import Model
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(args, csv_path="", df=None):
    # read csv file
    df = df if (csv_path=="") else pd.read_csv(csv_path, converters={'bbox_shape': eval}).dropna()

    # function to convert empty box to np.zeros(4)
    zero_pad = lambda y: np.zeros(4, dtype=int) if y==[] else y
  
    # extract info from df
    images = df["image"].values
    bbox_column = list(map(zero_pad, df['bbox_shape']))
    # convert from str to int
    boxes = [np.array(list(map(int, bbox))) for bbox in bbox_column]

    if 'eye_only' not in args.keys(): # detection: labels = [0,1]
        labels = df["label"]
    else: # categorization: labels = [0,1,2,3,4,5] or [0,1,2,3,4]
        labels = np.zeros(len(images), dtype=int)
        cnt = 0
        for index, row in df.iterrows():
            if df["label"][index] != 0:
                cat = df["image"][index].split('/')[1]
                labels[cnt] = int(cat[-1])
            cnt += 1

        if args['eye_only']:  # labels = [0,1,2,3,4]
            idx = np.where(labels == 0)[0]
            labels = np.delete(labels, idx)
            labels -= 1
            images = np.delete(images, idx)
            boxes = [boxes[i] for i in range(len(boxes)) if i not in idx]
            labels = tf.keras.utils.to_categorical(labels, num_classes=5)
        else:
            labels = tf.keras.utils.to_categorical(labels, num_classes=6)
      
    assert len(images) == len(labels) == len(boxes)
    return images, labels, boxes


def compute_class_weights(csv_path, args):
    # source: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
    # read csv file
    df = pd.read_csv(csv_path, converters={'bbox_shape': eval}).dropna()
    #print("Dataset dimension: {}".format(len(df)))

    # extract non-categorical labels for StratifiedKFold
    labels = np.zeros(len(df), dtype=int)
    for index, row in df.iterrows():
        if df["label"][index] != 0:
            cat = df["image"][index].split('/')[1]
            labels[index] = int(cat[-1])

    if args['eye_only']:  # labels = [0,1,2,3,4]
        assert isinstance(labels, np.ndarray)
        idx = np.where(labels == 0)[0]
        labels = np.delete(labels, idx)
        labels -= 1
        df.drop(idx, inplace=True)
        df.reset_index(drop=True, inplace=True)
        #print("New dimensions, Y: {} and df: {}".format(len(labels), len(df)))
    
    # plot histograms of labels
    hist = pd.DataFrame({csv_path: labels}).hist()
    plt.show()
    #plt.savefig("{}/class_distribution.jpg".format(os.path.dirname(csv_path)), bbox_inches='tight')
    
    # compute class weights
    if args['crop_mode'] != "weighted": 
        # if crop_mode == weighted (proporional to presence of classes),
        # the method compute_class_weights() will no longer represent 
        # the true distribution of classes in the dataset
        x = np.bincount(labels)
        total = len(labels)

        # Scaling by total/6 helps keep the loss to a similar magnitude;
        # The sum of the weights of all examples stays the same
        class_weight = {}
        diff_cats = len(x)
        
        # Precision values obtained in previous run
        #precisions = [0.793, 0.2, 0.36, 0.493, 0.6]
        #precisions = [0.851, 0.08, 0.4, 0.46, 0.5]

        cat = 0
        for count in x:
            class_weight[cat] = total/(count * diff_cats)
            #class_weight[cat] = total/(count * diff_cats * precisions[cat])
            cat += 1
    else:
        class_weight = None
    print("class_weights:", class_weight)
    return labels, df, class_weight


def get_model_name(k):
    name = 'model_' + str(k) + '.h5'
    return name


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
    

def rot180(image, label):
    image = tf.image.rot90(image, k=2)
    return image, label
    

def rot270(image, label):
    image = tf.image.rot90(image, k=3)
    return image, label


def config_performance(ds, args, shuffle=False, flag=False):
    '''Configures the input dataset to enhance training performance'''
    ds = ds.cache()
  
    if shuffle:
        ds = ds.shuffle(buffer_size=args['buffer_size'])
    
    # use data augmentation only on the training set (where flag==False)
    if not flag and 'data_aug' in args.keys() and args['data_aug']:   
        vert_ds = ds.map(vertical_flip, num_parallel_calls=AUTOTUNE)
        hor_ds = ds.map(horizontal_flip, num_parallel_calls=AUTOTUNE)
        
        if args['height'] == args['width']:
            rot90_ds = ds.map(rot90, num_parallel_calls=AUTOTUNE)
            rot180_ds = ds.map(rot180, num_parallel_calls=AUTOTUNE)
            rot270_ds = ds.map(rot270, num_parallel_calls=AUTOTUNE)
            ds = ds.concatenate(vert_ds)\
                    .concatenate(hor_ds)\
                        .concatenate(rot90_ds)\
                            .concatenate(rot180_ds)\
                                .concatenate(rot270_ds)
        else:
          #aug_ds = ds.map(data_augmentation, num_parallel_calls=AUTOTUNE)
          ds = ds.concatenate(vert_ds).concatenate(hor_ds)

    # batch all datasets
    ds = ds.batch(batch_size=args['batch_size'])
    # use buffered prefecting on all datasets
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


@tf.autograph.experimental.do_not_convert
def prepare_dataset(p, images, labels, bboxes):
    '''[DEPRECATED] Creates a tf.data.Dataset containing images and respective labels'''
    images_dataset = Dataset.from_tensor_slices((images, bboxes))
    labels_dataset = Dataset.from_tensor_slices(labels)
    # print(images_dataset.element_spec)

    # transformation that applies the preprocessing pipeline to each element (image, bbox)
    processed_images_dataset = images_dataset.map(
        lambda x, y: tf.py_function(func = p.preprocess_pipeline, inp=[x, y], Tout=[tf.float32, tf.float32]),
        num_parallel_calls = AUTOTUNE).map(
            # returns a dataset with only the processed images
            lambda x, y: x)
      
    # create a zipped dataset with the processed images and respective labels
    ds = Dataset.zip((processed_images_dataset, labels_dataset))
    #print(len(list(ds.as_numpy_iterator())))
    return ds


def percentage_black_pixels(image, threshold = 0.85):
    h, w = image.shape[:2]
    use = True
    percentage = (h*w - cv2.countNonZero(image[:,:,0])) / (h*w)
    if percentage > threshold:
        use = False
    return use


def create_dataset(p, images, labels, bboxes, args, flag=False):
    '''Creates a tf.data.Dataset containing images and respective labels.
    Unlike prepare_dataset(), this method allows to randomly crop each image multiple
    (n_crops) times as a way to augment the dataset'''
    assert args['nb_crops'] > 0
    processed_images_dataset, labels_dataset = [], []
    cnt = 0
    threshold = 0.85
    
    for image_path, label, bbox in zip(images, labels, bboxes):
        # get bbox
        bbox_rotated = None if (bbox == 0).all() else bbox
    
        # read the image path
        im = cv2.imread(image_path).astype(np.float32) # loads images as BGR in float32
        image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    
        if not percentage_black_pixels(image, threshold):
            cnt += 1
            continue
        
        # pad images to the dimensions required
        padded_image, bbox_padded = p.padding(image, bbox_rotated)
        h, w = padded_image.shape[:2]
        #print("Padded:")
        #cv2.imshow(padded_image)
    
        if h > args['height'] or w > args['width']:
            #print('Cropped:')
            if args['crop_mode'] == "uniform":
                if not flag:
                    # randomly crop each image (n_crops) times to augment the train dataset
                    for _ in range(args['nb_crops']):
                        sized_image,_ = p.random_crop(padded_image, bbox_padded)
                        processed_images_dataset.append(sized_image)
                        labels_dataset.append(label)
                        #cv2.imshow(sized_image)
                else:
                    # only crop once (test/val datasets)
                    sized_image,_ = p.random_crop(padded_image, bbox_padded)
                    processed_images_dataset.append(sized_image)
                    labels_dataset.append(label)
              
            #else: 
            # TO DO: implement weighted crop mode
        else:
            processed_images_dataset.append(padded_image)
            labels_dataset.append(label)      

    print("Number of images with a percentage of black pixels higher than {}%: {}".format(threshold*100, cnt))
    print("Size of data:", len(labels_dataset))
    # create a dataset with the processed images and respective labels
    ds = Dataset.from_tensor_slices((processed_images_dataset, labels_dataset))
    #print(len(list(ds.as_numpy_iterator())))
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


def normalisation(train_ds, val_ds, args):
    if not args['normalise']:
        return train_ds, val_ds

    if args['norm_mode'] == "model":
        def model_norm(image, label, model=args['cnn']):
            if model == "ResNet":
                new = resnet50.preprocess_input(image)
            elif model == "Mobile":
                new = mobilenet_v2.preprocess_input(image)
            else:   #VGG16
                new = vgg16.preprocess_input(image)
            return new, label
        func = model_norm
        
    elif args['norm_mode'] == "z-norm":
        mean, std = get_stats(train_ds, val_ds)

        def z_norm(image, label, mean=mean, std=std):
            #new = (image - mean) / std
            
            zero = tf.constant(0, dtype = tf.float32)
            nan = tf.constant(np.nan, dtype = tf.float32)
            img = tf.where(tf.equal(image, zero), nan, image)
            aux = (img - mean)/std
            new = tf.where(tf.math.is_nan(aux), tf.zeros_like(aux), aux)
            return new, label
        func = z_norm
    
    else:   # simple normalisation between 0 and 1
        def simple_norm(image, label):
            new = image/np.max(image)
            return new, label
        func = simple_norm
    
    train_norm_ds = train_ds.map(func, num_parallel_calls=AUTOTUNE)
    val_norm_ds = val_ds.map(func, num_parallel_calls=AUTOTUNE)
    return train_norm_ds, val_norm_ds
    
    
def make_gradcam_heatmap(img_array, model, args):
    if args['finetune']:
        if args['cnn'] == "ResNet":
            last_conv_layer_name = model.get_layer('resnet50').layers[-1].name
            last_conv_layer = model.get_layer('resnet50').get_layer(last_conv_layer_name)
            inputs = model.get_layer('resnet50').inputs
            
        else: #MobileNetV2
            last_conv_layer_name = model.get_layer('mobilenetv2_1.00_224').layers[-1].name
            last_conv_layer = model.get_layer('mobilenetv2_1.00_224').get_layer(last_conv_layer_name)
            inputs = model.get_layer('mobilenetv2_1.00_224').inputs
    else:
        last_conv_layer_name = model.layers[-4].name if args['dropout'] else model.layers[-3].name
        last_conv_layer = model.get_layer(last_conv_layer_name)
        inputs = model.inputs

    classifier_layer_names = [layer.name for layer in model.layers[-3:]] \
        if args['dropout'] else [layer.name for layer in model.layers[-2:]]
    #print(last_conv_layer_name)
    #print(classifier_layer_names)
    #print(inputs)
    
    # First, we create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = keras.Model(inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def grad_cam(model, val_dataset, val_norm_dataset, predictions, save_path, args):
    val_norm_images = val_norm_dataset.map(lambda x, y: x)
    val_norm_labels = val_norm_dataset.map(lambda x, y: y)
    val_images = val_dataset.map(lambda x, y: x)
    val_labels = val_dataset.map(lambda x, y: y)

    cnt = 0
    for orig_image, label, processed_image, processed_label, prediction in \
        zip(val_images, val_labels, val_norm_images, val_norm_labels, predictions):
        # label and processed_label should be the same
        #assert (label == processed_label).all()
        
        processed_image = tf.expand_dims(processed_image, axis=0)
        #orig_image = tf.transpose(orig_image, [1,0,2])
        #processed_image = tf.transpose(processed_image, [1,0,2])
        
        #pred = model.predict(processed_image)
        prediction = np.round(prediction, 2) #if categorization else pred[0]

        # compute heatmap
        heatmap = make_gradcam_heatmap(processed_image, model, args)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((orig_image.shape[1], orig_image.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        jet_heatmap_img = keras.preprocessing.image.array_to_img(jet_heatmap)
        
        # Superimpose the heatmap on original image
        vv_channel = np.uint8(np.dstack((orig_image[:,:,0], orig_image[:,:,0], orig_image[:,:,0])))
        vh_channel = np.uint8(np.dstack((orig_image[:,:,1], orig_image[:,:,1], orig_image[:,:,1])))
        superimposed_img = jet_heatmap * 0.4 + vv_channel
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        #print("vv_channel:", np.max(vv_channel), np.min(vv_channel))
        #print("vh_channel:", np.max(vh_channel), np.min(vh_channel))
        #print("superimposed_img:", np.max(superimposed_img), np.min(superimposed_img))

        # Display Grad CAM
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15,9.2))
        font_size = 18
        ax1.imshow(vv_channel)    # plot vv channel
        ax1.set_title("Original Image (VV) - label: {}".format(label), fontsize = font_size)
        ax2.imshow(vh_channel)  # plot vh channel
        ax2.set_title("Original Image (VH)", fontsize = font_size)
        heat_plot = ax3.imshow(jet_heatmap_img, cmap='jet')
        ax3.set_title("Heatmap - prediction: {}".format(prediction), fontsize = font_size)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(heat_plot, cax=cax)
        ax4.imshow(superimposed_img)
        ax4.set_title("Superimposed Image", fontsize = font_size)
        fig.tight_layout()
        #plt.show()

        fig.savefig("{}/heatmap_{}.jpg".format(save_path, cnt), bbox_inches='tight')
        plt.close(fig)
        cnt += 1
    return
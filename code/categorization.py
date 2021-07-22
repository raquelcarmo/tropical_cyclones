import helper_functions as hp
from models import make_cnn
from data_processor import DataProcessor
from visualization import plot_history, plot_history_ft

import os
import re
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall, TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def parse_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', required=True, type=str, help='Main directory where images and CSVs are stored')
    parser.add_argument('--model', default='ResNet', type=str, choices=['ResNet', 'Mobile', 'VGG'], help='Convolutional Neural Network to train')
    parser.add_argument('--height', default=416, type=int, help='Input image height')
    parser.add_argument('--width', default=416, type=int, help='Input image width')
    parser.add_argument('--eye-only', action='store_true', help='Whether to consider only images with eyes. True if specified')
    parser.add_argument('--num-vars', action='store_true', help='Whether to use also numerical variables. True if specified')
    parser.add_argument('--norm', action='store_true', help='Whether to perform normalisation to the images. True if specified')
    parser.add_argument('--norm-mode', default='model', type=str, choices=['z-norm', 'model', 'simple', 'none'], help='Normalization mode to apply to images')
    parser.add_argument('--rot', action='store_true', help='Whether to rotate images. True if specified')
    parser.add_argument('--crop', action='store_true', help='Whether to perform random cropping to the images. True if specified')
    parser.add_argument('--crop-mode', default='uniform', type=str, choices=['uniform', 'weighted'], help='Random crop mode to apply to images')
    parser.add_argument('--nb-crops', default=1, type=int, help='Number of random crops to perform')
    parser.add_argument('--aug', action='store_true', help='Whether to perform data augmentation. True if specified')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--buffer-size', default=100, type=int, help='Buffer size')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate to use in training')
    parser.add_argument('--splits', default=5, type=int, help='Number of splits to perform in StratifiedKFold')
    parser.add_argument('--dropout', action='store_true', help='Whether to use dropout in CNNs. True if specified')
    parser.add_argument('--drop-rate', default=0.5, type=float, help='dropout rate to apply if dropout == True')
    parser.add_argument('--fine-tune', required=True, action='store_true', help='Whether to perform fine-tuning. True if specified')
    parser.add_argument('--fine-tune-at', default='last5', help='Options: int - Layer to start fine-tuning from; str - Number of last layers to fine-tune: "lastX"')
    parser.add_argument('--initial-epochs', default=20, type=int, help='Initial epochs')
    parser.add_argument('--ft-epochs', default=10, type=int, help='Epochs for fine-tuning stage')

    #args = vars(parser.parse_args())
    args = parser.parse_args()
    return args


def train_categorization(args):
    ''' STRATIFIED CROSS VALIDATION '''
    network = args.model
    height = args.height
    width = args.width
    batch_size = args.batch_size
    eye_only = args.eye_only
    crop_mode = args.crop_mode
    dropout = args.dropout

    val_acc = []
    val_top2_acc = []
    val_loss = []
    val_tp = []
    val_fp = []
    val_tn = []
    val_fn = []
    val_prec = []
    val_rec = []

    full_dataset_path = "{}/csv/full_dataset.csv".format(args.data_path)
    df = pd.read_csv(full_dataset_path, converters={'bbox_shape': eval}).dropna()
    print("Dataset dimension: {}".format(len(df)))

    # COMPUTE CLASS WEIGHTS
    # if crop_mode == weighted (proporional to presence of classes), hp.compute_class_weights()
    # will no longer represent the true distribution of classes in the dataset
    class_weights = hp.compute_class_weights(full_dataset_path, eye_only = eye_only) if crop_mode != "weighted" else None

    # extract non-categorical labels for StratifiedKFold
    Y = np.zeros(len(df), dtype=int)
    for index, row in df.iterrows():
        if df["label"][index] != 0:
            cat = df["image"][index].split('/')[1]
            Y[index] = int(cat[-1])
    
    if eye_only:
        assert isinstance(Y, np.ndarray)
        idx = np.where(Y == 0)[0].tolist()
        Y = np.delete(Y, idx)
        Y -= 1
        df.drop(idx, inplace = True)
        df.reset_index(drop=True, inplace=True)
        print("New dimensions, Y: {} and df: {}".format(len(Y), len(df)))

    # directory to save results
    dir = '{}_nu-{}_bs-{}_{}x{}_lr-{}_ep-{}_sp-{}_no-{}{}_cr-{}{}_ag-{}_drp-{}{}'.format(
        network, str(args.num_vars)[0], batch_size, height, width, args.learning_rate, args.epochs, args.splits, 
        args.norm_mode[0], str(args.normalise)[0], crop_mode[0], args.nb_crops, str(args.aug)[0], str(dropout)[0], args.drop_rate)
    save_dir = args.data_path + '/classification_results/categorization/test_eye_only/' + dir + '/'
    os.makedirs(save_dir, exist_ok=True)

    # store run arguments in the save_dir folder
    with open(os.path.join(save_dir, 'run_arguments.json'), 'w') as outfile:
        json.dump(vars(args), outfile)

    # create an instance of the DataProcessor
    processor = DataProcessor(model = network, min_height = height, min_width = width, normalise = args.normalise, rotate = args.rot)

    print("Entering in k fold cross validation...")
    stratified_k_fold = StratifiedKFold(n_splits = args.splits, shuffle = False)
    fold_var = 1

    for train_index, val_index in stratified_k_fold.split(np.zeros(len(df)), Y):
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]

        # LOAD DATA
        train_images, train_labels, train_bbox = hp.load_data(df = training_data, eye_only = eye_only)
        val_images, val_labels, val_bbox = hp.load_data(df = validation_data, eye_only = eye_only)

        # GENERATE DATASETS
        #train_dataset = hp.prepare_dataset(processor, train_images, train_labels, train_bbox)
        #val_dataset = hp.prepare_dataset(processor, val_images, val_labels, val_bbox)
        train_dataset = hp.create_dataset(processor, train_images, train_labels, train_bbox, args.nb_crops, crop_mode, height, width)
        val_dataset = hp.create_dataset(processor, val_images, val_labels, val_bbox, 1, crop_mode, height, width)

        # PERFORM NORMALISATION
        if args.normalise:
            train_norm_dataset, val_norm_dataset = hp.normalisation(train_dataset, val_dataset, mode = args.norm_mode, model = network)
        else:
            train_norm_dataset = train_dataset
            val_norm_dataset = val_dataset
        unbatched_train_norm_dataset, unbatched_val_norm_dataset = train_norm_dataset, val_norm_dataset

        # CONFIGURE FOR PERFORMANCE
        SQUARED = True if height == width else False
        train_norm_dataset = hp.configure_for_performance(train_norm_dataset, args.buffer_size, batch_size, shuffle = True, augment = args.aug, squared_input = SQUARED)
        val_norm_dataset = hp.configure_for_performance(val_norm_dataset, args.buffer_size, batch_size)

        # CREATE NEW MODEL
        model = make_cnn(network, width, height, args.learning_rate, eye_only = eye_only, dropout = dropout, rate = args.drop_rate)

        # CREATE CALLBACKS
        callbacks = [
            EarlyStopping(patience = 15, verbose = 1),
            ReduceLROnPlateau(factor = 0.1, patience = 5, min_lr = 0.000001, verbose = 1),
            ModelCheckpoint(save_dir + hp.get_model_name(fold_var), verbose = 1, save_best_only = True),
            #TensorBoard(log_dir = save_dir + "logs/" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        ]

        # FIT THE MODEL
        history = model.fit(
            train_norm_dataset,
            steps_per_epoch = len(train_norm_dataset),
            validation_data = val_norm_dataset,
            validation_steps = len(val_norm_dataset), 
            epochs = args.epochs,
            callbacks = callbacks,
            verbose = 1, 
            class_weight = class_weights,
            shuffle = True)
        
        # PLOT TRAIN/VALIDATION LOSSES
        plot_history(history, fold_var, save_dir)
        
        # LOAD BEST MODEL
        time.sleep(10) # guarantees enough time for weights to be saved and loaded afterwards, otherwise gives concurrency problems
        print("Loaded best weights of the training")
        model.load_weights(save_dir + hp.get_model_name(fold_var))

        # EVALUATE PERFORMANCE of the model
        results = model.evaluate(val_norm_dataset, steps = len(val_norm_dataset))

        results = dict(zip(model.metrics_names, results))
        val_acc.append(results['accuracy'])
        val_top2_acc.append(results['top2_accuracy'])
        val_loss.append(results['loss'])
        val_tp.append(results["tp"])
        val_fp.append(results["fp"])
        val_tn.append(results["tn"])
        val_fn.append(results["fn"])
        val_prec.append(results["precision"])
        val_rec.append(results["recall"])

        # MAKE PREDICTIONS
        predictions = model.predict(val_norm_dataset)
        predictions_non_category = [ np.argmax(t) for t in predictions ]
        val_labels = val_dataset.map(lambda x, y: y)
        val_labels_non_category = [ np.argmax(t) for t in val_labels ]

        labels = [0,1,2,3,4,5] if not eye_only else [0,1,2,3,4]
        display_labels = labels if not eye_only else [x+1 for x in labels]
        conf_mat = confusion_matrix(val_labels_non_category, predictions_non_category, labels = labels)
        disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = display_labels)
        conf_mat_display = disp.plot()
        plt.savefig(save_dir + "confusion_matrix_" + str(fold_var) + ".jpg", bbox_inches='tight')

        # GRAD-CAM ANALYSIS
        train_gradcam_path = save_dir + "train_gradcam_heatmaps_{}".format(fold_var)
        os.makedirs(train_gradcam_path, exist_ok=True)
        gradcam_train_norm_dataset = hp.configure_for_performance(unbatched_train_norm_dataset, args.buffer_size, batch_size)
        hp.grad_cam(model, train_dataset, unbatched_train_norm_dataset, model.predict(gradcam_train_norm_dataset), train_gradcam_path, dropout = dropout)

        gradcam_path = save_dir + "gradcam_heatmaps_{}".format(fold_var)
        os.makedirs(gradcam_path, exist_ok=True)
        hp.grad_cam(model, val_dataset, unbatched_val_norm_dataset, predictions, gradcam_path, dropout = dropout)

        tf.keras.backend.clear_session()

        fold_var += 1


    metrics = {'val_acc':val_acc, 'val_top2_acc':val_top2_acc, 'val_loss':val_loss, 'val_tp':val_tp, 
                'val_fp':val_fp, 'val_tn':val_tn, 'val_fn':val_fn, 'val_prec':val_prec, 'val_rec':val_rec}

    # save the values for each fold
    csv_dir = save_dir + 'csv/'
    os.makedirs(csv_dir, exist_ok=True)
    hp.save_metrics(csv_dir, metrics)        
    return


def train_categorization_ft(args):
    ''' STRATIFIED CROSS VALIDATION WITH TRANSFER LEARNING AND FINE-TUNING '''
    network = args.model
    height = args.height
    width = args.width
    batch_size = args.batch_size
    eye_only = args.eye_only
    crop_mode = args.crop_mode
    dropout = args.dropout

    val_acc = []
    val_top2_acc = []
    val_loss = []
    val_tp = []
    val_fp = []
    val_tn = []
    val_fn = []
    val_prec = []
    val_rec = []

    full_dataset_path = "{}/csv/full_dataset.csv".format(args.data_path)
    df = pd.read_csv(full_dataset_path, converters={'bbox_shape': eval}).dropna()
    print("Dataset dimension: {}".format(len(df)))

    # COMPUTE CLASS WEIGHTS
    # if crop_mode == weighted (proporional to presence of classes), hp.compute_class_weights()
    # will no longer represent the true distribution of classes in the dataset
    class_weights = hp.compute_class_weights(full_dataset_path, eye_only = eye_only) if crop_mode != "weighted" else None

    # extract non-categorical labels for StratifiedKFold
    Y = np.zeros(len(df), dtype=int)
    for index, row in df.iterrows():
        if df["label"][index] != 0:
            cat = df["image"][index].split('/')[1]
            Y[index] = int(cat[-1])

    if eye_only:
        assert isinstance(Y, np.ndarray)
        idx = np.where(Y == 0)[0].tolist()
        Y = np.delete(Y, idx)
        Y -= 1
        df.drop(idx, inplace = True)
        df.reset_index(drop=True, inplace=True)
        #print(df)
        print("New dimensions, Y: {} and df: {}".format(len(Y), len(df)))
    
    # directory to save results
    total_epochs = args.initial_epochs + args.ft_epochs
    dir = '{}_nu-{}_bs-{}_{}x{}_lr-{}_ep-{}_sp-{}_no-{}{}_cr-{}{}_ag-{}_drp-{}{}_ft-{}'.format(
        network, str(args.num_vars)[0], batch_size, height, width, args.learning_rate, total_epochs, args.splits, 
        args.norm_mode[0], str(args.normalise)[0], crop_mode[0], args.nb_crops, str(args.aug)[0], str(dropout)[0], args.drop_rate, args.fine_tune_at)
    save_dir = args.data_path + '/classification_results/categorization/test_eye_only/' + dir + '/'

    # create an instance of the DataProcessor
    processor = DataProcessor(model = network, min_height = height, min_width = width, normalise = args.normalise, rotate = args.rot)

    print("Entering in k fold cross validation...")
    stratified_k_fold = StratifiedKFold(n_splits = args.splits, shuffle = False)    
    fold_var = 1

    for train_index, val_index in stratified_k_fold.split(np.zeros(len(df)), Y):
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]

        # LOAD DATA
        train_images, train_labels, train_bbox = hp.load_data(df = training_data, eye_only = eye_only)
        val_images, val_labels, val_bbox = hp.load_data(df = validation_data, eye_only = eye_only)
        
        # GENERATE DATASETS
        #train_dataset = hp.prepare_dataset(processor, train_images, train_labels, train_bbox)
        #val_dataset = hp.prepare_dataset(processor, val_images, val_labels, val_bbox)
        train_dataset = hp.create_dataset(processor, train_images, train_labels, train_bbox, args.nb_crops, crop_mode, height, width)
        val_dataset = hp.create_dataset(processor, val_images, val_labels, val_bbox, 1, crop_mode, height, width)

        # PERFORM NORMALISATION
        if args.normalise:
            train_norm_dataset, val_norm_dataset = hp.normalisation(train_dataset, val_dataset, mode = args.norm_mode, model = network)
        else:
            train_norm_dataset = train_dataset
            val_norm_dataset = val_dataset
        unbatched_train_norm_dataset, unbatched_val_norm_dataset = train_norm_dataset, val_norm_dataset

        # CONFIGURE FOR PERFORMANCE
        SQUARED = True if height == width else False
        train_norm_dataset = hp.configure_for_performance(train_norm_dataset, args.buffer_size, batch_size, shuffle = True, augment = args.aug, squared_input = SQUARED)
        val_norm_dataset = hp.configure_for_performance(val_norm_dataset, args.buffer_size, batch_size)
        
        # CREATE BASE PRE-TRAINED MODEL
        if network == "ResNet":
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(width, height, 3))
        elif network == "Mobile":
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(width, height, 3))
        else:
            sys.exit("Incert valid network model. Options: Mobile or ResNet (case sensitive)")

        # FREEZE BASE MODEL
        base_model.trainable = False

        inputs = Input(shape=(width, height, 3))
        x = base_model(inputs, training = False)
        x = GlobalAveragePooling2D()(x)
        if dropout:
            x = dropout(args.drop_rate)(x)
        classes = 6 if not eye_only else 5
        outputs = Dense(classes, activation="softmax")(x)
        model = Model(inputs, outputs)

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer = Adam(learning_rate = args.learning_rate), 
                      loss = "categorical_crossentropy",
                      metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                                 Precision(name="precision"), Recall(name="recall"), 
                                 TruePositives(name='tp'), FalsePositives(name='fp'),
                                 TrueNegatives(name='tn'), FalseNegatives(name='fn')])

        # CREATE CALLBACKS
        callbacks = [
            ModelCheckpoint(save_dir + "best_model_frozen_{}.h5".format(str(fold_var)), verbose = 1, save_best_only = True),
            #TensorBoard(log_dir = save_dir + "logs/" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        ]

        # FIT THE MODEL
        # train the top layer of the model on the dataset, the weights of the pre-trained network will not be updated during training
        history = model.fit(train_norm_dataset,
                            steps_per_epoch = len(train_norm_dataset),
                            validation_data = val_norm_dataset,
                            validation_steps = len(val_norm_dataset),
                            epochs = args.initial_epochs,
                            callbacks = callbacks,
                            class_weight = class_weights,
                            shuffle = True)

        acc = history.history['accuracy']
        validation_acc = history.history['val_accuracy']

        loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # UNFREEZE BASE MODEL
        base_model.trainable = True

        # Fine-tune from this layer onwards
        if isinstance(args.fine_tune_at, int):
            # Freeze all the layers before the 'FINE_TUNE_AT' layer
            for layer in base_model.layers[:args.fine_tune_at]:
                layer.trainable = False
            
        else: #string
            if "last" in args.fine_tune_at:
                nb_layers_to_fine_tune = int(re.findall(r'\d+', args.fine_tune_at)[0])
                print("nb_layers_to_fine_tune:", nb_layers_to_fine_tune)
                total_to_freeze = len(base_model.layers) - nb_layers_to_fine_tune
                print("total_to_freeze:", total_to_freeze)
                for layer in base_model.layers[:total_to_freeze]:
                    layer.trainable = False


        # recompile the model for the modifications to take effect, with a low learning rate
        model.compile(optimizer = Adam(learning_rate = 1e-5),
                      loss = "categorical_crossentropy",
                      metrics = [CategoricalAccuracy(name="accuracy"), TopKCategoricalAccuracy(k=2, name="top2_accuracy"),
                                 Precision(name="precision"), Recall(name="recall"), 
                                 TruePositives(name='tp'), FalsePositives(name='fp'),
                                 TrueNegatives(name='tn'), FalseNegatives(name='fn')])
        #base_model.summary()
        
        # adjust callbacks
        callbacks = [
            ModelCheckpoint(save_dir + "best_model_fine_tuned_{}.h5".format(str(fold_var)), verbose = 1, save_best_only = True),
            #TensorBoard(log_dir = save_dir + "logs/" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
        ]

        # train the entire model end-to-end
        history_fine = model.fit(train_norm_dataset,
                                 steps_per_epoch = len(train_norm_dataset),
                                 validation_data = val_norm_dataset,
                                 validation_steps = len(val_norm_dataset),
                                 epochs = total_epochs,
                                 initial_epoch = history.epoch[-1],
                                 callbacks = callbacks,
                                 class_weight = class_weights,
                                 shuffle = True)

        acc += history_fine.history['accuracy']
        validation_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        validation_loss += history_fine.history['val_loss']

        # PLOT TRAIN/VALIDATION LOSSES
        history_dict = {'acc':acc, 'validation_acc':validation_acc, 'loss':loss, 'validation_loss':validation_loss}
        plot_history_ft(args, history_dict, fold_var, save_dir)            

        # LOAD BEST MODEL
        time.sleep(15)
        print("Loaded best weights of the training")
        model.load_weights(save_dir + "best_model_fine_tuned_{}.h5".format(str(fold_var)))

        # EVALUATE PERFORMANCE of the model
        results = model.evaluate(val_norm_dataset, steps = len(val_norm_dataset))

        results = dict(zip(model.metrics_names, results))
        val_acc.append(results['accuracy'])
        val_top2_acc.append(results['top2_accuracy'])
        val_loss.append(results['loss'])
        val_tp.append(results["tp"])
        val_fp.append(results["fp"])
        val_tn.append(results["tn"])
        val_fn.append(results["fn"])
        val_prec.append(results["precision"])
        val_rec.append(results["recall"])

        # MAKE PREDICTIONS
        predictions = model.predict(val_norm_dataset)
        predictions_non_category = [ np.argmax(t) for t in predictions ]
        val_labels = val_dataset.map(lambda x, y: y)
        val_labels_non_category = [ np.argmax(t) for t in val_labels ]

        labels = [0,1,2,3,4,5] if not eye_only else [0,1,2,3,4]
        display_labels = labels if not eye_only else [x+1 for x in labels]
        conf_mat = confusion_matrix(val_labels_non_category, predictions_non_category, labels = labels)
        disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = display_labels)
        conf_mat_display = disp.plot()
        plt.savefig(save_dir + "confusion_matrix_" + str(fold_var) + ".jpg", bbox_inches='tight')

        # GRAD-CAM ANALYSIS
        train_gradcam_path = save_dir + "train_gradcam_heatmaps_{}".format(fold_var)
        os.makedirs(train_gradcam_path, exist_ok=True)
        gradcam_train_norm_dataset = hp.configure_for_performance(unbatched_train_norm_dataset, args.buffer_size, batch_size)
        hp.grad_cam(model, train_dataset, unbatched_train_norm_dataset, model.predict(gradcam_train_norm_dataset), train_gradcam_path, dropout = dropout, fine_tuning = True, network = network)

        gradcam_path = save_dir + "gradcam_heatmaps_{}".format(fold_var)
        os.makedirs(gradcam_path, exist_ok=True)
        hp.grad_cam(model, val_dataset, unbatched_val_norm_dataset, predictions, gradcam_path, dropout = dropout, fine_tuning = True, network = network)

        tf.keras.backend.clear_session()

        fold_var += 1

    metrics = {'val_acc':val_acc, 'val_top2_acc':val_top2_acc, 'val_loss':val_loss, 'val_tp':val_tp, 
                'val_fp':val_fp, 'val_tn':val_tn, 'val_fn':val_fn, 'val_prec':val_prec, 'val_rec':val_rec}

    # save metrics for each fold
    csv_dir = save_dir + 'csv/'
    os.makedirs(csv_dir, exist_ok=True)
    hp.save_metrics(csv_dir, metrics)
    return



if __name__ == "__main__":
    args = parse_args()
    if args.fine_tune:
        train_categorization_ft(args)
    else:
        train_categorization(args)
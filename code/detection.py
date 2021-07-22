import helper_functions as hp
from models import make_cnn
from data_processor import DataProcessor
from visualization import plot_history

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def parse_args():
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', required=True, type=str, help='Main directory where images and CSVs are stored')
    parser.add_argument('--model', default='ResNet', type=str, choices=['ResNet', 'Mobile', 'VGG'], help='Convolutional Neural Network to train')
    parser.add_argument('--height', default=416, type=int, help='Input image height')
    parser.add_argument('--width', default=416, type=int, help='Input image width')
    parser.add_argument('--num-vars', action='store_true', help='Whether to use also numerical variables. True if specified')
    parser.add_argument('--norm', action='store_true', help='Whether to perform normalisation to the images. True if specified')
    parser.add_argument('--norm-mode', default='model', type=str, choices=['z-norm', 'model', 'simple', 'none'], help='Normalization mode to apply to images')
    parser.add_argument('--rot', action='store_true', help='Whether to rotate images. True if specified')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size')
    parser.add_argument('--buffer-size', default=100, type=int, help='Buffer size')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate to use in training')
    parser.add_argument('--splits', default=5, type=int, help='Number of splits to perform in StratifiedKFold')

    #args = vars(parser.parse_args())
    args = parser.parse_args()
    return args


def train_detection(args):
    network = args.model
    height = args.height
    width = args.width
    batch_size = args.batch_size

    val_acc = []
    val_loss = []
    val_tp = []
    val_fp = []
    val_tn = []
    val_fn = []
    val_prec = []
    val_rec = []

    df = pd.read_csv(args.data_path + "/csv/full_dataset.csv", converters={'bbox_shape': eval}).dropna()
    print("Dataset dimension: {}".format(len(df)))
    Y = df["label"]

    # directory to save results
    dir = '{}_nu-{}_bs-{}_{}x{}_lr-{}_ep-{}_sp-{}_no-{}{}'.format(
        network, str(args.num_vars)[0], batch_size, height, width, args.learning_rate, args.epochs, args.splits, args.norm_mode[0], str(args.norm)[0])
    save_dir = args.data_path + '/classification_results/identification/' + dir + '/'
    os.makedirs(save_dir, exist_ok=True)

    # store run arguments in the save_dir folder
    with open(os.path.join(save_dir, 'run_arguments.json'), 'w') as outfile:
        json.dump(vars(args), outfile)
    
    # create an instance of the DataProcessor
    processor = DataProcessor(model = network, min_height = height, min_width = width, rotate = args.rot)
    
    print("Entering in k fold cross validation...")
    stratified_k_fold = StratifiedKFold(n_splits = args.splits, random_state = 42, shuffle = False)
    fold_var = 1

    for train_index, val_index in stratified_k_fold.split(np.zeros(len(df)), Y):
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]

        train_images, train_labels, train_bbox = hp.load_data(df = training_data, with_cat = False)
        val_images, val_labels, val_bbox = hp.load_data(df = validation_data, with_cat = False)
        
        # generate datasets
        #train_dataset = hp.prepare_dataset(processor, train_images, train_labels, train_bbox)
        #val_dataset = hp.prepare_dataset(processor, val_images, val_labels, val_bbox)
        train_dataset = hp.create_dataset(processor, train_images, train_labels, train_bbox, 1, 'uniform', height, width)
        val_dataset = hp.create_dataset(processor, val_images, val_labels, val_bbox, 1, 'uniform', height, width)

        if args.norm:
            train_norm_dataset, val_norm_dataset = hp.normalisation(train_dataset, val_dataset, mode = args.norm_mode, model = network)
        else:
            train_norm_dataset, val_norm_dataset = train_dataset, val_dataset
        #unbatch_sizeed_train_norm_dataset, unbatch_sizeed_val_norm_dataset = train_norm_dataset, val_norm_dataset

        # configure for performance
        train_norm_dataset = hp.configure_for_performance(train_norm_dataset, args.buffer_size, batch_size, shuffle = True)
        val_norm_dataset = hp.configure_for_performance(val_norm_dataset, args.buffer_size, batch_size)

        # CREATE NEW MODEL
        model = make_cnn(network, width, height, args.learning_rate, detection = True)

        # CREATE CALLBACKS
        callbacks = [
            EarlyStopping(patience = 10, verbose = 1),
            ReduceLROnPlateau(factor = 0.1, patience = 5, min_lr = 0.00001, verbose = 1),
            ModelCheckpoint(save_dir + hp.get_model_name(fold_var), verbose = 1, save_best_only = True),
            TensorBoard(log_dir = save_dir + "logs/" + datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
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
            class_weight = {0:1.6, 1:1},
            shuffle = True)


        # PLOT TRAIN/VALIDATION LOSSES
        plot_history(history, fold_var, save_dir)
    
        # guarantee enough time so that weights are saved and can be loaded after
        time.sleep(12) 

        # LOAD BEST MODEL
        print("Loaded best weights of the training")
        model.load_weights(save_dir + hp.get_model_name(fold_var))

        # EVALUATE PERFORMANCE of the model
        results = model.evaluate(val_norm_dataset, steps = len(val_norm_dataset))

        results = dict(zip(model.metrics_names, results))
        val_acc.append(results['accuracy'])
        val_loss.append(results['loss'])
        val_tp.append(results["tp"])
        val_fp.append(results["fp"])
        val_tn.append(results["tn"])
        val_fn.append(results["fn"])
        val_prec.append(results["precision"])
        val_rec.append(results["recall"])

        # MAKE PREDICTIONS
        predictions = model.predict(val_norm_dataset)
        predictions = tf.where(predictions < 0.5, 0, 1)

        conf_mat = confusion_matrix(val_labels, predictions, labels = [0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = [0,1])
        conf_mat_display = disp.plot()
        plt.savefig(save_dir + "confusion_matrix_" + str(fold_var) + ".jpg", bbox_inches='tight')

        tf.keras.backend.clear_session()

        fold_var += 1

    metrics = {'val_acc':val_acc, 'val_loss':val_loss, 'val_tp':val_tp, 'val_fp':val_fp, 
                'val_tn':val_tn, 'val_fn':val_fn, 'val_prec':val_prec, 'val_rec':val_rec}

    # save the values for each fold
    csv_dir = save_dir + 'csv/'
    os.makedirs(csv_dir, exist_ok=True)
    hp.save_metrics(csv_dir, metrics)        
    return


if __name__ == "__main__":
    train_detection(parse_args())
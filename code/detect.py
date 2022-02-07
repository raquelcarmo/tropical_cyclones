# import modules
import utils
from models import DetectionCNN
from data_process import DataProcessor
from visualization import plot_history

# general imports
import os
import yaml
import time
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf


def train_detection(args):
    features = args['data']['features']
    args = args['detection']

    # directory specific to the selected features
    folderName = ''
    for idx, item in zip(range(len(features)), features):
        folderName += '{}_'.format(item) if idx != 2 else item
    #print(folderName)
    feats_path = os.path.join(args['data_path'], folderName)

    dataset_path = f"{feats_path}/csv/full_dataset.csv"
    df = pd.read_csv(dataset_path, converters={'bbox_shape': eval}).dropna()
    #print("Dataset dimension: {}".format(len(df)))
    Y = df["label"].values

    # directory to save results
    save_dir = os.path.join(feats_path, args['results_path'])
    args['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # store run arguments in the save_dir folder
    # with open(os.path.join(save_dir, 'run_arguments.json'), 'w') as outfile:
    #     json.dump(vars(args), outfile)
    
    # create an instance of the DataProcessor
    p = DataProcessor(args)

    # create an instance of the model
    detNet = DetectionCNN(args)
    
    print("Entering in K-fold Cross Validation...")
    stratified_k_fold = StratifiedKFold(n_splits=args['nb_splits'], shuffle=False)
    fold_var = 1

    for train_index, val_index in stratified_k_fold.split(np.zeros(len(df)), Y):
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]

        # load data
        train_images, train_labels, train_bbox = utils.load_data(args, df=training_data)
        val_images, val_labels, val_bbox = utils.load_data(args, df=validation_data)
        
        # generate datasets
        train_ds = utils.create_dataset(p, train_images, train_labels, train_bbox, args)
        val_ds = utils.create_dataset(p, val_images, val_labels, val_bbox, args, flag=True)

        # perform normalisation
        train_ds_norm, val_ds_norm = utils.normalisation(train_ds, val_ds, args)

        # configure for performance
        train_ds_perf = utils.config_performance(train_ds_norm, args, shuffle=True)
        val_ds_perf = utils.config_performance(val_ds_norm, args, flag=True)

        # train model
        history = detNet.train(train_ds_perf, val_ds_perf, fold_var)

        # plot train/val losses
        plot_history(history, fold_var, save_dir)
    
        # guarantee time for weights to be saved and loaded again
        time.sleep(10)

        print("Loading best weights from training...")
        detNet.get_eval(val_ds_perf, fold_var)
        detNet.get_preds(val_ds_perf, np.array([tfds.as_numpy(label) for image, label in val_ds]), fold_var)

        # reset model and clear session
        detNet.reset()
        fold_var += 1

    # save the values for each fold
    detNet.save_metrics()
    return


if __name__ == "__main__":
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            train_detection(config)
        except yaml.YAMLError as exe:
            print(exe)
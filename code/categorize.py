# import modules
import utils
from models import CategorizationCNN
from data_process import DataProcessor
from visualization import plot_history, plot_history_ft

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


def train_categorization(args):
    features = args['data']['features']
    args = args['categorization']

    # directory specific to the selected features
    folderName=''
    for idx, item in zip(range(len(features)), features):
        folderName += '{}_'.format(item) if idx != 2 else item
    #print(folderName)
    feats_path = os.path.join(args['data_path'], folderName)

    dataset_path = f"{feats_path}/csv/full_dataset.csv"
    Y, df, class_weights = utils.compute_class_weights(dataset_path, args)

    # directory to save results
    save_dir = os.path.join(feats_path, args['results_path'])
    args['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # store run arguments in the save_dir folder
    # with open(os.path.join(save_dir, 'run_arguments.json'), 'w') as outfile:
    #     json.dump(vars(args), outfile)

    # create an instance of the DataProcessor
    p = DataProcessor(args)

    # create and build model
    catNet = CategorizationCNN(args)

    print("Entering in K-fold Cross Validation...")
    stratified_k_fold = StratifiedKFold(n_splits=n_splits=args['nb_splits'], shuffle=False)
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

        if args['finetune']:
            # train classifier with frozen base model
            history1 = catNet.trainftStage1(train_ds_perf, val_ds_perf, class_weights, fold_var)

            acc = history1.history['accuracy']
            validation_acc = history1.history['val_accuracy']
            loss = history1.history['loss']
            validation_loss = history1.history['val_loss']

            # build and train unfrozen model
            catNet.buildftStage2()
            history2 = catNet.trainftStage2(train_ds_perf, val_ds_perf, class_weights, history1, fold_var)

            acc += history2.history['accuracy']
            validation_acc += history2.history['val_accuracy']
            loss += history2.history['loss']
            validation_loss += history2.history['val_loss']

            # plot train/val losses
            hDict = {'acc':acc, 'validation_acc':validation_acc, 
                     'loss':loss, 'validation_loss':validation_loss}
            plot_history_ft(args, hDict, fold_var, save_dir)
        else:
            history = catNet.train(train_ds_perf, val_ds_perf, class_weights, fold_var)
            # plot train/val losses
            plot_history(history, fold_var, save_dir)
        
        # guarantee time for weights to be saved and loaded again
        time.sleep(10)

        print("Loading best weights from training...")
        catNet.get_eval(val_ds_perf, fold_var)
        catNet.get_preds(val_ds_perf, np.array([tfds.as_numpy(label) for image, label in val_ds]), fold_var)

        ### Grad-CAM analysis ###
        # train dataset
        train_gradcam_path = f"{save_dir}/train_gradcam_heatmaps_{fold_var}"
        os.makedirs(train_gradcam_path, exist_ok=True)
        gradcam_train_ds_norm = utils.config_performance(train_ds_norm, args, flag=True)
        utils.grad_cam(catNet.model, train_ds, train_ds_norm, catNet.model.predict(gradcam_train_ds_norm), train_gradcam_path, args)
        # validation dataset
        gradcam_path = f"{save_dir}/gradcam_heatmaps_{fold_var}"
        os.makedirs(gradcam_path, exist_ok=True)
        utils.grad_cam(catNet.model, val_ds, val_ds_norm, catNet.model.predict(val_ds_perf), gradcam_path, args)

        # reset model and clear session
        catNet.reset()
        fold_var += 1

    # save the values for each fold
    catNet.save_metrics()        
    return


if __name__ == "__main__":
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            train_categorization(config)
        except yaml.YAMLError as exe:
            print(exe)
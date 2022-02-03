import pickle
from glob import glob
import numpy as np
import scipy
import itertools
from visualization import plot_frozen_layers, plot_model_comparisons


def mean_confidence_interval(data, confidence=0.80):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def compute_f1(dir):
    precision = np.asarray(pickle.load(open(f"{dir}/test_precision.pkl", "rb")))
    recall = np.asarray(pickle.load(open(f"{dir}/test_recall.pkl", "rb")))
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1)
    with open(f'{dir}/test_f1-score.pkl', 'wb') as handle:
        pickle.dump(f1.tolist(), handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_stats(nb_models, criticalTvalue, dir_dict, metrics):
    inter_dict = {}
    mean_dict = {}
    for metric in metrics:
        metric_dict = {}
        for i in range(1, nb_models+1):
            load_metric = pickle.load(open(f"{dir_dict[i]}/test_{metric}.pkl", "rb"))
            metric_dict['model%d_%s' %(i, metric)] = load_metric

            mean, b, u = mean_confidence_interval(load_metric)
            #print(mean, b, u)
            #conf_dict['model%d_%s' %(i, metric)] = [mean, b, u]

            if 'model%d' %(i) in inter_dict:
                inter_dict['model%d' %(i)].append(u - mean)
                mean_dict['model%d' %(i)].append(mean)
            else:
                inter_dict['model%d' %(i)] = [u - mean]
                mean_dict['model%d' %(i)] = [mean]

        #print(inter_dict)
        #print(mean_dict)
        print(f"---------------- {metric.upper()} analysis ----------------")
        for x, y in itertools.combinations(metric_dict.keys(), 2):
            t, p = scipy.stats.ttest_ind(metric_dict[x], metric_dict[y])
            print(f"t value {metric} between {x} and {y}: {t}")
            if t > criticalTvalue or t < -criticalTvalue :
                print(f"Null hypothesis rejected -> there is a significant difference in the two {metric} distributions")
            else:
                print(f"Null hypothesis not rejected -> I assume that each difference in {metric} is due to the randomness")
        print("-------------------------------------------------")
    return mean_dict, inter_dict


###### TEST ######
# Comparison between X models
main_dir = "VV_VH_WS_dilated/"
sub_dir = f"{main_dir}/results/categorization/test_eye_only/"

# note that the result (in this case the recall) could be null if the classifier is broken (all recall = 1 because is always classified as positive class)
MODEL_1_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-1e-05_ep-30_sp-5_no-zT_cr-u1_ag-T_drp-F/csv"
MODEL_2_DIRECTORY = f"{sub_dir}/test_new_weights/ResNet_nu-F_bs-8_416x416_lr-1e-05_ep-30_sp-5_no-zT_cr-u1_ag-T_drp-F/csv"
#MODEL_3_DIRECTORY = f"{sub_dir}/test_new_weights/ResNet_nu-F_bs-8_416x416_lr-1e-05_ep-30_sp-5_no-zT_cr-u1_ag-T_drp-F/csv"
#MODEL_4_DIRECTORY = f"{sub_dir}/test_rmse/ResNet_nu-F_bs-8_416x416_lr-1e-05_ep-30_sp-5_no-mT_cr-u1_ag-T_drp-F/csv"
#MODEL_5_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-0.0001_ep-35_sp-5_no-zT_cr-u1_ag-T_drp-F_ft-last50/csv"
#MODEL_6_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-0.0001_ep-35_sp-5_no-zT_cr-u1_ag-T_drp-F_ft-last75/csv"
#MODEL_7_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-0.0001_ep-35_sp-5_no-zT_cr-u1_ag-T_drp-F_ft-last125/csv"
#MODEL_8_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-0.0001_ep-35_sp-5_no-zT_cr-u1_ag-T_drp-F_ft-last175/csv"
#MODEL_9_DIRECTORY = f"{sub_dir}/ResNet_nu-F_bs-8_416x416_lr-1e-05_ep-30_sp-5_no-zT_cr-u1_ag-T_drp-F/csv"

#dir_dict = {1: MODEL_1_DIRECTORY, 2: MODEL_2_DIRECTORY, 3: MODEL_3_DIRECTORY,\
#            4: MODEL_4_DIRECTORY, 5: MODEL_5_DIRECTORY, 6: MODEL_6_DIRECTORY,\
#            7: MODEL_7_DIRECTORY, 8: MODEL_8_DIRECTORY, 9: MODEL_9_DIRECTORY}
#dir_dict = {1: MODEL_1_DIRECTORY, 2: MODEL_2_DIRECTORY, 3: MODEL_3_DIRECTORY, 4: MODEL_4_DIRECTORY, 5: MODEL_5_DIRECTORY}
#dir_dict = {1: MODEL_1_DIRECTORY, 2: MODEL_2_DIRECTORY, 3: MODEL_3_DIRECTORY, 4: MODEL_4_DIRECTORY}
#dir_dict = {1: MODEL_1_DIRECTORY, 2: MODEL_2_DIRECTORY, 3: MODEL_3_DIRECTORY}
dir_dict = {1: MODEL_1_DIRECTORY, 2: MODEL_2_DIRECTORY}

# Settings
models = len(dir_dict)
save_plot = True
title = ''
file_name = title.replace(" ", "_")

# Critical t-value to see the difference in two distribution with 5 samples
criticalTvalue = 1.533      # 80% confidence
#criticalTvalue = 2.132     # 90% confidence
#criticalTvalue = 2.776     # 95% confidence

for check_dir in list(dir_dict.values()):
    print(check_dir.split('/')[0])
    file = glob(f'{check_dir}/test_f1-score.pkl')

    if file == []:
        # compute F1-score if not already computed previously
        compute_f1(check_dir)
    #check_dir = dir_dict[9]

    acc = np.asarray(pickle.load(open(f"{check_dir}/test_accuracy.pkl", "rb")))
    prec = np.asarray(pickle.load(open(f"{check_dir}/test_precision.pkl", "rb")))
    rec = np.asarray(pickle.load(open(f"{check_dir}/test_recall.pkl", "rb")))
    f1 = np.asarray(pickle.load(open(f"{check_dir}/test_f1-score.pkl", "rb")))
    
    print("Accuracy:", acc, np.mean(acc), np.std(acc))
    print("Precision:", prec, np.mean(prec), np.std(prec))
    print("Recall:", rec, np.mean(rec), np.std(rec))
    print("F1-score:", f1, np.mean(f1), np.std(f1))
    print('\n')


# metrics to use for analysis
metrics = ["accuracy", "precision", "recall", "f1-score"]
# compute mean and confidence intervals for each metric and each model
mean_dict, inter_dict = compute_stats(models, criticalTvalue, dir_dict, metrics)
# plot model comparison
plot_model_comparisons(models, metrics, dir_dict, mean_dict, inter_dict, sub_dir, title, save_plot)
# plot scores vs. nb of frozen layers
plot_frozen_layers(models, metrics, dir_dict, mean_dict, inter_dict, sub_dir, title, save_plot)

import numpy as np
import os

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, accuracy_score
from utils.helper import load_single_pkl


def metric_sequence(final_result_list):
    # evalute the predict sequences (the sequences have different lengths)
    # min_max_scaler = preprocessing.MinMaxScaler()
    for fold in final_result_list:
        subject, subject_result = fold

        single_subject_mse, single_subject_pearson, trial_count = 0, 0, 0
        for batch_result in subject_result:
            single_batch = batch_result[0]
            single_batch_inference = batch_result[1]
            single_batch_label = batch_result[2]
            single_batch_mask = batch_result[3]

            single_batch_mse, single_batch_pearson = 0, 0
            for inference, label, mask in zip(single_batch_inference, single_batch_label, single_batch_mask):
                inference, label, mask = np.asarray(inference), np.asarray(label), np.asarray(mask)
                indices = np.where(mask == 1)
                single_batch_mse += mean_squared_error(inference[indices], label[indices])
                single_batch_pearson += pearsonr(inference[indices].squeeze(), label[indices].squeeze())[0]
                trial_count += 1
            single_batch_mse = single_batch_mse / len(single_batch_inference)
            single_batch_pearson = single_batch_pearson / len(single_batch_inference)

            single_subject_mse += single_batch_mse
            single_subject_pearson += single_batch_pearson

        single_subject_mse = np.round(single_subject_mse / len(subject_result), 4)
        single_subject_pearson = np.round(single_subject_pearson / len(subject_result), 4)

        print("Subject: {}, Trial_count: {}, MSE = {}, Pearson = {}.".format(
            str(subject), str(trial_count), str(single_subject_mse), str(single_subject_pearson)))

    return  0 # np.mean(mse), np.sqrt(np.mean(mse)), np.mean(pearson)

if __name__ == "__main__":
    directory = os.path.join("..", "load")
    result = load_single_pkl(directory, "final_result_list", ".pkl")
    metric_sequence(result)
import mne
import statistics

# from scipy.spatial import ConvexHull
from utils.landmark_template import facial_landmark_template
from utils.helper import load_single_pkl, load_pkl_file, dict_combine

import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
from operator import itemgetter
import cv2
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


class CKplusArrangerStatic(object):
    def __init__(self, config, num_folds):
        self.root_directory = config['remote_root_directory']
        self.num_folds = num_folds
        self.emotion_dict = self.init_emotion_dict()
        self.subject_list = self.get_subject_list()

    def init_emotion_dict(self):
        return {'Neutral': 0, 'Anger': 1, 'Contempt': 2, 'Disgust': 3,
                'Fear': 4, 'Happiness': 5, 'Sad': 6, 'Surprise': 7}

    def establish_fold(self):
        foldwise_subject_count = self.count_subject_for_each_fold()
        fold_list = [[] for i in range(self.num_folds)]

        start = 0
        for i in range(self.num_folds):
            end = start + foldwise_subject_count[i]
            subject_id_in_this_fold = list(itemgetter(*range(start, end))(self.subject_list))

            for subject_id in subject_id_in_this_fold:
                subject_directory = os.path.join(self.root_directory, subject_id)

                for path in Path(subject_directory).rglob('*.csv'):
                    label = self.emotion_dict[path.name.split(".csv")[0]]
                    image_directory = str(path).split(".csv")[0] + "_aligned"
                    peak_image_directory = [str(image) for image in Path(image_directory).rglob('*.jpg')][-1]
                    fold_list[i].append([label, peak_image_directory])

            start = end

        return fold_list

    def get_subject_list(self):
        subject_list = [folder for folder in os.listdir(self.root_directory)]
        return subject_list

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [12, 12, 12, 12, 12, 12, 12, 12, 12, 10]
        return foldwise_subject_count


class RafdArrangerStatic(CKplusArrangerStatic):
    def __init__(self, config, num_folds):
        super().__init__(config, num_folds)

    def establish_fold(self):
        foldwise_subject_count = self.count_subject_for_each_fold()
        fold_list = [[] for i in range(self.num_folds)]

        start = 0
        for i in range(self.num_folds):
            end = start + foldwise_subject_count[i]
            subject_id_in_this_fold = list(itemgetter(*range(start, end))(sorted(self.subject_list)))

            for subject_id in subject_id_in_this_fold:
                subject_directory = os.path.join(self.root_directory, subject_id)

                for path in Path(subject_directory).rglob('*.jpg'):
                    label = self.emotion_dict[str(path).split(os.sep)[7]]
                    image_directory = path.parent
                    for image in Path(image_directory).rglob('*.jpg'):
                        fold_list[i].append([label, str(image)])

            start = end

        return fold_list

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [7, 7, 7, 7, 7, 7, 7, 6, 6, 6]
        return foldwise_subject_count

    def init_emotion_dict(self):
        return {'Angry': 0, 'Contemptuous': 1, 'Disgusted': 2, 'Fearful': 3,
                'Happy': 4, 'Neutral': 5, 'Sad': 6, 'Surprised': 7}


class OuluArrangerStatic(CKplusArrangerStatic):
    def __init__(self, config, num_folds):
        super().__init__(config, num_folds)
        self.emotion_dict = self.init_emotion_dict()

    def init_emotion_dict(self):
        return {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happiness': 3, 'Sad': 4, 'Surprise': 5}

    @staticmethod
    def count_subject_for_each_fold():
        foldwise_subject_count = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        return foldwise_subject_count


class PlotHandler:
    r"""
    A class to plot the output-label figures.
    """

    def __init__(self, metrics, emotional_dimension, epoch_result_dict,
                 sessionwise_output_dict, sessionwise_continuous_label_dict,
                 epoch=None, train_mode=None, directory_to_save_plot=None):
        self.metrics = metrics
        self.emotional_dimension = emotional_dimension
        self.epoch_result_dict = epoch_result_dict

        self.epoch = epoch
        self.train_mode = train_mode
        self.directory_to_save_plot = directory_to_save_plot

        self.sessionwise_output_dict = sessionwise_output_dict
        self.sessionwise_continuous_label_dict = sessionwise_continuous_label_dict

    def complete_directory_to_save_plot(self):
        r"""
        Determine the full path to save the plot.
        """
        if self.train_mode:
            exp_folder = "train"
        else:
            exp_folder = "validate"

        if self.epoch is None:
            exp_folder = "test"

        directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder, "epoch_" + str(self.epoch))
        if self.epoch == "test":
            directory = os.path.join(self.directory_to_save_plot, "plot", exp_folder)

        os.makedirs(directory, exist_ok=True)
        return directory

    def reshape_dictionary_to_plot(self, sessionwise_date_dict):
        reshaped_dict = {key: [] for key in sessionwise_date_dict}

        for subject_id, subjectwise_data in sessionwise_date_dict.items():
            # reshaped_dict[subject_id] = []
            number_of_session = len(subjectwise_data[self.emotional_dimension[0]])
            intermediate_list = [{key: [] for key in self.emotional_dimension} for _ in range(number_of_session)]

            for emotion in self.emotional_dimension:
                for session_id, session_data in enumerate(subjectwise_data[emotion]):
                    intermediate_list[session_id][emotion] = session_data

            reshaped_dict[subject_id] = intermediate_list

        return reshaped_dict

    def save_output_vs_continuous_label_plot(self):
        r"""
        Plot the output versus continuous label figures for each session.
        """

        reshaped_sessionwise_output_dict = self.reshape_dictionary_to_plot(self.sessionwise_output_dict)
        reshaped_sessionwise_continuous_label = self.reshape_dictionary_to_plot(self.sessionwise_continuous_label_dict)

        # Determine the full path to save the figures.
        complete_directory = self.complete_directory_to_save_plot()

        # Read the sessionwise data.

        for (subject_id, sessionwise_output), (_, sessionwise_continuous_label) \
                in zip(reshaped_sessionwise_output_dict.items(), reshaped_sessionwise_continuous_label.items()):

            for session_id, (session_output, session_continuous_label) \
                    in enumerate(zip(sessionwise_output, sessionwise_continuous_label)):
                plot_filename = "subject_{}_trial_{}_epoch_{}".format(subject_id, session_id, self.epoch)
                full_plot_filename = os.path.join(complete_directory, plot_filename + ".jpg")

                self.plot_and_save(full_plot_filename, subject_id, session_id, session_output, session_continuous_label)

    def plot_and_save(self, full_plot_filename, subject_id, session_id, output, continuous_label):
        fig, ax = plt.subplots(len(self.emotional_dimension), 1)

        for index, emotion in enumerate(self.emotional_dimension):
            result_list = []

            for metric in self.metrics:
                result = self.epoch_result_dict[subject_id][emotion][metric][session_id][0]
                # The pcc usually have two output, one for value and one for confidence. So
                # here we only read and the value and discard the confidence.
                if metric == "pcc":
                    result = self.epoch_result_dict[subject_id][emotion][metric][session_id][0][0]
                result_list.append(result)

            # Plot the sub-figures, each for one emotional dimension.
            ax[index].plot(output[emotion], "r-", label="Output")
            ax[index].plot(continuous_label[emotion], "g-", label="Label")
            ax[index].set_xlabel("Sample")
            ax[index].set_ylabel("Value")
            ax[index].legend(loc="upper right", framealpha=0.2)
            ax[index].title.set_text(
                "{}: rmse={:.3f}, pcc={:.3f}, ccc={:.3f}".format(emotion, *result_list))

        fig.tight_layout()
        plt.savefig(full_plot_filename)
        plt.close()


class ContinuousMetricsCalculator:
    r"""
    A class to calculate the metrics, usually rmse, pcc, and ccc for continuous regression.
    """

    def __init__(
            self,
            metrics,
            emotional_dimension,
            output_handler,
            continuous_label_handler,
    ):

        # What metrics to calculate.
        self.metrics = metrics

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension

        # The instances saving the data for evaluation.
        self.output_handler = output_handler
        self.continuous_label_handler = continuous_label_handler

        # Initialize the dictionary for saving the metric results.
        self.metric_record_dict = self.init_metric_record_dict()

    def get_partitionwise_output_and_continuous_label(self):
        return self.output_handler.partitionwise_dict, \
               self.continuous_label_handler.partitionwise_dict

    def get_subjectwise_output_and_continuous_label(self):
        return self.output_handler.subjectwise_dict, \
               self.continuous_label_handler.subjectwise_dict

    def get_sessionwise_output_and_continuous_label(self):
        return self.output_handler.sessionwise_dict, \
               self.continuous_label_handler.sessionwise_dict

    def init_metric_record_dict(self):
        sessionwise_output, _ = self.get_sessionwise_output_and_continuous_label()
        metric_record_dict = {key: [] for key in sessionwise_output}
        return metric_record_dict

    @staticmethod
    def calculator(output, label, metric):
        if metric == "rmse":
            result = np.sqrt(((output - label) ** 2).mean())
        elif metric == "pcc":
            result = pearsonr(output, label)
        elif metric == "ccc":
            result = ConcordanceCorrelationCoefficient.calculate_CCC(output, label)
        else:
            SystemExit("Metric {} is not defined.".format(metric))
        return result

    def calculate_metrics(self):

        # Load the data for three scenarios.
        # They will all be evaluated.
        sessionwise_output, sessionwise_continuous_label = self.get_sessionwise_output_and_continuous_label()
        subjectwise_output, subjectwise_continuous_label = self.get_subjectwise_output_and_continuous_label()
        partitionwise_output, partitionwise_continuous_label = self.get_partitionwise_output_and_continuous_label()

        for (subject_id, output_list), (_, label_list) in zip(
                sessionwise_output.items(), sessionwise_continuous_label.items()):

            session_record_dict = {key: {} for key in self.emotional_dimension}

            for column, emotion in enumerate(self.emotional_dimension):
                session_number = len(output_list[emotion])

                for metric in self.metrics:
                    session_record_dict[emotion][metric] = {session_id: [] for session_id in range(session_number)}

                for session_id, (output, label) in enumerate(zip(output_list[emotion], label_list[emotion])):
                    # Session-wise evaluation
                    output = np.asarray(output)
                    label = np.asarray(label)

                    for metric in self.metrics:
                        result = self.calculator(output, label, metric)
                        session_record_dict[emotion][metric][session_id].append(result)

                # Subject-wise evaluation
                output = np.asarray(subjectwise_output[subject_id][emotion])
                label = np.asarray(subjectwise_continuous_label[subject_id][emotion])

                for metric in self.metrics:
                    result = self.calculator(output, label, metric)
                    session_record_dict[emotion][metric]['overall'] = []
                    session_record_dict[emotion][metric]['overall'].append(result)

            self.metric_record_dict[subject_id] = session_record_dict

        self.metric_record_dict['overall'] = {}

        # Partition-wise evaluation
        for emotion in self.emotional_dimension:
            partitionwise_dict = {metric: [] for metric in self.metrics}
            output = np.asarray(partitionwise_output[emotion][0])
            label = np.asarray(partitionwise_continuous_label[emotion][0])

            for metric in self.metrics:
                result = self.calculator(output, label, metric)
                partitionwise_dict[metric].append(result)

            self.metric_record_dict['overall'][emotion] = partitionwise_dict

class ContinuousOutputHandlerNPY(object):
    def __init__(self, length_to_track, emotion_dimension):
        self.length_to_track = length_to_track
        self.emotion_dimension = emotion_dimension
        self.subjectwise_dict = self.init_subjectwise_dict()
        self.sessionwise_dict = self.init_sessionwise_dict()
        self.partitionwise_dict = self.init_partition_dict()

    def place_clip_output_to_subjectwise_dict(self, clip_output, indices, sessions):
        for index, session in enumerate(sessions):
            subject_id = int(session.split("-")[0].split("P")[1])
            target_range = indices[index]


            self.append_clip_output_to_each_element(str(subject_id), clip_output[index, :, :], target_range)

    def append_clip_output_to_each_element(self, subject_id, clip_output, target_range):

        for emotion, column in zip(self.emotion_dimension, range(clip_output.shape[1])):
            for relative_idx, absolute_idx in enumerate(iter(target_range)):
                self.subjectwise_dict[subject_id][emotion][absolute_idx].append(
                    clip_output[relative_idx, column])

    def average_subjectwise_output(self):
        for subject_id in self.subjectwise_dict:
            for emotion in self.subjectwise_dict[subject_id]:
                length = len(self.subjectwise_dict[subject_id][emotion])

                for index in range(length):
                    self.subjectwise_dict[subject_id][emotion][index] = statistics.mean(
                        self.subjectwise_dict[subject_id][emotion][index])

    def get_partitionwise_dict(self):
        for emotion in self.emotion_dimension:
            for subject_id in self.subjectwise_dict:
                self.partitionwise_dict[emotion].append(self.subjectwise_dict[subject_id][emotion])

    def get_sessionwise_dict(self):
        self.average_subjectwise_output()

        for subject_id, related_session_length_list in self.length_to_track.items():

            # Use cumsum function to sneakily compute the start indices.
            start_list = np.insert(np.cumsum(self.length_to_track[subject_id]), 0, 0)[:-1]

            for emotion in self.emotion_dimension:
                for index, session_length in enumerate(related_session_length_list):
                    start = start_list[index]
                    end = start + session_length
                    self.sessionwise_dict[subject_id][emotion][index] = self.subjectwise_dict[subject_id][emotion][
                                                                        start:end]

    def init_sessionwise_dict(self):
        sessionwise_dict = {}
        for subject in self.length_to_track:
            intermediate_subjectwise_dict = {key: [] for key in self.emotion_dimension}
            for emotion in self.emotion_dimension:

                for _ in self.length_to_track[subject]:
                    intermediate_subjectwise_dict[emotion].append([])
            sessionwise_dict[subject] = intermediate_subjectwise_dict
        return sessionwise_dict

    def init_subjectwise_dict(self):
        subjectwise_dict = {}
        for subject in self.length_to_track:
            length_sum = np.sum(self.length_to_track[subject])
            subjectwise_list = {emotion: self.init_long_list(length_sum)
                                for emotion in self.emotion_dimension}
            subjectwise_dict[subject] = subjectwise_list
        return subjectwise_dict

    def init_partition_dict(self):
        partitionwise_dict = {key: [] for key in self.emotion_dimension}
        return partitionwise_dict

    @staticmethod
    def init_long_list(length):
        return [[] for _ in range(length)]

class ContinuousOutputHandler:
    r"""
    A class to handle the continuous output. For example, to restore the
        small clip of outputs and labels to session-wise, subject-wise, or overall
        manners, and also for calculating the metrics.
    """

    def __init__(self, length_to_track, clip_sample_map_to_track,
                 emotion_dimension, clipwise_frame_number, time_depth, step_size):
        self.length_to_track = length_to_track
        self.clip_sample_map_to_track = clip_sample_map_to_track
        self.emotion_dimension = emotion_dimension
        self.clipwise_frame_number = clipwise_frame_number
        self.time_depth = time_depth
        self.step_size = step_size
        self.subjectwise_dict = self.init_subjectwise_dict()
        self.sessionwise_dict = self.init_sessionwise_dict()
        self.partitionwise_dict = self.init_partition_dict()

    def get_partitionwise_dict(self):
        for emotion in self.emotion_dimension:
            for subject_id in self.subjectwise_dict:
                self.partitionwise_dict[emotion].append(self.subjectwise_dict[subject_id][emotion])

    def get_sessionwise_dict(self):
        self.average_subjectwise_output()

        for subject_id, related_session_length_list in self.length_to_track.items():

            # Use cumsum function to sneakily compute the start indices.
            start_list = np.insert(np.cumsum(self.length_to_track[subject_id]), 0, 0)[:-1]

            for emotion in self.emotion_dimension:
                for index, session_length in enumerate(related_session_length_list):
                    start = start_list[index]
                    end = start + session_length
                    self.sessionwise_dict[subject_id][emotion][index] = self.subjectwise_dict[subject_id][emotion][
                                                                        start:end]

    def average_subjectwise_output(self):
        for subject_id in self.subjectwise_dict:
            for emotion in self.subjectwise_dict[subject_id]:
                length = len(self.subjectwise_dict[subject_id][emotion])

                for index in range(length):
                    self.subjectwise_dict[subject_id][emotion][index] = statistics.mean(
                        self.subjectwise_dict[subject_id][emotion][index])

    def place_clip_output_to_subjectwise_dict(self, data_to_load_list, clip_output, sample_id_list):
        r"""
        Reshape the output to the subject-wise manner. I.e., the output of each subject is taken as a long array,
            by concatenating those from all the sessions. Thus, the metrics of each session can be calculated. (We
            should not calculate the metrics for only a small sample. That's why we have to restore the shape first.
        :param data_to_load_list: (list), the batch_size files corresponding to the current dataloader.
        :param sample_id_list: (list), the idx of the __getitem__ function of the dataloader.
        :param clip_output: (tensor), the output to reshape.
        :return: the reshaped output.
        """
        # The list recording to which subject, clip length, sampler index the mini-batch output belong.
        # For example, if mini-batch=4, then the 4 clips each has its own subject and sampler index,
        # with the constant clip length pre-determined.
        info_matrix = np.vstack([[self.get_subject(filename), self.get_frame_number(filename),
                                  self.get_clip_id(filename)] for index, filename in enumerate(data_to_load_list)])

        # The list indicating to which file the sample belong.
        file_id_list, relative_sample_id_list = self.generate_sample_to_file_map(sample_id_list, info_matrix)

        # Loop the mini-batch samples in this mini-batch.
        for index in range(sample_id_list.shape[0]):
            # Get the info for one sample at a time.
            subject_id, frame_number, clip_id = info_matrix[file_id_list[index], :]

            sample_id_relative_to_clip = relative_sample_id_list[index]

            sample_id_relative_to_subject = self.clip_sample_map_to_track[
                str(subject_id)][clip_id]['sample_id'][sample_id_relative_to_clip]
            session_id_relative_to_subject = self.clip_sample_map_to_track[
                str(subject_id)][clip_id]['session_id'][sample_id_relative_to_clip]

            # The range the outputs should be positioned.
            session_start_interval = np.insert(np.cumsum(self.length_to_track[str(subject_id)]), 0, 0)[:-1]
            start = session_start_interval[
                        session_id_relative_to_subject] + self.step_size * sample_id_relative_to_subject

            self.append_clip_output_to_each_element(str(subject_id), clip_output[index, :, :], int(start))

    def append_clip_output_to_each_element(self, subject_id, clip_output, start):
        end = start + self.time_depth

        for emotion, column in zip(self.emotion_dimension, range(clip_output.shape[1])):
            for relative_idx, absolute_idx in enumerate(np.arange(start, end)):
                self.subjectwise_dict[subject_id][emotion][absolute_idx].append(clip_output[relative_idx, column])

        # [[self.subjectwise_dict[subject_id][emotion][absolute_idx].append(clip_output[relative_idx, column])
        #   for relative_idx, absolute_idx in enumerate(np.arange(start, end))]
        #  for emotion, column in zip(self.emotion_dimension, range(clip_output.shape[1]))]

    def generate_sample_to_file_map(self, sample_id, info_matrix):
        r"""
        According to the sample id, identify the index of a sample related to it's source clip.
        :param sample_id: (ndarray), the id of the sample generate by the dataloader.
        :param info_matrix: (ndarray), storing the subject_id, frame_number, clip_id for each sample.
        :return: the file id and the relative sample id of the samples in sample_id. Here, the
            file id indicates to which clip file a sample belong.
        """

        # Generate the bisect interval.
        interval = np.insert(np.cumsum(info_matrix[:, 1]), 0, 0)[:-1]
        bisect_matrix = (sample_id * self.time_depth)[:, np.newaxis] - interval[np.newaxis, :]

        # Find the row-wise indices of the minimum positive elements.
        file_id = [int(np.where(bisect_matrix[row, :] >= 0)[0][-1])
                   for row in range(bisect_matrix.shape[0])]

        # Find the row-wise minimum positive elements then divide time_depth.
        relative_sample_id = [int(min([
            min_positive for min_positive in bisect_matrix[row, :] if min_positive >= 0])
                                  // self.time_depth) for row in range(bisect_matrix.shape[0])]

        return file_id, relative_sample_id

    def init_sessionwise_dict(self):
        sessionwise_dict = {}
        for subject in self.length_to_track:
            intermediate_subjectwise_dict = {key: [] for key in self.emotion_dimension}
            for emotion in self.emotion_dimension:

                for _ in self.length_to_track[subject]:
                    intermediate_subjectwise_dict[emotion].append([])
            sessionwise_dict[subject] = intermediate_subjectwise_dict
        return sessionwise_dict

    def init_subjectwise_dict(self):
        subjectwise_dict = {}
        for subject in self.length_to_track:
            length_sum = np.sum(self.length_to_track[subject])
            subjectwise_list = {emotion: self.init_long_list(length_sum)
                                for emotion in self.emotion_dimension}
            subjectwise_dict[subject] = subjectwise_list
        return subjectwise_dict

    def init_partition_dict(self):
        partitionwise_dict = {key: [] for key in self.emotion_dimension}
        return partitionwise_dict

    @staticmethod
    def init_long_list(length):
        return [[] for _ in range(length)]

    @staticmethod
    def get_subject(filename):
        subject = int(filename.split(os.sep)[-2])
        return subject

    @staticmethod
    def get_frame_number(filename):
        frame_number = int(os.path.splitext(filename)[0].split("_")[-1])
        return frame_number

    @staticmethod
    def get_clip_id(filename):
        clip_id = int(os.path.splitext(filename)[0].split("_")[-2])
        return clip_id


class NFoldArranger:
    r"""
    A class to prepare files according to the n-fold manner.
    """

    def __init__(self, root_directory):

        # The root directory of the dataset.
        self.root_directory = root_directory

        # Load the dataset information
        self.dataset_info = self.get_dataset_info()

        # Get the sessions having continuous labels.
        self.sessions_having_continuous_label = self.get_session_indices_having_continuous_label()

    def get_dataset_info(self):
        r"""
        Read the dataset info pkl file.
        :return: (dict), the dataset info.
        """
        dataset_info = load_single_pkl(self.root_directory, "dataset_info")
        return dataset_info

    def get_session_indices_having_continuous_label(self):
        r"""
        Get the session indices having continuous labels.
        :return: (list), the indices indicating which sessions have continuous labels.
        """
        indices = np.where(self.dataset_info['feeltrace_bool'] == 1)[0]
        return indices

    def get_subject_list_and_frequency(self):
        r"""
        Get the subject-wise session counts. It will be used for fold partition.
        :return: (list), the session counts of each subject.
        """
        subject_list, trial_count = np.unique(
            self.dataset_info['subject_id'][self.sessions_having_continuous_label], return_counts=True)
        return subject_list, trial_count

    def assign_session_to_subject(self):
        r"""
        Assign the sessions having continuous labels to its subjects.
        :return: (list), the list recording the session id having continuous labels for each subject.
        """
        subject_list, trial_count_for_each_subject = self.get_subject_list_and_frequency()

        session_id_of_each_subject = [
            np.where(self.dataset_info['subject_id'][self.sessions_having_continuous_label] == subject_id)[0] for
            _, subject_id in enumerate(subject_list)]
        return session_id_of_each_subject

    def assign_subject_to_fold(self, fold_number):
        r"""
        Assign the subjects and their sessions to a fold.
        :param fold_number: (int), how many fold the partition will create.
        :return: (list), the list recording the subject id and its associated session for each fold.
        """

        # Count the session number for each subject.
        subject_list, trial_count_for_each_subject = self.get_subject_list_and_frequency()

        # Calculate the expected session number for a fold, in order to partition it as evenly as possible.
        expected_trial_number_in_a_fold = np.sum(trial_count_for_each_subject) / fold_number

        # For preprocessing, or Leave One Subject Out scenario, which leaves one subject as a fold.
        if fold_number >= len(subject_list):
            expected_trial_number_in_a_fold = 0

        # In order to evenly partition the fold, we employ a simple algorithm. For each unprocessed
        # subjects, we always check if the current session number exceed the expected number. If
        # not, then assign the subject with the currently smallest number of session to be in the
        # current fold.

        # The mask is used to indicate whether the subject is assigned.
        mask = np.ones(len(subject_list), dtype=bool)

        subject_id_of_all_folds = []

        # Loop the subject.
        for i, (subject, trial_count) in enumerate(zip(subject_list, trial_count_for_each_subject)):

            # If the subject has not been assigned.
            if mask[i]:

                # Assign this subject to a new fold, then count the current session number,
                # and set the mask of this subject to False showing that it is assigned.
                one_fold = [subject]
                current_trial_number_in_a_fold = trial_count_for_each_subject[i]
                mask[i] = False

                # If the current session number is fewer than 90% of the expected number,
                # and there are still subjects that are not assigned.
                while (current_trial_number_in_a_fold <
                       expected_trial_number_in_a_fold * 0.9 and True in mask):
                    # Find the unassigned subject having the smallest session number currently.
                    trial_count_to_check = trial_count_for_each_subject[mask]
                    current_min_remaining = min(trial_count_to_check)

                    # Sometimes there are multiple subjects having the smallest number of session.
                    # If so, pick the first one to assign.
                    index_of_current_min = [j for j, count in
                                            enumerate(trial_count_for_each_subject)
                                            if (mask[j] and current_min_remaining == count)][0]

                    # Assign the subject to the fold.
                    one_fold.append(subject_list[index_of_current_min])

                    # Update the current count and mask.
                    current_trial_number_in_a_fold += current_min_remaining
                    mask[index_of_current_min] = False

                # Append the subjects of one fold to the final list.
                subject_id_of_all_folds.append(one_fold)

        # Also output the session id of all folds for convenience.
        session_id_of_all_folds = [np.hstack([np.where(
            self.dataset_info['subject_id'][self.sessions_having_continuous_label] == subject_id)[0]
                                              for subject_id in subject_id_of_one_fold])
                                   for subject_id_of_one_fold in subject_id_of_all_folds]

        return subject_id_of_all_folds, session_id_of_all_folds

    def get_partition_list(
            self,
            subject_id_of_all_folds,
            clip_number_to_load,
            partition_dictionary,
            downsampling_interval,
            compact_folder,
            time_depth,
            step_size,
            frame_number_to_load,
            shuffle=False,
            debug=False
    ):
        r"""
        A crucial function. It assigns mp4 files and their associated continuous
            labels to the training, validation, and test sets. It also produce the session-wise length dictionary
            for the purpose of restoring the model output to original, for the later metric calculation.
        :param subject_id_of_all_folds: (list), the list recording the fold-wise subject id.
        :param clip_number_to_load: (int), how many video clips to read at one time. It should
            be set according to the memory available.
        :param partition_dictionary: (dict), the dictionary indicating how many folds
            should be used for training, validation, and test partitions, respectively.
        :param downsampling_interval: (int), the downsampling interval indicating for every
            downsampling_interval frames will the jpg files be read. It specifies the folder of the
            correct data to be load.
        :param compact_folder: (str), the folder having the mp4 files and continuous labels.
        :param time_depth: (str), the frame count of a sample, i.e., its length.
        :param step_size: (str), the stride of the sampling window.
        :param frame_number_to_load: (int), the number of a video clip. It specifies the folder of the
            correct data to be read.
        :param shuffle: (boolean), the flag indicating whether to shuffle the video clips or not.
        :param debug: (boolean), the flag indicating whether to load much less data for a quick debugging.
        :return: (dict), (dict), it returns two dictionaries. One records the video files and their associated
            continuous label files for training, validation and testing. Another one records the session-wise length
            for the partitions.
        """

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputed.
        file_of_all_partitions = {key: [] for key in subject_id_of_all_partitions}
        length_of_all_partitions = {key: {} for key in subject_id_of_all_partitions}
        clip_sample_map_of_all_partitions = {key: {} for key in subject_id_of_all_partitions}

        # Inatialize the directories to read the video clips for non-testing and testing purpose.
        scale = "timedepth_" + str(time_depth) + "_stepsize_" + str(step_size)
        directory_string = os.path.join(
            self.root_directory, compact_folder, "{}", scale, str(frame_number_to_load))

        sliced_subject_session_dict = load_pkl_file(os.path.join(
            directory_string.format(str(downsampling_interval)), 'subject_sliced_session_length.pkl'))
        sliced_subject_session_non_downsampled_dict = load_pkl_file(os.path.join(
            directory_string.format(str(1)), 'subject_sliced_session_length.pkl'))

        subject_clip_sample_map = load_pkl_file(os.path.join(
            directory_string.format(str(downsampling_interval)), 'subject_clip_sample_map.pkl'))
        subject_clip_sample_non_downsampled_dict_map = load_pkl_file(os.path.join(
            directory_string.format(str(1)), 'subject_clip_sample_map.pkl'))

        # Loop the three partitions.
        for key in subject_id_of_all_partitions:
            fold_intermediate_dict = {'frame': [], 'continuous_label': []}

            # Loop the folds of a partition.
            for fold in subject_id_of_all_partitions[key]:

                # Loop the subjects of a fold.
                for subject in fold:

                    directory = os.path.join(
                        directory_string.format(str(downsampling_interval)), str(subject))

                    # Get the length of this subject. The length is the total number of frames to load for a subject.
                    length_of_all_partitions[key][str(subject)] = sliced_subject_session_dict[str(subject)]
                    clip_sample_map_of_all_partitions[key][str(subject)] = subject_clip_sample_map[subject]
                    # For testing the downsampling is disabled.
                    if key == 'test':
                        directory = os.path.join(directory_string.format(str(1)), str(subject))
                        length_of_all_partitions[key][str(subject)] = \
                            sliced_subject_session_non_downsampled_dict[str(subject)]
                        clip_sample_map_of_all_partitions[key][str(subject)] = \
                            subject_clip_sample_non_downsampled_dict_map[subject]

                    # Get the filenames of the data to read.
                    frame_list = sorted([os.path.join(directory, file)
                                         for file in os.listdir(directory) if file.startswith('frame')])
                    continuous_label_list = sorted([os.path.join(directory, file)
                                                    for file in os.listdir(directory) if file.startswith('ndarray')])

                    # Record the data filename to a dictionary.
                    fold_intermediate_dict = dict_combine(
                        fold_intermediate_dict, {'frame': frame_list, 'continuous_label': continuous_label_list})

            # Stack the intermediate dictionary so that a fold cares only the filenames, not the subject id anymore.
            fold_intermediate_dict = {key: np.hstack(value).tolist() for key, value in fold_intermediate_dict.items()}

            # If the shuffle is enabled, generate a random indices for the partition and shuffle it.
            if shuffle and key == 'train':
                total = len(fold_intermediate_dict['frame'])
                random_indices = random.sample(range(total), total)
                fold_intermediate_dict = {key: list(itemgetter(*random_indices)(value)) for key, value in
                                          fold_intermediate_dict.items()}

            # Group the filenames of every batch_size for a partition.
            fold_intermediate_dict = {
                key: [value[slice(clip_number_to_load * n, clip_number_to_load * (n + 1))]
                      for n in range(int(np.ceil(len(value) / clip_number_to_load)))]
                for key, value in fold_intermediate_dict.items()}

            # Restructure the dictionary so that for each partition, the grouped data is the element of a list.
            restructured_intermediate_dict = []
            for frame, continuous_label in zip(fold_intermediate_dict['frame'],
                                               fold_intermediate_dict['continuous_label']):
                restructured_intermediate_dict.append({'frame': frame, 'continuous_label': continuous_label})
            file_of_all_partitions[key] = restructured_intermediate_dict

            # If the debug is enabled, then select only the first file for fast running.
            if debug:
                file_of_all_partitions[key] = restructured_intermediate_dict[:1]

        return file_of_all_partitions, length_of_all_partitions, clip_sample_map_of_all_partitions

    @staticmethod
    def partition_train_validate_test_for_subjects(subject_id_of_all_folds, partition_dictionary):
        r"""
        A static function to assign the subjects to folds according to the partition dictionary.
        :param subject_id_of_all_folds: (list), the list recording the subject id of all folds.
        :param partition_dictionary: (dict), the dictionary indicating the fold numbers of each partition.
        :return: (dict), the dictionary indicating the subject ids of each partition.
        """

        # To assure that the fold number equals the value sum of the partition dictionary.
        assert len(subject_id_of_all_folds) == np.sum([value for value in partition_dictionary.values()])

        # Assign the subject id according to the dictionary.
        subject_id_of_all_partitions = {
            'train': subject_id_of_all_folds[0:partition_dictionary['train']],
            'validate': subject_id_of_all_folds[partition_dictionary['train']:
                                                partition_dictionary['train'] + partition_dictionary['validate']],
            'test': subject_id_of_all_folds[-partition_dictionary['test']:]
        }

        return subject_id_of_all_partitions


class AVEC19ArrangerNPY(object):
    def __init__(self, config):
        self.root_directory = config['root_directory']
        self.compact_folder = config['compact_folder']
        self.time_depth = config['time_depth']
        self.step_size = config['step_size']
        self.dataset_info = self.get_dataset_info()
        self.data_dict = self.init_data_dict()

    def make_length_dict(self, train_country="all", validate_country="all"):
        length_dict = self.generate_length_dict()
        if train_country == "all":
            train_dict = {**length_dict['Train']['DE'], **length_dict['Train']['HU']}
        else:
            train_dict = length_dict['Train'][train_country]

        if validate_country == "all":
            validate_dict = {**length_dict['Devel']['DE'], **length_dict['Devel']['HU']}
        else:
            validate_dict = length_dict['Devel'][validate_country]

        return train_dict, validate_dict


    def generate_length_dict(self):
        length_dict = self.init_length_dict()

        for index, subject_id in enumerate(self.dataset_info['subject_id']):
            length = [self.dataset_info['frame_count'][index] // 5]
            country = self.dataset_info['country'][index]
            partition = self.dataset_info['partition'][index]
            length_dict[partition][country].update({str(subject_id): length})

        return length_dict

    def make_data_dict(self, train_country="all", validate_country="all"):
        data_dict = self.generate_data_dict()
        if train_country == "all":
            train_dict = data_dict['Train']['DE']  + data_dict['Train']['HU']
        else:
            train_dict = data_dict['Train'][train_country]

        if validate_country == "all":
            validate_dict = data_dict['Devel']['DE']  + data_dict['Devel']['HU']
        else:
            validate_dict = data_dict['Devel'][validate_country]

        return train_dict, validate_dict


    def generate_data_dict(self):
        data_dict = self.init_data_dict()
        directory = os.path.join(self.root_directory, self.compact_folder)
        for index, trial in enumerate(sorted(os.listdir(directory))):

            trial_directory = os.path.join(directory, trial)
            country = self.get_country(self.dataset_info['partition_country_to_participant_trial_map'][trial])
            partition = self.get_partition(self.dataset_info['partition_country_to_participant_trial_map'][trial])
            session = trial_directory.split(os.sep)[-1]
            length = self.dataset_info['frame_count'][index] // 5
            num_windows = int(np.ceil((length - self.time_depth) / self.step_size)) + 1
            item_list = []

            for window in range(num_windows):
                start = window * self.step_size
                end = start + self.time_depth

                if end > length:
                    break

                indices = np.arange(start, end)
                item_list.append([trial_directory, indices, session])

            if (length - self.time_depth) % self.step_size != 0:
                start = length - self.time_depth
                end = length
                indices = np.arange(start, end)
                item_list.append([trial_directory, indices, session])

            # if partition == 'Train':
            #     for window in range(num_windows):
            #         start = window * self.step_size
            #         end = start + self.time_depth
            #         indices = np.arange(start, end)
            #         item_list.append([trial_directory, indices, session])
            #
            #     if (length - self.time_depth) % self.step_size != 0:
            #         start = length - self.time_depth
            #         end = length
            #         indices = np.arange(start, end)
            #         item_list.append([trial_directory, indices, session])
            #
            # elif partition == 'Devel':
            #     status = 0
            #     indices = np.arange(0, length)
            #     item_list.append([trial_directory, indices, session])
            # else:
            #     raise ValueError('Partition not supported!')

            data_dict[partition][country].extend(item_list)

        return data_dict

    def init_length_dict(self):
        length_dict = {"Train": {
            "DE": {},
            "HU": {},
        },
            "Devel": {
                "DE": {},
                "HU": {},
            }}
        return length_dict

    def init_data_dict(self):
        data_dict = {"Train": {
            "DE": [],
            "HU": [],
        },
            "Devel": {
                "DE": [],
                "HU": [],
            }}
        return data_dict

    def get_dataset_info(self):
        r"""
        Read the dataset info pkl file.
        :return: (dict), the dataset info.
        """
        dataset_info = load_single_pkl(self.root_directory, "dataset_info")
        return dataset_info

    @staticmethod
    def get_partition(string):
        partition = string.split("_")[0]
        return partition

    @staticmethod
    def get_country(string):
        country = string.split("_")[1]
        return country


class Avec2019Arranger(NFoldArranger):
    def __init__(self, root_directory, config):
        super().__init__(root_directory)
        self.train_country = config['train_country']
        self.validate_country = config['validate_country']

    def assign_subject_to_fold(self, fold_number=None):
        # subject_id_of_all_folds, session_id_of_all_folds
        subject_id_of_all_folds = [[], []]
        session_id_of_all_folds = [[], []]

        for index, (subject_trial, part_country) in enumerate(
                self.dataset_info['partition_country_to_participant_trial_map'].items()):
            partition = self.get_partition(part_country)
            country = self.get_country(part_country)
            subject = self.dataset_info['subject_id'][index]

            if partition == "Train" and (self.train_country == "all" or self.train_country == country):
                subject_id_of_all_folds[0].append(subject)
                session_id_of_all_folds[0].append(index)

            if partition == "Devel" and (self.validate_country == "all" or self.validate_country == country):
                subject_id_of_all_folds[1].append(subject)
                session_id_of_all_folds[1].append(index)

        return subject_id_of_all_folds, session_id_of_all_folds

    def get_partition_list(
            self,
            subject_id_of_all_folds,
            clip_number_to_load,
            downsampling_interval,
            compact_folder,
            time_depth,
            step_size,
            frame_number_to_load,
            partition_dictionary=None,
            shuffle=False,
            debug=False
    ):
        r"""
        A crucial function. It assigns mp4 files and their associated continuous
            labels to the training, validation, and test sets. It also produce the session-wise length dictionary
            for the purpose of restoring the model output to original, for the later metric calculation.
        :param subject_id_of_all_folds: (list), the list recording the fold-wise subject id.
        :param clip_number_to_load: (int), how many video clips to read at one time. It should
            be set according to the memory available.
        :param partition_dictionary: (dict), the dictionary indicating how many folds
            should be used for training, validation, and test partitions, respectively.
        :param downsampling_interval: (int), the downsampling interval indicating for every
            downsampling_interval frames will the jpg files be read. It specifies the folder of the
            correct data to be load.
        :param compact_folder: (str), the folder having the mp4 files and continuous labels.
        :param time_depth: (str), the frame count of a sample, i.e., its length.
        :param step_size: (str), the stride of the sampling window.
        :param frame_number_to_load: (int), the number of a video clip. It specifies the folder of the
            correct data to be read.
        :param shuffle: (boolean), the flag indicating whether to shuffle the video clips or not.
        :param debug: (boolean), the flag indicating whether to load much less data for a quick debugging.
        :return: (dict), (dict), it returns two dictionaries. One records the video files and their associated
            continuous label files for training, validation and testing. Another one records the session-wise length
            for the partitions.
        """

        # Get the partition-wise subject dictionary.
        subject_id_of_all_partitions = self.partition_train_validate_test_for_subjects(
            subject_id_of_all_folds, partition_dictionary)

        # Initialize the dictionary to be outputed.
        file_of_all_partitions = {key: [] for key in subject_id_of_all_partitions}
        length_of_all_partitions = {key: {} for key in subject_id_of_all_partitions}
        clip_sample_map_of_all_partitions = {key: {} for key in subject_id_of_all_partitions}

        # Inatialize the directories to read the video clips for non-testing and testing purpose.
        scale = "timedepth_" + str(time_depth) + "_stepsize_" + str(step_size)
        directory_string = os.path.join(
            self.root_directory, compact_folder, "{}", scale, str(frame_number_to_load))

        sliced_subject_session_dict = load_pkl_file(os.path.join(
            directory_string.format(str(downsampling_interval)), 'subject_sliced_session_length.pkl'))

        subject_clip_sample_map = load_pkl_file(os.path.join(
            directory_string.format(str(downsampling_interval)), 'subject_clip_sample_map.pkl'))

        # Loop the three partitions.
        for key in subject_id_of_all_partitions:
            fold_intermediate_dict = {'frame': [], 'continuous_label': []}

            # Loop the folds of a partition.
            for fold in subject_id_of_all_partitions[key]:

                # Loop the subjects of a fold.
                for subject in fold:
                    directory = os.path.join(
                        directory_string.format(str(downsampling_interval)), str(subject))

                    # Get the length of this subject. The length is the total number of frames to load for a subject.
                    length_of_all_partitions[key][str(subject)] = sliced_subject_session_dict[str(subject)]
                    clip_sample_map_of_all_partitions[key][str(subject)] = subject_clip_sample_map[subject]
                    # For testing the downsampling is disabled.

                    # Get the filenames of the data to read.
                    frame_list = sorted([os.path.join(directory, file)
                                         for file in os.listdir(directory) if file.startswith('frame')])
                    continuous_label_list = sorted([os.path.join(directory, file)
                                                    for file in os.listdir(directory) if file.startswith('ndarray')])

                    # Record the data filename to a dictionary.
                    fold_intermediate_dict = dict_combine(
                        fold_intermediate_dict, {'frame': frame_list, 'continuous_label': continuous_label_list})

            # Stack the intermediate dictionary so that a fold cares only the filenames, not the subject id anymore.
            fold_intermediate_dict = {key: np.hstack(value).tolist() for key, value in fold_intermediate_dict.items()}

            # If the shuffle is enabled, generate a random indices for the partition and shuffle it.
            if shuffle and key == 'train':
                total = len(fold_intermediate_dict['frame'])
                random_indices = random.sample(range(total), total)
                fold_intermediate_dict = {key: list(itemgetter(*random_indices)(value)) for key, value in
                                          fold_intermediate_dict.items()}

            # Group the filenames of every batch_size for a partition.
            fold_intermediate_dict = {
                key: [value[slice(clip_number_to_load * n, clip_number_to_load * (n + 1))]
                      for n in range(int(np.ceil(len(value) / clip_number_to_load)))]
                for key, value in fold_intermediate_dict.items()}

            # Restructure the dictionary so that for each partition, the grouped data is the element of a list.
            restructured_intermediate_dict = []
            for frame, continuous_label in zip(fold_intermediate_dict['frame'],
                                               fold_intermediate_dict['continuous_label']):
                restructured_intermediate_dict.append({'frame': frame, 'continuous_label': continuous_label})
            file_of_all_partitions[key] = restructured_intermediate_dict

            # If the debug is enabled, then select only the first file for fast running.
            if debug:
                file_of_all_partitions[key] = restructured_intermediate_dict[:1]

        return file_of_all_partitions, length_of_all_partitions, clip_sample_map_of_all_partitions

    @staticmethod
    def partition_train_validate_test_for_subjects(subject_id_of_all_folds, partition_dictionary=None):
        r"""
        A static function to assign the subjects to folds according to the partition dictionary.
        :param subject_id_of_all_folds: (list), the list recording the subject id of all folds.
        :param partition_dictionary: (dict), the dictionary indicating the fold numbers of each partition.
        :return: (dict), the dictionary indicating the subject ids of each partition.
        """

        # Assign the subject id according to the dictionary.
        subject_id_of_all_partitions = {
            'train': [subject_id_of_all_folds[0]],
            'validate': [subject_id_of_all_folds[1]]
        }

        return subject_id_of_all_partitions

    @staticmethod
    def get_partition(string):
        partition = string.split("_")[0]
        return partition

    @staticmethod
    def get_country(string):
        country = string.split("_")[1]
        return country


class VideoSplit:
    r"""
        A base class to  split video according to a list. For example, given
        [(0, 1000), (1200, 1500), (1800, 1900)] as the indices, the associated
        frames will be split and combined  to form a new video.
    """

    def __init__(self, input_filename, output_filename, trim_range):
        r"""
        The init function of the class.
        :param input_filename: (str), the absolute directory of the input video.
        :param output_filename:  (str), the absolute directory of the output video.
        :param trim_range: (list), the indices of useful frames.
        """

        self.input_filename = input_filename
        self.output_filename = output_filename

        self.video = cv2.VideoCapture(self.input_filename)

        # The frame count.
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # The fps count.
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        # The size of the video.
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # The range to trim the video.
        self.trim_range = trim_range

        # The settings for video writer.
        self.codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(output_filename,
                                      self.codec, self.fps,
                                      (self.width, self.height), isColor=True)

    def jump_to_frame(self, frame_index):
        r"""
        Jump to a specific frame by its index.
        :param frame_index:  (int), the index of the frame to jump to.
        :return: none.
        """
        self.video.set(1, frame_index)

    def read(self, start, end, visualize):
        r"""
        Read then write the frames within (start, end) one frame at a time.
        :param start:  (int), the starting index of the range.
        :param end:  (int), the ending index of the range.
        :param visualize:  (boolean), whether to visualize the process.
        :return:  none.
        """

        # Jump to the starting frame.
        self.jump_to_frame(start)

        # Sequentially write the next end-start frames.
        for index in range(end - start):
            ret, frame = self.video.read()
            self.writer.write(frame)
            if ret and visualize:
                cv2.imshow('frame', frame)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(1)

    def combine(self, visualize=False):
        r"""
        Combine the clips  into a single video.
        :param visualize: (boolean), whether to visualize the process.
        :return:  none.
        """

        # Iterate over the pair of start and end.
        for clip_index in range(self.trim_range.shape[0]):
            (start, end) = self.trim_range[clip_index]
            self.read(start, end, visualize)

        self.video.release()
        self.writer.release()
        if visualize:
            cv2.destroyWindow('frame')


# class GenericLandmarkLib:
#     r"""
#     A base class to extract facial landmark using facial detector , predictor, and trackor.
#         The default method is dlib detector, predictor and opencv lk trackor.
#     """
#
#     def __init__(self, config):
#         r"""
#         The init function, which:
#             1: downloads the landmark dat file,
#             2: configure the detector, predictor, and trackor,
#         :param config:
#         """
#         self.url = config["landmark_model_download_url"]
#
#         predictor_filename = os.path.join(
#             config["landmark_model_folder"],
#             config["landmark_model_filename"])
#
#         if not os.path.isfile(predictor_filename):
#             print("Downloading: " + config["landmark_model_filename"] + "...")
#             urllib.request.urlretrieve(self.url, predictor_filename)
#             print("Downloading completed.")
#
#         self.output_frame_size = config["output_frame_size"]
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = dlib.shape_predictor(predictor_filename)
#
#         # Initialize the template of 68 landmarks. The dummy column is to form the projective coordinate.
#         self.template, self.template_key_indices, self.template_outline_indices = facial_landmark_template()
#         self.dummy = np.ones((config["landmark_number"], 1), dtype=np.float32)
#
#         # Config of Opencv lk algorithm.
#         self.lk_params = dict(winSize=(35, 35), maxLevel=5,
#                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))
#
#     def extract_landmark(self, frame):
#         r"""
#         Extract the facial landmark for one frame.
#         :param frame: (uint8 ndarray), the frame to be processed.
#         :return: the landmark coordinates and the bounding box.
#         """
#         rectangle = self.detector(frame, 1)[0]
#
#         if rectangle is None:
#             return None, None
#
#         shape = self.predictor(frame, rectangle)
#
#         landmark = face_utils.shape_to_np(shape)
#         bounding_box = np.array(face_utils.rect_to_bb(rectangle)).reshape(2, 2)
#
#         return landmark, bounding_box
#
#     def track_landmark(self, frame, frame_old, points_old):
#         r"""
#         Track the landmarks extracted in a given frame.
#         :param frame: (uint8 ndarray), the frame to process.
#         :param frame_old:  (uint8 ndarray), the old frame containing the landmarks already.
#         :param points_old:  (float ndarray), the old landmarks from frame_old.
#         :return: (float ndarray), the landmarks for frame.
#         """
#         # Reshape the ndarray to satisfy the input dimension of lk algorithm,
#         points_expended = np.expand_dims(points_old, axis=1).astype(np.float32)
#         new_points_expended, state, _ = cv2.calcOpticalFlowPyrLK(frame_old, frame, points_expended, None,
#                                                                  **self.lk_params)
#
#         # If successfully find the landmarks by lk algorithm,
#         if new_points_expended.shape == (68, 1, 2):
#             new_points = new_points_expended.reshape(68, 2)
#
#         # Otherwise use the latest coordinates.
#         else:
#             new_points = points_old
#
#         return new_points
#
#     def get_affine_matrix(self, landmark):
#         r"""
#         Calculate the affine matrix from the source to the target coordinates.
#             Here, the template_key_indices defines which points to select.
#         :param landmark: (float ndarray), the landmark to align.
#         :return: (float ndarray), the 2x3 affine matrix.
#         """
#         source = landmark[self.template_key_indices]
#         target = self.template[self.template_key_indices] * self.output_frame_size
#         affine_matrix = cv2.getAffineTransform(source, target)
#         return affine_matrix
#
#     def align_landmark(self, landmark, affine_matrix):
#         r"""
#         Warp the landmark by the defined affine transformation.
#         :param landmark: (float ndarray), the landmark to warp.
#         :param affine_matrix: (float ndarray), the affine matrix.
#         :return: (float ndarray), the aligned landmarks.
#         """
#         aligned_landmark = np.c_[landmark, self.dummy].dot(affine_matrix.T)
#         return aligned_landmark
#
#     def align_frame(self, frame, affine_matrix):
#         r'''
#         Warp the frame by the defined affine transformation.
#         :param frame: (uint8 ndarray), the frame to warp.
#         :param affine_matrix: (float ndarray), the affine matrix.
#         :return: (uint8 ndarray), the aligned frame.
#         '''
#         aligned_frame = cv2.warpAffine(frame, affine_matrix,
#                                        (self.output_frame_size,
#                                         self.output_frame_size))
#         return aligned_frame
#
#     def crop_facial_region(self, frame, landmark):
#         r"""
#         Crop the facial region according to the convex hull of the landmark outline.
#         :param frame: (uint8 ndarray), the frame to crop.
#         :param landmark: (float ndarray), the landmarks defining the convex hull.
#         :return: (uint8 ndarray), the cropped frame, whose regions outside the facial outline are filled in black.
#         """
#         # Select the outline points.
#         outline_points = landmark[self.template_outline_indices, :]
#
#         # Obtain the vertices of the polygon from the convex hull.
#         vertices = ConvexHull(outline_points).vertices
#         row_pixel_indices, column_pixel_indices = polygon(
#             outline_points[vertices, 1],
#             outline_points[vertices, 0])
#
#         # To ensure that the indices are within the desired frame of, e.g.,  250x250.
#         row_pixel_indices = np.clip(row_pixel_indices, 0, self.output_frame_size - 1)
#         column_pixel_indices = np.clip(column_pixel_indices, 0, self.output_frame_size - 1)
#
#         # Crop the facial region by setting pixel values to 0 for those outside the convex hull.
#         cropped_frame = np.zeros_like(frame)
#         cropped_frame[row_pixel_indices, column_pixel_indices, :] = frame[
#                                                                     row_pixel_indices, column_pixel_indices,
#                                                                     :]
#
#         return cropped_frame

class facial_image_crop_by_landmark(object):
    def __init__(self, config):
        self.dummy = np.ones((config["landmark_number"], 1), dtype=np.float32)
        self.template, self.template_key_indices, self.template_outline_indices = facial_landmark_template()
        self.landmark_number = config['landmark_number']
        self.output_image_size = config['output_image_size']

    def crop_image(self, image, landmark):
        affine_matrix = self.get_affine_matrix(landmark)
        aligned_landmark = self.align_landmark(landmark, affine_matrix)
        aligned_image = self.align_image(image, affine_matrix)
        return aligned_image

    def align_image(self, image, affine_matrix):
        r'''
        Warp the frame by the defined affine transformation.
        :param frame: (uint8 ndarray), the frame to warp.
        :param affine_matrix: (float ndarray), the affine matrix.
        :return: (uint8 ndarray), the aligned frame.
        '''
        aligned_image = cv2.warpAffine(image, affine_matrix,
                                       (self.output_image_size,
                                        self.output_image_size))
        return aligned_image

    def align_landmark(self, landmark, affine_matrix):
        r"""
        Warp the landmark by the defined affine transformation.
        :param landmark: (float ndarray), the landmark to warp.
        :param affine_matrix: (float ndarray), the affine matrix.
        :return: (float ndarray), the aligned landmarks.
        """
        aligned_landmark = np.c_[landmark, self.dummy].dot(affine_matrix.T)
        return aligned_landmark

    def get_affine_matrix(self, landmark):
        r"""
        Calculate the affine matrix from the source to the target coordinates.
            Here, the template_key_indices defines which points to select.
        :param landmark: (float ndarray), the landmark to align.
        :return: (float ndarray), the 2x3 affine matrix.
        """
        source = np.asarray(landmark[self.template_key_indices], dtype=np.float32)
        target = np.asarray(self.template[self.template_key_indices] * self.output_image_size, dtype=np.float32)
        affine_matrix = cv2.getAffineTransform(source, target)
        return affine_matrix


def variable_length_collate_fn(data):
    r"""
    A custom collate_fn of dataloader for variable length sequential data.
    :param data: (ndarray), the data of a mini-batch to collate. In this case, it should be data from "batch_size" trials.
        Given a mini-batch, the data will be padded according to the current maximum length dynamically.
    :return: (tensor), the padded  mini-batch features, the mini-batch length, the padded mini-batch labels,
        the padded mini-batch length. Note, the lengths are used for packing the data before
        feeding to a Pytorch RNN/LSTM layer.
    """

    def merge(sequences):
        r"""
        It first create a batch_size x max_length dimensional matrix, to which it then copies the variable sequential data.
        :param sequences: (ndarray), the variable length batch sequences to pad.
        :return: (ndarray), the padded sequential data with fixed length.
        """

        # Get the length for each sequence in a mini-batch.
        lengths = [len(seq) for seq in sequences]

        # Create the zero matrix of batch_size x max_length.
        padded_sequences = np.zeros((len(sequences), max(lengths), len(sequences[0][1])), dtype=np.float32)

        # Copy the mini-batch data to the zero matrix.
        for index, sequence in enumerate(sequences):
            end = lengths[index]
            padded_sequences[index, :end, :] = sequence[:end, :]

        return padded_sequences, lengths

    # Sort the data in this mini-batch, this will benefit the packing and unpacking
    #   process.
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # Separate the data into features and labels.
    feature_sequences, label_sequences = zip(*data)

    # Dynamically pad the features and labels.
    feature_sequences, feature_lengths = merge(feature_sequences)
    label_sequences, label_lengths = merge(label_sequences)

    # Convert the data type from ndarray to tensor.
    feature_sequences = torch.from_numpy(feature_sequences)
    label_sequences = torch.from_numpy(label_sequences)

    return feature_sequences, feature_lengths, label_sequences, label_lengths


class ConcordanceCorrelationCoefficient:
    """
    A class for performing concordance correlation coefficient (CCC) centering. Basically, when multiple continuous labels
    are available, it is not a good choice to perform a direct average. Formally, a Lin's CCC centering has to be done.

    This class is a Pythonic equivalence of CCC centering to the Matlab scripts ("run_gold_standard.m")
        from the AVEC2016 dataset.

    Ref:
        "Lawrence I-Kuei Lin (March 1989).  A concordance correlation coefficient to evaluate reproducibility".
            Biometrics. 45 (1): 255–268. doi:10.2307/2532051. JSTOR 2532051. PMID 2720055.
    """

    def __init__(self, data):
        self.data = data
        if data.shape[0] > data.shape[1]:
            self.data = data.T
        self.rator_number = self.data.shape[0]
        self.combination_list = self.generate_combination_pair()
        self.cnk_matrix = self.generate_Cnk_matrix()
        self.CCC = self.calculate_paired_CCC()
        self.agreement = self.calculate_rator_wise_agreement()
        self.mean_data = self.calculate_mean_data()
        self.weight = self.calculate_weight()
        self.centered_data = self.perform_centering()

    def perform_centering(self):
        """
        The centering is done by directly average the shifted and weighted data.
        :return: (ndarray), the centered  data.
        """
        centered_data = self.data - np.repeat(self.mean_data[:, np.newaxis], self.data.shape[1], axis=1) + self.weight
        return centered_data

    def calculate_weight(self):
        """
        The weight of the m continuous labels. It will be used to weight (actually translate) the data when
            performing the final step.
        :return: (float), the weight of the given m continuous labels.
        """
        weight = np.sum((self.mean_data * self.agreement) / np.sum(self.agreement))
        return weight

    def calculate_mean_data(self):
        """
        A directly average of data.
        :return: (ndarray), the averaged data.
        """
        mean_data = np.mean(self.data, axis=1)
        return mean_data

    def generate_combination_pair(self):
        """
        Generate all possible combinations of Cn2.
        :return: (ndarray), the combination list of Cn2.
        """
        n = self.rator_number
        combination_list = []

        for boy in range(n - 1):
            for girl in np.arange(boy + 1, n, 1):
                combination_list.append([boy, girl])

        return np.asarray(combination_list)

    def generate_Cnk_matrix(self):
        """
        Generate the Cn2 matrix. The j-th column of the matrix records all the possible candidate
            to the j-th rater. So that for the j-th column, we can acquire all the possible unrepeated
            combination for the j-th rater.
        :return:
        """
        total = self.rator_number
        cnk_matrix = np.zeros((total - 1, total))

        for column in range(total):
            cnk_matrix[:, column] = np.concatenate((np.where(self.combination_list[:, 0] == column)[0],
                                                    np.where(self.combination_list[:, 1] == column)[0]))

        return cnk_matrix.astype(int)

    @staticmethod
    def calculate_CCC(array1, array2):
        """
        Calculate the CCC.
        :param array1: (ndarray), an 1xn array.
        :param array2: (ndarray), another 1xn array.
        :return: the CCC.
        """
        array1_mean = np.mean(array1)
        array2_mean = np.mean(array2)

        array1_var = np.var(array1)
        array2_var = np.var(array2)

        covariance = np.mean((array1 - array1_mean) * (array2 - array2_mean))
        concordance_correlation_coefficient = (2 * covariance) / (
                array1_var + array2_var + (array1_mean - array2_mean) ** 2 + 1e-100)
        return concordance_correlation_coefficient

    def calculate_paired_CCC(self):
        """
        Calculate the CCC for all the pairs from the combination list.
        :return: (ndarray), the CCC for each combinations.
        """
        CCC = np.zeros((self.combination_list.shape[0]))
        for index in range(len(self.combination_list)):
            CCC[index] = self.calculate_CCC(self.data[self.combination_list[index, 0], :],
                                            self.data[self.combination_list[index, 1], :])

        return CCC

    def calculate_rator_wise_agreement(self):
        """
        Calculate the inter-rater CCC agreement.
        :return: (ndarray), a array recording the CCC agreement of each single rater to all the rest raters.
        """

        CCC_agreement = np.zeros(self.rator_number)

        for index in range(self.rator_number):
            CCC_agreement[index] = np.mean(self.CCC[self.cnk_matrix[:, index]])

        return CCC_agreement


class GenericEegPreprocessing:
    r"""
    A generic class for EEG signal preprocessing. It generally carry out the following operations:
    1. Load the data in bdf format.
    2. Segment the stimulated signal and concatenate to form a continuous one.
    3. Filter the eeg signal so that only the components in bands of 1-50Hz remain.
    4. Interactively carry out:
        4.1 Notch filtering.
        4.2 Independent Component Analysis.
    5. Mean reference the EEG signal.
    """

    def __init__(self, filename):

        self.filename = filename
        self.frequency = 256
        self.raw_data = self.read_data()
        self.channel_slice = self.get_channel_slice()
        self.channel_type_dictionary = self.get_channel_type_dictionary()
        self.raw_data = self.set_channel_types_from_dictionary()
        self.crop_range = self.get_crop_range_in_second()
        self.cropped_raw_data = self.crop_data()

        self.cropped_eeg_data = self.get_eeg_data()

        self.filtered_cropped_eeg_data = self.filter_eeg_data()

        self.ica = self.independent_component_analysis()

        self.repaired_data, self.log = self.repair_data()

    def repair_data(self):
        r"""
        The key function to repair the data. Currently it conduct in the following logic.
            If the user inputs "1", then the power spectrum density is shown.
            If the user inputs "2", then the notch filtering is running, and the user can further
                input a signal float to determine the band to notch.
            If the user inputs "3", then the ICA interactive plot is shown, and the user can further
                input an array of integers to determine which component to discard. Once inputted, the
                reconstructed signal will appear as another interactive plot, and waiting for the user to confirm.
            If the user inputs "4", then the data will be reset to raw data.
            If the user inputs "5", then the data will be skipped. This option is here to deal with failed data.
            If the user inputs "6", then the function will break and go to the next data.
        Note that the operations will be record in a txt file.
        :return: (mne object), the repaired data, alone with a new line in a log file.
        """
        self.filtered_cropped_eeg_data.plot()

        self.cropped_eeg_data.load_data()
        self.ica.plot_sources(self.cropped_eeg_data, block=True)

        data_to_repair = self.filtered_cropped_eeg_data.copy()

        log = "file-{}-skip-{}-notch-{}-ica-{}-confidence-{}\n"
        skip, frequencies, exclude_list, confidence = 0, " ", " ", " "
        while True:
            command = input("Please specify a task by index\n"
                            "1: plot power spectrum density.\n"
                            "2: carry out notch filtering.\n"
                            "3: carry out ica reconstruction.\n"
                            "4: reset the data.\n"
                            "5: skip the data.\n"
                            "6: quit.\n")

            if command == "1":

                data_to_repair.plot_psd(
                    fmax=self.frequency // 2, average=True)
                data_to_repair.plot(block=True)

            elif command == "2":
                while True:
                    command_notch = input("Which frequency to filter?\t"
                                          "Please split your input by space.\n").split(" ")

                    if command_notch != [""]:
                        frequencies = np.asarray([float(cmd) for cmd in command_notch])
                        data_to_notch = data_to_repair.copy().notch_filter(freqs=frequencies)

                        data_to_notch.plot_psd(fmax=self.frequency // 2, average=True)
                        data_to_notch.plot(block=True)

                        command_notch_confirm = input("Confirm the notch filtering?\n"
                                                      "y: yes.\n"
                                                      "n: no.\n")

                        if command_notch_confirm == ("y" or "yes"):
                            data_to_repair = data_to_notch
                            # log.update({"notch": frequencies})
                            break
                    else:
                        break

            elif command == "3":
                while True:

                    data_to_ica = data_to_repair.copy()
                    self.ica.plot_sources(data_to_ica, block=True)

                    indices_str_list = input(
                        "Which component(s) to exclude?\n"
                        " Place input integer and split by space. "
                    ).split(" ")
                    exclude_list = [int(string) for string in indices_str_list if string != '']
                    self.ica.exclude = exclude_list
                    self.ica.plot_sources(data_to_ica, block=True)
                    self.ica.apply(data_to_ica)
                    data_to_ica.plot(block=True)
                    command_ica_confirm = input("Confirm the ica reconstruction?\n"
                                                "y: yes.\n"
                                                "n: no.\n")
                    if command_ica_confirm == ("y" or "yes"):
                        data_to_repair = data_to_ica

                        break
                    else:
                        self.ica.exclude = []

            elif command == "4":
                data_to_repair = self.filtered_cropped_eeg_data.copy()
                skip, frequencies, exclude_list, confidence = 0, " ", " ", " "

            elif command == "5":
                skip = 1

            elif command == "6":
                confidence = input("What's your confidence about your judgement?\n"
                                   "3: high, (I am sure the signal has been well processed.)\n"
                                   "2: normal, (I have made conservative reparation.\n"
                                   "1: low, (I am not sure, may be I should ask experts.\n")

                break

        data_to_repair.set_eeg_reference(ref_channels='average')

        log = log.format(self.filename, skip, frequencies, exclude_list, confidence)
        return data_to_repair[:, :], log

    def independent_component_analysis(self):
        r"""
        Fit the ica to the filtered eeg data.
        :return: (mne object), the fitted ica object.
        """
        n_components = len(self.cropped_eeg_data.ch_names)
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=1)
        ica.fit(self.filtered_cropped_eeg_data)

        return ica

    def filter_eeg_data(self):
        r"""
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = self.cropped_eeg_data.load_data().filter(l_freq=1, h_freq=50)
        return filtered_eeg_data

    def get_eeg_data(self):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = self.cropped_raw_data.copy().pick_types(eeg=True)
        return eeg_data

    def set_channel_types_from_dictionary(self):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = self.raw_data.set_channel_types(self.channel_type_dictionary)
        return data

    def get_channel_type_dictionary(self):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in self.raw_data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    def get_channel_slice(self):
        r"""
        Assign a tag to each channel according to the dataset paradigm.
        :return:
        """
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def crop_data(self):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(self.crop_range):

            if index == 0:
                cropped_data = self.raw_data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(self.raw_data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    def get_crop_range_in_second(self):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the time interval.
        """
        crop_range = [[30., self.raw_data.times.max() - 30 + self.buffer]]
        return crop_range

    def read_data(self):
        r"""
        Load the bdf data using mne API.
        :return: (mne object), the raw signal containing different channels.
        """
        filename = self.filename

        if filename.endswith(".bdf"):
            raw_data = mne.io.read_raw_bdf(filename)

        return raw_data

class GenericEegPreprocessingMahnob(object):
    def __init__(self, filename, buffer=2):
        self.filename = filename
        self.buffer = buffer
        self.frequency = 256
        self.raw_data = self.read_data()
        self.channel_slice = self.get_channel_slice()
        self.channel_type_dictionary = self.get_channel_type_dictionary()
        self.raw_data = self.set_channel_types_from_dictionary()
        self.crop_range = self.get_crop_range_in_second()
        self.cropped_raw_data = self.crop_data()

        self.cropped_eeg_data = self.get_eeg_data()

        self.average_referenced_data = self.average_reference()
        self.filtered_data = self.filter_eeg_data()

    def filter_eeg_data(self):
        r"""
        Filter the eeg signal using lowpass and highpass filter.
        :return: (mne object), the filtered eeg signal.
        """
        filtered_eeg_data = self.average_referenced_data.filter(l_freq=0.3, h_freq=45)
        return filtered_eeg_data[:][0].T

    def average_reference(self):
        average_referenced_data = self.cropped_eeg_data.copy().load_data().set_eeg_reference()
        return average_referenced_data

    def read_data(self):
        r"""
        Load the bdf data using mne API.
        :return: (mne object), the raw signal containing different channels.
        """
        filename = self.filename

        if filename.endswith(".bdf"):
            raw_data = mne.io.read_raw_bdf(filename)

        return raw_data

    def get_channel_slice(self):
        r"""
        Assign a tag to each channel according to the dataset paradigm.
        :return:
        """
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def get_channel_type_dictionary(self):
        r"""
        Generate a dictionary where the key is the channel names, and the value
            is the modality name (such as eeg, ecg, eog, etc...)
        :return: (dict), the dictionary of channel names to modality name.
        """
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in self.raw_data.ch_names[
                                                self.channel_slice[modal]]})

        return channel_type_dictionary

    def set_channel_types_from_dictionary(self):
        r"""
        Set the channel types of the raw data according to a dictionary. I did this
            in order to call the automatic EOG, ECG remover. But it currently failed. Need to check.
        :return:
        """
        data = self.raw_data.set_channel_types(self.channel_type_dictionary)
        return data

    def get_crop_range_in_second(self):
        r"""
        Assign the stimulated time interval for cropping.
        :return: (list), the list containing the time interval.
        """
        crop_range = [[30., self.raw_data.times.max() - 30 + self.buffer]]
        return crop_range

    def crop_data(self):
        r"""
        Crop the signal so that only the stimulated parts are preserved.
        :return: (mne object), the cropped data.
        """
        cropped_data = []
        for index, (start, end) in enumerate(self.crop_range):

            if index == 0:
                cropped_data = self.raw_data.copy().crop(tmin=start, tmax=end)
            else:
                cropped_data.append(self.raw_data.copy().crop(tmin=start, tmax=end))

        return cropped_data

    def get_eeg_data(self):
        r"""
        Get only the eeg data from the raw data.
        :return: (mne object), the eeg signal.
        """
        eeg_data = self.cropped_raw_data.copy().pick_types(eeg=True)
        return eeg_data
import glob
import pickle
import shutil
import cv2
import os

import pandas as pd
import numpy as np
import torch

def load_single_csv(directory=None, filename=None, extension=None, header=0):
    fullname = os.path.join(directory, filename + extension)
    csv_data = pd.read_csv(fullname, header=header)
    return csv_data


def length_to_mask(lengths, device):
    r"""
    Generate mask according to lengths.
    :param lengths: (ndarray), the length of each useful data sequence.
    :param device: the device currently in use.
    :return: the mask sent to device.
    """
    row = len(lengths)
    column = max(lengths)
    mask = torch.zeros((row, column), dtype=torch.float32)
    for index, length in enumerate(lengths):
        mask[index, :length] = 1.

    return mask.to(device)


def set_parameter_requires_grad(model, feature_extracting):
    r"""
    Config for the pretrained model, to use an existing model with the parameter frozen.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_video_length(video_filename):
    video = cv2.VideoCapture(video_filename)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return count


def get_filename_from_full_path(full_path):
    return full_path.split(os.sep)[-1]


def get_filename_from_a_folder_given_extension(folder, extension):
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            file_list.append(os.path.join(folder, file))

    return file_list


def load_single_pkl(directory, filename, extension='.pkl'):
    r"""
    Load one pkl file according to the filename.
    """
    fullname = os.path.join(directory, filename + extension)
    fullname = glob.glob(fullname)[0]
    return load_pkl_file(fullname)


def save_pkl_file(directory, filename, data):
    os.makedirs(directory, exist_ok=True)
    fullname = os.path.join(directory, filename)

    with open(fullname, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)


def load_pkl_file(full_directory):
    r"""
    Load one pkl file according to the absolute directory.
    """
    with open(full_directory, 'rb') as f:
        pkl_file = pickle.load(f)
        return pkl_file


def video_to_video_hauler(input_name, output_name):
    r"""
    Copy an  file to another directory.
    """
    if not os.path.isfile(output_name):
        shutil.copy(input_name, output_name)


def h5_to_h5_hauler(input_h5, output_h5):
    r"""
    Copy a h5 file to another directory.
    """
    if not os.path.isfile(output_h5):
        shutil.copy(input_h5, output_h5)


def copy_file(input_filename, output_filename):
    if not os.path.isfile(output_filename):
        shutil.copy(input_filename, output_filename)

def ndarray_to_txt_hauler(input_ndarray, output_txt, column_name, time_interval):
    r"""
    Copy a txt file to a new directory. The old txt file has no headers. Thew new txt file will have
        headers for a column.
    It is designed to re-format the label file in txt format.
    """
    if not os.path.isfile(output_txt):

        with open(output_txt, "w") as txt_file:
            first_line_string = " ".join(column_name) + "\n"
            txt_file.write(first_line_string)

            for index, value in enumerate(input_ndarray[0]):
                string = "{} {}\n".format(time_interval * (index + 1), value)
                txt_file.write(string)


def visualize_frame(*args):
    r"""
    Visualize the frame and the landmarks if any.
    """
    frame = args[0]
    if args[1].all():
        landmark = args[1]
        for (x1, x2) in landmark:
            cv2.circle(frame, (x1, x2), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


def print_progress(*args):
    # The string to show the progress.
    if len(args) == 4:
        string = "Processed: {}/{}, files: {}/{} , {}%".format(
            str(args[0] + 1), str(args[1]),
            str(args[2] + 1), str(args[3]),
            str(np.round((args[0] + 1) / args[3] * 100, 2))
        )
    elif len(args) == 2:
        string = "Processed: {}/{} , {}%".format(
            str(args[0] + 1), str(args[1]),
            str(np.round((args[0] + 1) / args[1] * 100, 2))
        )

    print("\r" + string, end='', flush=True)


def batch_count_for_each_subject(config):
    r"""
    Count the video batch of a subject given the batch size. For example, Subject A has 10 videos, and the batch_size = 4,
        then 3 batches are counted for this subject.
    """
    dataset_info_filename = glob.glob(config['root_directory'] + "*" + config['dataset_info_extension'])[0]
    with open(dataset_info_filename, 'rb') as pkl_file:
        dataset_info = pickle.load(pkl_file)
    subjects, counts = np.unique(dataset_info['subject'], return_counts=True)
    # subject_to_batch_count = np.c_[subjects, np.int64(np.ceil(counts / config['batch_size']))]
    subject_to_batch_count = dict(zip(subjects, np.int64(np.ceil(counts / config['batch_size']))))
    return subjects, subject_to_batch_count


def read_semaine_xml_for_dataset_info(xml_file, role):
    r"""
    Read the seamin xml log by XPath, so that the session_id, subject_id,
        subject_role and feeltrace boolean can be obtained.
    :param xml_file: (list), the xml file list recording the session information.
    :param role: (list), the role of a subject can be.
    :return: the dictionary obtained, recording the session_id, subject_id,
        subject_role and whether this session has the continuous label.
    """
    contain_continuous_valence_label = 0
    contain_continuous_arousal_label = 0
    feeltrace_bool = 0
    role_dict = {"User": 0, "Operator": 1}

    session_id = xml_file.find('.').attrib['sessionId']

    role_string = './/subject[@role="' + role + '"]'
    subject_id = xml_file.find(role_string).attrib["id"]

    subject_role = role_dict[role]

    target_string = "TU"
    if role == "Operator":
        target_string = "TO"

    feeltrace_string = './/track[@type="AV"]/annotation[@type="FeelTrace"]'

    annotation_tag = xml_file.findall(feeltrace_string)

    if annotation_tag:
        for tag in annotation_tag:
            if "DV.txt" in tag.attrib['filename'] and target_string in tag.attrib['filename']:
                contain_continuous_valence_label = 1
            if "DA.txt" in tag.attrib['filename'] and target_string in tag.attrib['filename']:
                contain_continuous_arousal_label = 1

    if contain_continuous_valence_label == contain_continuous_arousal_label == 1:
        feeltrace_bool = 1

    info = {"session_id": int(session_id),
            "subject_id": int(subject_id),
            "subject_role": int(subject_role),
            "feeltrace_bool": int(feeltrace_bool)}

    return info


def dict_combine(main_dict, new_dict):
    r"""
    Combine two dictionaries having the same keys.
    :param main_dict: (dict), the main dictionary to be appended.
    :param new_dict: (dict), the given dictionary to append.
    :return: (dict), the combined dictionary.
    """
    for (key, value) in main_dict.items():
        if new_dict[key] != [] or "":
            main_dict[key].append(new_dict[key])

    return main_dict


def generate_trial_info(dataset_info):
    r"""
    Generate unrepeated trial index given the subject index.
        The Semaine has not a existing records on trials of a same subject.
        Therefore, this function is used to generate so that the Subject-trial index is unique for each session.
    :param dataset_info: (dict), the dictionary recording the information of the dataset.
    :return: (dict), a new dictionary having a new key named "trial_id".
    """
    trial_info = np.zeros_like(dataset_info['subject_id'])
    unique_subject_array, count = np.unique(dataset_info['subject_id'], return_counts=True)

    for idx, subject in enumerate(unique_subject_array):
        indices = np.where(dataset_info['subject_id'] == subject)[0]
        trial_info[indices] = np.arange(1, count[idx] + 1, 1)

    dataset_info['trial_id'] = trial_info
    return dataset_info

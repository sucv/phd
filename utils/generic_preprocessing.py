import re
from pathlib import Path
from utils.preprocessing_static_function import *
from utils.base_class import *
from utils.dataset import *
from utils.helper import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv
import pickle
from PIL import Image
import scipy.io as sio
import xml.etree.ElementTree as et


class GenericPreprocessingAffectNet(object):
    def __init__(self, config):
        self.root_directory = config['local_root_directory']
        self.image_folder = config['local_image_folder']
        self.output_directory = config['local_output_directory']
        self.label_csv_list = ["validation", "training"]
        self.emotion_dict = {0: "Neutral", 1: "Happy", 2: "Sad",
                             3: "Surprise", 4: "Fear", 5: "Disgust",
                             6: "Anger", 7: "Contempt", 8: "None",
                             9: "Uncertain", 10: "Non-Face"}

        self.selected_emotion = {"Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"}
        self.config = config
        self.image_preprocessing()

    def image_preprocessing(self):

        landmark_handler = facial_image_crop_by_landmark(self.config)

        for label_csv_file in self.label_csv_list:
            partition = label_csv_file

            csv_data = load_single_csv(self.root_directory, label_csv_file, ".csv")
            num_rows = len(csv_data.index)
            for index, data in csv_data.iterrows():

                image_fullname = os.path.join(
                    self.root_directory, self.image_folder, data[0].split("/")[0], data[0].split("/")[1])

                emotion = self.emotion_dict[data[6]]

                if emotion in self.selected_emotion:
                    output_directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
                    os.makedirs(output_directory, exist_ok=True)

                    output_fullname = os.path.join(output_directory, data[0].split("/")[0] + "_" + data[0].split("/")[1])

                    if not os.path.isfile(output_fullname):
                        landmark = self.restore_landmark_to_ndarray(data[5])

                        img_ndarray = np.array(Image.open(image_fullname))
                        croped_image = landmark_handler.crop_image(img_ndarray, landmark)

                        croped_image = Image.fromarray(croped_image)

                        croped_image.save(output_fullname, "JPEG")

                print_progress(index, num_rows)


    @staticmethod
    def restore_landmark_to_ndarray(landmark_string):
        landmark = np.fromstring(landmark_string, sep=";").astype(np.float).reshape((68, 2))
        return landmark


class GenericPreprocessingRAFDB(object):
    def __init__(self, config):
        self.root_directory = config['local_root_directory']
        self.image_folder = config['local_image_folder']
        self.output_directory = config['local_output_directory']
        self.emotion_dict = {1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Sadness", 6: "Surprise", 7: "Neutral"}
        self.image_preprocessing()

    @staticmethod
    def load_txt(directory, filename, extension=".txt"):

        fullname = os.path.join(directory, filename + extension)
        data = pd.read_csv(fullname, sep=" ", header=None)
        return data

    def load_label(self):
        label = self.load_txt(self.root_directory, "list_patition_label")
        return label

    def image_preprocessing(self):
        label = self.load_label()
        directory = os.path.join(self.root_directory, self.image_folder)

        for index, data in label.iterrows():
            image_fullname = os.path.join(directory, data[0].split(".jpg")[0] + "_aligned" + ".jpg")
            partition = data[0].split("_")[0]
            emotion = self.emotion_dict[data[1]]
            output_directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
            os.makedirs(output_directory, exist_ok=True)
            output_fullname = os.path.join(output_directory, data[0].split(".jpg")[0] + "_aligned" + ".jpg")
            if not os.path.isfile(output_fullname):
                copy_file(image_fullname, output_fullname)

            print_progress(index, 15339)


class GenericPreprocessingFER2013(object):
    def __init__(self, config):
        self.root_directory = config['local_root_directory']
        self.root_csv_filename = config['root_csv_filename']
        self.output_directory = config['local_output_directory']
        self.emotion_dict = self.init_emotion_dict()
        self.partition_dict = {"Training": "Train", "PrivateTest": "Validate", "PublicTest": "Test"}
        self.image_preprocessing()

    def init_emotion_dict(self):
        emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
        return emotion_dict

    def load_csv(self, extension):
        r""""
        FER2013 is stored in a csv file.
        """
        csv_data = load_single_csv(
            directory=self.root_directory, filename=self.root_csv_filename, extension=extension)

        return csv_data

    @staticmethod
    def restore_img_from_csv_row(pixel_array):
        img = np.fromstring(pixel_array, dtype=int, sep=" ").reshape((48, 48)).astype(np.uint8)
        img = Image.fromarray(img, 'L')
        return img

    def image_preprocessing(self):
        csv_data = self.load_csv(".csv")

        for index, row in csv_data.iterrows():
            emotion = self.emotion_dict[row['emotion']]
            pixels = row['pixels']
            partition = self.partition_dict[row['Usage']]

            directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
            os.makedirs(directory, exist_ok=True)

            fullname = os.path.join(directory, str(index) + ".jpg")
            if not os.path.isfile(fullname):
                img = self.restore_img_from_csv_row(pixels)
                img.save(fullname, "JPEG")

            print_progress(index, 35887)

class GenericPreprocessingFerPlus(GenericPreprocessingFER2013):
    def __init__(self, config):
        super().__init__(config)

    def init_emotion_dict(self):

        emotion_dict = {0: "Neutral", 1: "Happiness", 2: "Surprise", 3: "Sadness", 4: "Anger", 5: "Disgust", 6: "Fear",
                        7: "Contempt"}
        return emotion_dict

    def image_preprocessing(self):
        fer_csv_data = self.load_csv(".csv")
        fer_plus_csv_data = self.load_csv("new.csv")

        for (index, fer_row), (_, fer_plus_row) in zip(fer_csv_data.iterrows(), fer_plus_csv_data.iterrows()):

            emotion_pointer = np.argmax(fer_plus_row[2:10].array)
            if emotion_pointer < 8:
                labels = np.array(fer_plus_row[2:10].array)
                max_vote = np.max(labels)
                if max_vote > 5:
                # max_vote_emotion = np.where(labels == max_vote)[0]
                # num_max_votes = max_vote_emotion.size
                # if not (num_max_votes >= 3) and not (num_max_votes * max_vote <= 0.5 * num_votes[ii]):

                    emotion = self.emotion_dict[emotion_pointer]
                    pixels = fer_row['pixels']
                    partition = self.partition_dict[fer_row['Usage']]

                    directory = os.path.join(self.root_directory, self.output_directory, partition, emotion)
                    os.makedirs(directory, exist_ok=True)

                    fullname = os.path.join(directory, str(index) + ".jpg")
                    if not os.path.isfile(fullname):
                        img = self.restore_img_from_csv_row(pixels)
                        img.save(fullname, "JPEG")

            print_progress(index, 35887)

class GenericPreprocessingCKplus(object):
    def __init__(self, config):
        self.root_directory = config['root_directory']
        self.image_folder = config['image_folder']
        self.label_folder = config['label_folder']
        self.output_directory = config['output_directory']
        self.openface_config = config['openface_config']

        self.image_sequence_preprocessing()

    def get_label_list(self):
        label_directory = os.path.join(self.root_directory, self.label_folder)
        label_list = [str(path) for path in Path(label_directory).rglob('*.txt')]
        return label_list

    def image_sequence_preprocessing(self):
        openface_directory = self.openface_config['openface_directory']
        input_flag = self.openface_config['input_flag']
        output_features = self.openface_config['output_features']
        output_action_unit = self.openface_config['output_action_unit']
        output_image_flag = self.openface_config['output_image_flag']
        output_image_size = self.openface_config['output_image_size']
        output_image_format = self.openface_config['output_image_format']
        output_filename_flag = self.openface_config['output_filename_flag']
        output_directory_flag = self.openface_config['output_directory_flag']
        output_image_mask_flag = self.openface_config['output_image_mask_flag']

        label_list = self.get_label_list()

        for index, label_file in enumerate(label_list):
            subject_id = self.get_subject_id(label_file)
            emotion_category = self.get_label(label_file)

            image_sequence_directory = self.get_image_sequence_directory(label_file)

            output_filename = emotion_category
            output_directory = os.path.join(self.output_directory, subject_id)
            os.makedirs(output_directory, exist_ok=True)

            # The actual command to call.
            command = openface_directory + input_flag + image_sequence_directory + output_features \
                      + output_action_unit + output_image_flag + output_image_size \
                      + output_image_format + output_filename_flag + output_filename \
                      + output_image_mask_flag + output_directory_flag + output_directory

            # If the file having the same name already exists, then skip the call.
            if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
                subprocess.call(command, shell=True)

            print_progress(index, len(label_file))



    def get_image_sequence_directory(self, label_filename):
        label_filename_parts = label_filename.split(os.sep)
        label_filename_parts[2] = self.image_folder
        image_sequence_directory = os.sep.join(label_filename_parts[:-1])
        return image_sequence_directory
    @staticmethod
    def get_subject_id(label_filename):
        subject_id = label_filename.split(os.sep)[3]
        return subject_id

    @staticmethod
    def get_label(label_filename):
        emotion_code = np.loadtxt(label_filename)

        if emotion_code == 0:
            emotion_category = "Neutral"
        elif emotion_code == 1:
            emotion_category = "Anger"
        elif emotion_code == 2:
            emotion_category = "Contempt"
        elif emotion_code == 3:
            emotion_category = "Disgust"
        elif emotion_code == 4:
            emotion_category = "Fear"
        elif emotion_code == 5:
            emotion_category = "Happiness"
        elif emotion_code == 6:
            emotion_category = "Sad"
        elif emotion_code == 7:
            emotion_category = "Surprise"
        else:
            sys.exit("Unknown expression code!")

        return emotion_category


class GenericPreprocessingOuluCISIA(object):
    def __init__(self, config):
        self.root_directory = config['root_directory']
        self.output_directory = config['output_directory']
        self.openface_config = config['openface_config']
        self.get_video_list()
        self.video_preprocessing()

    def video_preprocessing(self):

        openface_directory = self.openface_config['openface_directory']
        input_flag = self.openface_config['input_flag']
        output_features = self.openface_config['output_features']
        output_action_unit = self.openface_config['output_action_unit']
        output_image_flag = self.openface_config['output_image_flag']
        output_image_size = self.openface_config['output_image_size']
        output_image_format = self.openface_config['output_image_format']
        output_filename_flag = self.openface_config['output_filename_flag']
        output_directory_flag = self.openface_config['output_directory_flag']
        output_image_mask_flag = self.openface_config['output_image_mask_flag']

        video_list = self.get_video_list()

        for index, file in enumerate(video_list):
            if " " in file:
                file = '"' + file + '"'

            subject_id = self.get_subject_id(file)
            emotion_category = self.get_expression_category(file)
            output_filename = emotion_category
            output_directory = os.path.join(self.output_directory, subject_id)
            os.makedirs(output_directory, exist_ok=True)


            # The actual command to call.
            command = openface_directory + input_flag + file + output_features \
                      + output_action_unit + output_image_flag + output_image_size \
                      + output_image_format + output_filename_flag + output_filename \
                      + output_image_mask_flag + output_directory_flag + output_directory

            # If the file having the same name already exists, then skip the call.
            if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
                subprocess.call(command, shell=True)

            print_progress(index, len(video_list))

    def get_video_list(self):
        r"""
        Get the videos captured under strong illumination.
        """
        reg_compile = re.compile(r'.+_S_.+')
        video_list = [os.path.join(self.root_directory, file) for file in os.listdir(self.root_directory) if reg_compile.match(file)]
        return video_list

    @staticmethod
    def get_subject_id(video_name):
        subject_id = video_name.split(os.sep)[-1].split("_")[1]
        return subject_id

    @staticmethod
    def get_expression_category(video_name):
        emotion_code = video_name.split(os.sep)[-1].split("_")[-1].split(".avi")[0]

        if emotion_code == "A":
            emotion_category = "Anger"
        elif emotion_code == "D":
            emotion_category = "Disgust"
        elif emotion_code == "F":
            emotion_category = "Fear"
        elif emotion_code == "H":
            emotion_category = "Happiness"
        elif emotion_code == "S1":
            emotion_category = "Surprise"
        elif emotion_code == "S2":
            emotion_category = "Sad"
        else:
            sys.exit("Unknown expression code!")

        return emotion_category

class GenericPreprocessingRAFD(GenericPreprocessingOuluCISIA):
    def __init__(self, config):
        super().__init__(config)

    def get_video_list(self):
        directory = os.path.join(self.root_directory, "raw_data")
        video_list = [os.path.join(directory, image) for image in os.listdir(directory)]
        return video_list

    @staticmethod
    def get_expression_category(video_name):
        emotion_category = video_name.split(os.sep)[-1].split("_")[4].split(".avi")[0].capitalize()
        return emotion_category

    @staticmethod
    def get_gaze(video_name):
        gaze = video_name.split(os.sep)[-1].split("_")[5].split(".avi")[0].capitalize()[:-4]
        return gaze

    def video_preprocessing(self):

        openface_directory = self.openface_config['openface_directory']
        input_flag = self.openface_config['input_flag']
        output_features = self.openface_config['output_features']
        output_action_unit = self.openface_config['output_action_unit']
        output_image_flag = self.openface_config['output_image_flag']
        output_image_size = self.openface_config['output_image_size']
        output_image_format = self.openface_config['output_image_format']
        output_filename_flag = self.openface_config['output_filename_flag']
        output_directory_flag = self.openface_config['output_directory_flag']
        output_image_mask_flag = self.openface_config['output_image_mask_flag']

        video_list = self.get_video_list()

        for index, file in enumerate(video_list):
            if " " in file:
                file = '"' + file + '"'

            subject_id = self.get_subject_id(file)
            emotion_category = self.get_expression_category(file)
            gaze = self.get_gaze(file)
            output_filename = emotion_category + "_" + gaze
            output_directory = os.path.join(self.output_directory, subject_id)
            os.makedirs(output_directory, exist_ok=True)


            # The actual command to call.
            command = openface_directory + input_flag + file + output_features \
                      + output_action_unit + output_image_flag + output_image_size \
                      + output_image_format + output_filename_flag + output_filename \
                      + output_image_mask_flag + output_directory_flag + output_directory

            # If the file having the same name already exists, then skip the call.
            if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
                subprocess.call(command, shell=True)

            processed_filefullname = os.path.join(output_directory, output_filename)
            self.copy_paste(processed_filefullname)

            print_progress(index, len(video_list))

    def copy_paste(self, filename):
        image_fullname = os.path.join(filename + "_aligned", "frame_det_00_000001.jpg")

        subject_id = filename.split(os.sep)[3]
        emotion_category = filename.split(os.sep)[4].split("_")[0]
        new_filename = filename.split(os.sep)[4].split("_")[1] + ".jpg"
        new_directory = os.path.join(self.root_directory, "Cropped", subject_id, emotion_category)
        os.makedirs(new_directory, exist_ok=True)
        output_image_fullname =os.path.join(new_directory, new_filename)
        copy_file(image_fullname, output_image_fullname)

class GenericPreprocessing:
    r""" A general preprocessing class for emotional video. Currently it has the following functions:
    1., recoding the video to a desired frame rate while preserving the video duration.
    2., trimming the video according to the continuous annotation;
    3., extracting the 68 facial fiducial points for each video frame;
    4., cropping each frame according to the bounding box (currently determined by the fiducial points);
    """

    def __init__(self, config):

        # The root directory of the downloaded dataset.
        self.root_directory = config['root_directory']

        # The modalities to process.
        self.modal = config['modal']

        # The regular expression of the file name.
        self.filename_pattern = config['filename_pattern']

        # To how many frames is a label corresponds.
        self.label_to_video_ratio = config['label_to_video_ratio']

        # The config for facial_landmark_extraction.
        self.facial_landmark_extraction_config = config['facial_landmark_extraction']

        #
        self.landmark_alignment_config = config['facial_landmark_alignment']

        self.landmark_lib_config = config['landmark_lib']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the continuous label for each trial.
        self.continuous_label = self.get_continuous_label()

        # Obtain the data length for each trial.
        self.dataset_info['continuous_label_length'] = self.get_continuous_label_length()

        # Obtain the trimmed length of the video for each trial.
        self.dataset_info['trim_length'] = self.get_video_trimmed_length()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        self.change_video_fps_config = config['change_video_fps']
        # Carry out the video preprocessing,
        # Alter the fps ---> Trim the video ---> Extract the facial fiducial points
        # ---> Obtain the bounding box ---> crop the video frames
        self.video_list, self.landmark_list = self.video_preprocessing()

        self.dataset_folder_structuring_config = config['dataset_folder_structuring']

        self.dataset_folder_structuring()

    def count_session(self):
        session_number = len(self.dataset_info["session_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        :return:  the session_id, subject_id and trial_id of the dataset (dict).
        """
        directory = os.path.join(self.root_directory, "Sessions")
        session_id = np.asarray(
            sorted([idx for idx in os.listdir(directory) if os.path.isdir(os.path.join(directory, idx))], key=float),
            dtype=int)
        subject, trial = session_id // 130 + 1, session_id % 130

        dataset_info = {'session_id': session_id, 'subject': subject, 'trial': trial}
        return dataset_info

    def get_file_list_by_pattern(self, pattern):
        r"""
        :param pattern:  the regular expression of a file name (str). :return: the list of file names which satisfy
        the pattern (list), usually all the videos, or annotation files, etc.
        """
        dataset_info = self.dataset_info
        file_list = []

        # Iterate over the sessions.
        # For each session, find the file that matches the pattern, and store the file name in the python list.
        for index in range(self.session_number):
            directory = os.path.join(self.root_directory, "Sessions", str(dataset_info['session_id'][index]))

            # If the pattern is fixed.
            file_pattern = pattern

            # If the pattern contains variables, then fill the bracket accordingly.
            if "{}" in pattern:
                file_pattern = pattern.format(dataset_info['subject'][index], dataset_info['trial'][index])

            # Carry out the regular expression matching
            reg_compile = re.compile(file_pattern)
            filename = [os.path.join(directory, file) for file in os.listdir(directory) if reg_compile.match(file)]
            if filename:
                file_list.append(filename[0])

        # If nothing found after the iteration, then the file should be single and located in the parent directory.
        if len(file_list) == 0:
            file_list = [os.path.join(self.root_directory, pattern)]

        return file_list

    def get_continuous_label(self):
        r"""
        :return: the continuous labels for each trial (dict).
        """

        label_file = self.get_file_list_by_pattern(self.filename_pattern['continuous_label'])
        mat_content = sio.loadmat(label_file[0])
        annotation_cell = np.squeeze(mat_content['labels'])

        label_list = []
        for index in range(self.session_number):
            label_list.append(annotation_cell[index])
        return label_list

    def get_continuous_label_length(self):
        r"""
        :return: the length of the continuous labels for each trial (dict).
        """

        lengths = np.zeros(self.session_number, dtype=int)
        for index in range(self.session_number):
            lengths[index] = self.continuous_label[index].shape[1]

        return lengths

    def get_video_length(self):
        r"""
        :return: the length (frame count) of videos for each trial (dict).
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])
        lengths = np.zeros((len(video_list)), dtype=int)
        for index, video_file in enumerate(video_list):
            video = cv2.VideoCapture(video_file)
            lengths[index] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return lengths

    def get_video_trimmed_length(self):
        r"""
        :return: the frame range of a video that corresponds to its annotation (dict).
        By default, the range starts from 0.
        """
        lengths = self.dataset_info['continuous_label_length'] * self.label_to_video_ratio
        return lengths

    def get_video_trimming_range(self):
        zero = np.zeros((len(self.dataset_info['trim_length']), 1), dtype=int)

        ranges = np.c_[zero, self.dataset_info['trim_length']]
        ranges = ranges[:, np.newaxis, :]

        return ranges

    def video_preprocessing(self):
        r"""
        :return: carry out the preprocessing for videos.
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])

        # Change the fps of the video to a specified one which can match
        # the resolution of annotation tools.
        param = self.change_video_fps_config
        video_list = change_video_fps(video_list, **param)

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=False, visualize=False)

        # Extract the facial landmarks for a video.
        param = self.facial_landmark_extraction_config
        landmark_list = facial_landmark_extraction_by_dlib_lk(
            self.root_directory, video_list, self.dataset_info, self.landmark_lib_config, **param)

        # Cropping the facial region by aligning the facial landmarks to a template with the desired size.
        param = self.landmark_alignment_config
        video_list, landmark_list = landmark_alignment(
            self.root_directory, video_list, landmark_list, self.dataset_info, self.landmark_lib_config, **param)

        return video_list, landmark_list

    def dataset_folder_structuring(self):
        r"""
        Structure all the preprocessed files by a subject/trial/files manner.
        subject1
            trial2.mp4 (video)
            trial2.h5 (facial landmark)
            trial2.txt (continuous label)
            trial4.mp4
            trial4.h5
            trial4.txt
            ...
        subject2
            trial6.mp4
            trial6.h5
            trial6.txt
            ...
        """

        output_directory = self.dataset_folder_structuring_config['output_directory']
        subject, trial = self.dataset_info['subject'], self.dataset_info['trial']

        pkl_filename = os.path.join(self.root_directory, "dataset_info.pkl")
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(self.dataset_info, pkl_file, pickle.HIGHEST_PROTOCOL)

        for index in range(self.session_number):
            folder = os.path.join(self.root_directory, output_directory, str(subject[index]))
            os.makedirs(folder, exist_ok=True)
            filename = str(trial[index])

            # Generate files for continuous labels.
            output_continuous_label_filename = os.path.join(folder, filename + '.txt')
            continuous_label_ndarray = self.continuous_label[index]
            ndarray_to_txt_hauler(continuous_label_ndarray,
                                  output_continuous_label_filename,
                                  self.dataset_folder_structuring_config['continuous_label_column'],
                                  time_interval=0.25)

            # Generate files for videos.
            output_video_filename = os.path.join(folder, filename + '.mp4')
            input_video_filename = self.video_list[index]
            video_to_video_hauler(input_video_filename, output_video_filename)

            # Generate files for landmarks.
            output_landmark_filename = os.path.join(folder, filename + '.h5')
            input_landmark_filename = self.landmark_list[index]
            h5_to_h5_hauler(input_landmark_filename, output_landmark_filename)

            string = "Structured: {}/{} session.".format(str(index + 1), str(self.session_number))
            print(string)


class GenericPreprocessingMahnobFull:
    r""" A general preprocessing class for emotional video. Currently it has the following functions:
    1., recoding the video to a desired frame rate while preserving the video duration.
    2., trimming the video according to the continuous annotation;
    3., extracting the 68 facial fiducial points for each video frame;
    4., cropping each frame according to the bounding box (currently determined by the fiducial points);
    5., the eeg preprocessing for filtering, independent component analysis, and mean reference.
    """

    def __init__(self, config):

        # The root directory of the downloaded dataset.
        self.root_directory = config['root_directory']

        self.frame_size = config['frame_size']

        # The folder to store the output files.
        self.output_folder = config['output_folder']

        # The modalities to process.
        self.modal = config['modal']

        # The regular expression of the file name.
        self.filename_pattern = config['filename_pattern']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        self.continuous_label_list = self.get_continuous_label()
        self.get_continuous_label_bool()
        self.get_eeg_bool()
        self.get_eeg_length()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        self.change_video_fps_config = config['change_video_fps']

        # The config for the powerful openface.
        self.openface_config = config['openface_config']

        # Carry out the video preprocessing,
        # Trim the video ---> Extract the facial fiducial points
        # ---> Obtain the bounding box ---> crop the video frames
        self.video_preprocessing()

        self.dataset_info['output_folder'] = self.get_output_folder_list()
        # Carry out the eeg preprocessing.
        # Filter the signal by bandpass filter ---> Filter the signal by notch filter
        # ---> independent component analysis ---> mean reference
        self.eeg_preprocessing()



        # Get the length (the amount of images in each folder)
        self.dataset_info['processed_length'] = self.get_processed_video_length()

        self.dataset_info['refined_processed_length'] = self.refine_processed_video_length()


    def get_output_folder_list(self):
        output_folder_list = []
        for folder in self.dataset_info['processed_folder']:
            output_folder = folder.replace("processed", "compacted_" + str(self.frame_size))
            output_folder_list.append(output_folder)
        return output_folder_list


    def count_session(self):
        r"""
        Count the sessions.
        :return: (int), the amount of sessions.
        """
        session_number = len(self.dataset_info["session_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        :return:  the session_id, subject_id and trial_id of the dataset (dict).
        """
        directory = os.path.join(self.root_directory, "Sessions")
        session_id = np.asarray(
            sorted([idx for idx in os.listdir(directory) if os.path.isdir(os.path.join(directory, idx))], key=float),
            dtype=int)
        subject, trial = session_id // 130 + 1, session_id % 130

        dataset_info = {'session_id': session_id, 'subject_id': subject, 'trial_id': trial}
        return dataset_info

    def get_continuous_label_bool(self):
        continuous_label_mat_file = os.path.join(self.root_directory, "lable_continous_Mahnob.mat")
        self.dataset_info['having_continuous_label'] = np.zeros(len(self.dataset_info['session_id']), dtype=np.int32)
        mat_content = sio.loadmat(continuous_label_mat_file)
        sessions_having_continuous_label = mat_content['trials_included']
        unique_subject_index = [np.where(self.dataset_info['subject_id'] == n)[0][0] for n in np.unique(self.dataset_info['subject_id'])]

        for index in range(len(sessions_having_continuous_label)):
            subject, trial = sessions_having_continuous_label[index]
            start_idx = np.where(self.dataset_info['subject_id'] == subject)[0]
            offset = np.where(self.dataset_info['trial_id'][start_idx] == trial)[0][0]

            self.dataset_info['having_continuous_label'][start_idx[offset]] = 1

    def get_eeg_bool(self):
        r"""
        Some trials have no eeg recording. This function will indicate it.
        :return: (list), the binary to indicate the availability of eeg bdf file.
        """
        eeg_bool_list = []
        for folder in self.dataset_info['session_id']:
            flag = 0
            directory = os.path.join(self.root_directory, "Sessions", str(folder))
            for file in os.listdir(directory):
                if "emotion.bdf" in file:
                    flag = 1
            eeg_bool_list.append(flag)

        self.dataset_info['having_eeg'] =eeg_bool_list

    def get_eeg_length(self):

        eeg_length_list = []
        for folder in tqdm(self.dataset_info['session_id']):
            directory = os.path.join(self.root_directory, "Sessions", str(folder))
            length = 1e10
            for file in os.listdir(directory):
                if "emotion.bdf" in file:
                    eeg_file = os.path.join(directory, file)
                    raw_data = mne.io.read_raw_bdf(eeg_file, verbose=False)
                    length = raw_data.n_times - 256 * 60
            eeg_length_list.append(length)
        self.dataset_info['eeg_length'] = eeg_length_list

    def get_continuous_label(self):
        r"""
        :return: the continuous labels for each trial (dict).
        """

        label_file = os.path.join(self.root_directory, "lable_continous_Mahnob.mat")
        mat_content = sio.loadmat(label_file)
        annotation_cell = np.squeeze(mat_content['labels'])

        label_list = []
        for index in range(len(annotation_cell)):
            label_list.append(annotation_cell[index].T)
        return label_list

    def get_file_list_by_pattern(self, pattern):
        r"""
        :param pattern:  the regular expression of a file name (str). :return: the list of file names which satisfy
        the pattern (list), usually all the videos, or annotation files, etc.
        """
        dataset_info = self.dataset_info
        file_list = []

        # Iterate over the sessions.
        # For each session, find the file that matches the pattern, and store the file name in the python list.
        for index in range(self.session_number):
            directory = os.path.join(self.root_directory, "Sessions", str(dataset_info['session_id'][index]))

            # If the pattern is fixed.
            file_pattern = pattern

            # If the pattern contains variables, then fill the bracket accordingly.
            if "{}" in pattern:
                file_pattern = pattern.format(dataset_info['subject_id'][index], dataset_info['trial_id'][index])
                if "bdf" in pattern:
                    file_pattern = pattern.format(dataset_info['subject_id'][index],
                                                  dataset_info['trial_id'][index] // 2)

            # Carry out the regular expression matching
            reg_compile = re.compile(file_pattern)
            filename = [os.path.join(directory, file) for file in os.listdir(directory) if reg_compile.match(file)]
            if filename:
                file_list.append(filename[0])

        # If nothing found after the iteration, then the file should be single and located in the parent directory.
        if len(file_list) == 0:
            file_list = [os.path.join(self.root_directory, pattern)]

        return file_list

    def get_video_length(self):
        r"""
        :return: the length (frame count) of videos for each trial (dict).
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])
        lengths = np.zeros((len(video_list)), dtype=int)
        for index, video_file in enumerate(video_list):
            video = cv2.VideoCapture(video_file)
            lengths[index] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return lengths

    def get_video_trimming_range(self):
        r"""
        Get the trimming intervial of the videos.
        :return: (ndarray), the ranges to guide the video trimming.
        """
        intermediate_range_pkl_file = os.path.join(self.root_directory, "intermediate_range.pkl")
        if not os.path.isfile(intermediate_range_pkl_file):
            record_file_list = self.get_file_list_by_pattern(self.filename_pattern['timestamp'])
            ranges = read_start_end_from_mahnob_tsv(record_file_list)
            with open(intermediate_range_pkl_file, 'wb') as f:
                pickle.dump(ranges, f)
        else:
            with open(intermediate_range_pkl_file, 'rb') as f:
                ranges = pickle.load(f)
        return ranges

    def get_processed_video_length(self):
        r"""
        After changing the fps and processing the video using OpenFace, the video length
            has been changed, therefore we have to get the length again.
        :return: (list) the length list of the processed video in the format of static images.
        """

        length_list = []

        if not os.path.isfile(os.path.join(self.root_directory, 'processed_video_length.pkl')):
            for folder in tqdm(self.dataset_info['processed_folder']):
                csv_file = folder + ".csv"
                length = len(pd.read_csv(csv_file))
                length_list.append(length)
            save_pkl_file(self.root_directory, 'processed_video_length.pkl', length_list)
        else:
            length_list = load_pkl_file(os.path.join(self.root_directory, 'processed_video_length.pkl'))

        return length_list

    def refine_processed_video_length(self):
        r"""
        Some trials have continuous labels while others are not. This function will refine the ones
            having continuous labels, so that the length will be corresponding with the continuous
            label length and frame-to-label ratio.
        :return: (list) the length list of the processed video in the format of static images.
        """
        length_list = []
        pointer = 0
        for index in tqdm(range(len(self.dataset_info['trial_id']))):
            video_length = self.dataset_info['processed_length'][index]
            eeg_length = self.dataset_info['eeg_length'][index]
            if self.dataset_info['having_continuous_label'][index]:
                length = len(self.continuous_label_list[pointer])
                fine_length = min(length, video_length // 16, eeg_length // 64)
                pointer += 1
            else:
                fine_length = min(video_length // 16, eeg_length // 64)

            length_list.append(fine_length)

        return length_list


    def eeg_preprocessing(self):
        r"""
        :return: Carry out the eeg preprocessing.
        """
        eeg_bdf_list = self.get_file_list_by_pattern(self.filename_pattern['eeg'])
        pointer = 0
        for index in tqdm(range(len(self.dataset_info['session_id']))):

            if self.dataset_info['having_eeg'][index]:
                output_directory = self.dataset_info['output_folder'][index]
                os.makedirs(output_directory, exist_ok=True)
                output_file = os.path.join(output_directory, "eeg_raw.npy")

                if not os.path.isfile(output_file):
                    eeg_handler = GenericEegPreprocessingMahnob(eeg_bdf_list[pointer], buffer=5)
                    eeg_data = eeg_handler.filtered_data
                    np.save(output_file, eeg_data)

                pointer += 1


    def video_preprocessing(self):
        r"""
        :return: carry out the preprocessing for videos.
        """
        video_list = self.get_file_list_by_pattern(self.filename_pattern['video'])

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=False, visualize=False)

        # Change the fps of the video to a integer.
        param = self.change_video_fps_config
        video_list = change_video_fps(video_list, **param)

        # Extract facial landmark, warp, crop, and save each frame.
        param = self.openface_config
        file_list = facial_video_preprocessing_by_openface(
            self.root_directory, self.output_folder, self.dataset_info, video_list, **param)

        # Save the static folders
        self.dataset_info['processed_folder'] = file_list


class GenericPreprocessingAVEC19:

    def __init__(self, config):
        self.config = config
        self.root_directory = config['local_root_directory']
        self.raw_data_folder = config['raw_data_folder']
        self.output_folder = config['data_folder']
        self.compact_folder = config['compact_folder']
        self.partition_list = config['partition_list']
        self.country_list = config['country_list']
        self.feature_list = config['feature_list']
        self.emotion_dimension = config['emotion_dimension']
        self.downsampling_interval = config['downsampling_interval']
        self.target_fps = config['target_fps']
        self.time_depth = config['time_depth']
        self.step_size = config['step_size']
        self.frame_number_to_compact = config['frame_number_to_compact']
        self.frame_size = config['frame_size']
        self.openface_config = config['openface_config']

        self.dataset_info = self.init_dataset_info()
        self.establish_dataset_info()

        self.video_preprocessing()
        self.feature_preprocessing()
        self.label_preprocessing()

        self.create_npy_for_frame()
        self.create_npy_for_success()
        self.create_npy_for_continuous_label()
        # self.compact_data_for_n_fold_cross_validation()
        print(0)

    def establish_dataset_info(self):
        raw_directory = os.path.join(self.root_directory, self.raw_data_folder)
        participant_trial_to_partition_country_map, partition_country_to_participant_trial_map = {}, {}
        subject_id = 1
        for partition in self.partition_list:
            directory = os.path.join(raw_directory, partition)
            # video_list = get_filename_from_a_folder_given_extension(directory, "avi")
            label_list = get_filename_from_a_folder_given_extension(directory, "csv")
            for raw_fullname in label_list:
                raw_filename = get_filename_from_full_path(raw_fullname).split('.csv')[0]
                raw_video_fullname = raw_fullname.split(".csv")[0] + ".avi"
                new_filename = "P" + str(subject_id).zfill(2) + "-T01"
                participant_trial_to_partition_country_map[raw_filename] = new_filename
                partition_country_to_participant_trial_map[new_filename] = raw_filename
                this_partition = raw_filename.split("_")[0]
                country = raw_filename.split("_")[1]
                self.dataset_info['subject_id'].append(subject_id)
                self.dataset_info['trial_id'].append(1)
                self.dataset_info['frame_count'].append(get_video_length(raw_video_fullname) // 5 * 5)
                self.dataset_info['partition'].append(this_partition)
                self.dataset_info['country'].append(country)
                self.dataset_info['feeltrace_bool'].append(1)
                subject_id += 1
        self.dataset_info['participant_trial_to_partition_country_map'] = participant_trial_to_partition_country_map
        self.dataset_info['partition_country_to_participant_trial_map'] = partition_country_to_participant_trial_map
        self.dataset_info['subject_id'] = np.asarray(self.dataset_info['subject_id'])
        self.dataset_info['trial_id'] = np.asarray(self.dataset_info['trial_id'])
        self.dataset_info['feeltrace_bool'] = np.asarray(self.dataset_info['feeltrace_bool'])
        if not os.path.isfile(os.path.join(self.root_directory, 'dataset_info.pkl')):
            save_pkl_file(self.root_directory, 'dataset_info.pkl', self.dataset_info)

    def init_dataset_info(self):
        dataset_info = {
            "participant_trial_to_partition_country_map": [],
            "partition_country_to_participant_trial_map": [],
            "subject_id": [],
            "trial_id": [],
            "frame_count": [],
            "partition": [],
            "country": [],
            "feeltrace_bool": [],

        }

        return dataset_info

    # @staticmethod
    # def get_country_and_subject_id(filename):
    #     info = filename.split("_")
    #     country, subject_id = info[1], info[2].split(".avi")[0]
    #     return country, subject_id

    # def get_subject_trial_info(self):
    #
    #     raw_directory = os.path.join(self.root_directory, self.raw_data_folder)
    #     for partition in self.partition_list:
    #         directory = os.path.join(raw_directory, partition)
    #         video_list = get_filename_from_a_folder_given_extension(directory, "avi")
    #         label_list = get_filename_from_a_folder_given_extension(directory, "csv")
    #
    #         if label_list:
    #             for label_fullname in label_list:
    #                 filename = get_filename_from_full_path(label_fullname)
    #                 country, subject_id = self.get_country_and_subject_id(filename)
    #                 self.dataset_info[partition][country]['label_fullname'].append(label_fullname)
    #
    #         for video_fullname in video_list:
    #             filename = get_filename_from_full_path(video_fullname)
    #             country, subject_id = self.get_country_and_subject_id(filename)
    #             self.dataset_info[partition][country]['trial_id'].append(1)
    #             self.dataset_info[partition][country]['subject_id'].append(subject_id)
    #             self.dataset_info[partition][country]['video_fullname'].append(video_fullname)

    # def get_video_length_info(self):
    #     for partition_name, partition_dict in self.dataset_info.items():
    #         for country_name, country_dict in partition_dict.items():
    #             intermediate_list = []
    #             for video_fullname in country_dict['video_fullname']:
    #
    #                 # The last label cannot fully correspond to the video frames, so
    #                 # the last 5 frames are not counted.
    #                 count = get_video_length(video_fullname) // 5 * 5
    #                 intermediate_list.append(count)
    #
    #             self.dataset_info[partition_name][country_name]['frame_count'] = intermediate_list

    # def file_copy(self):
    #     for partition_name in self.partition_list:

    def video_preprocessing(self):
        video_list = []
        param = self.openface_config
        subject_id = 1
        for partition_name in self.partition_list:
            if partition_name != "Test":
                directory = os.path.join(self.root_directory, self.raw_data_folder, partition_name)
                label_list = get_filename_from_a_folder_given_extension(directory, "csv")
                for raw_fullname in label_list:
                    new_name = "P" + str(subject_id).zfill(2) + "-T01"
                    raw_name = self.dataset_info['partition_country_to_participant_trial_map'][new_name]

                    raw_fullname = os.path.join(self.root_directory, self.raw_data_folder, partition_name, raw_name + '.avi')

                    output_folder = os.path.join(self.root_directory, self.output_folder)
                    os.makedirs(output_folder, exist_ok=True)


                    facial_video_preprocessing_by_openface_for_AVEC19(
                        self.root_directory, self.output_folder, new_name, raw_fullname, **param)

                    subject_id += 1



    def feature_preprocessing(self):
        directory = os.path.join(self.root_directory, self.output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "T01.csv")
        for csv_file in csv_list:
            output_csv_success_indices_fullname = csv_file.split(".csv")[0] + "_success.csv"
            if not os.path.isfile(output_csv_success_indices_fullname):
                frame_indices = pd.read_csv(csv_file,
                                            skipinitialspace=True, usecols=["success"],
                                            index_col=False).values.squeeze()
                # success_frame_indices = np.where(frame_indices == 1)[0]
                # Saving the indices indicating the successful frames.
                data_frame = pd.DataFrame(frame_indices, columns=["success"])
                data_frame.to_csv(output_csv_success_indices_fullname, index=False)

    def label_preprocessing(self):

        for partition_name in self.partition_list:
            directory = os.path.join(self.root_directory, self.raw_data_folder, partition_name)
            label_list = get_filename_from_a_folder_given_extension(directory, ".csv")
            for label_fullname in label_list:
                raw_filename = get_filename_from_full_path(label_fullname).split(".csv")[0]
                new_filename = self.dataset_info['participant_trial_to_partition_country_map'][raw_filename] + "_continuous_label.csv"
                new_fullname = os.path.join(self.root_directory, self.output_folder, new_filename)
                if not os.path.isfile(new_fullname):
                    copy_file(label_fullname, new_fullname)

    def create_npy_for_continuous_label(self):
        directory = os.path.join(self.root_directory, self.output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "_continuous_label.csv")

        for index, csv_file in enumerate(csv_list):
            npy_directory = os.path.join(self.root_directory, self.compact_folder, csv_file.split(os.sep)[-1].split("_continuous_label.csv")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_continuous_label = os.path.join(npy_directory, "continuous_label.npy")

            cols = [emotion.lower() for emotion in self.emotion_dimension]

            if not os.path.isfile(npy_filename_continuous_label):
                continuous_label = pd.read_csv(csv_file, sep=";",
                                            skipinitialspace=True, usecols=cols,
                                            index_col=False).values.squeeze()
                continuous_label = continuous_label[:self.dataset_info['frame_count'][index] // 5]

                with open(npy_filename_continuous_label, 'wb') as f:
                    np.save(f, continuous_label)

            print_progress(index, 96)

    def create_npy_for_success(self):
        directory = os.path.join(self.root_directory, self.output_folder)
        csv_list = get_filename_from_a_folder_given_extension(directory, "T01.csv")

        for index, csv_file in enumerate(csv_list):
            npy_directory = os.path.join(self.root_directory, self.compact_folder, csv_file.split(os.sep)[-1].split(".csv")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_success = os.path.join(npy_directory, "success.npy")
            if not os.path.isfile(npy_filename_success):
                success_indices = pd.read_csv(csv_file,
                                            skipinitialspace=True, usecols=["success"],
                                            index_col=False).values.squeeze()
                success_indices = success_indices[:self.dataset_info['frame_count'][index]]

                with open(npy_filename_success, 'wb') as f:
                    np.save(f, success_indices)

            print_progress(index, 96)

    def create_npy_for_frame(self):

        directory = os.path.join(self.root_directory, self.output_folder)
        trials_folder = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

        for i, folder in enumerate(trials_folder):
            npy_directory = os.path.join(self.root_directory, self.compact_folder, folder.split(os.sep)[-1].split("_aligned")[0])
            os.makedirs(npy_directory, exist_ok=True)
            npy_filename_frame = os.path.join(npy_directory, "frame.npy")
            if not os.path.isfile(npy_filename_frame):
                frame_list = get_filename_from_a_folder_given_extension(folder, ".jpg")[:self.dataset_info['frame_count'][i]]
                frame_length = len(frame_list)
                frame_matrix = np.zeros((frame_length, 112, 112, 3), dtype=np.uint8)

                for j, frame in enumerate(frame_list):
                    frame_matrix[j] = Image.open(frame)

                with open(npy_filename_frame, 'wb') as f:
                    np.save(f, frame_matrix)

            print_progress(i, 96)



    def compact_data_for_n_fold_cross_validation(self):
        r"""
        To accelerate the data loading, instead of reading frame-by-frame, we seek to
            first compact the frame into one mp4 file, then load it in one go. This function is
            for this purpose.
        """
        folder_arranger = NFoldArranger(self.root_directory)

        # Get the session id for each folds.
        session_id_of_all_folds = folder_arranger.assign_session_to_subject()

        # Get the subjects having the continuous labels.
        data_arranger = DataArrangerAVEC19(session_id_of_all_folds, self.config)
        subjects_having_feeltrace = data_arranger.get_subject_id_having_feeltrace()

        # Generate the dictionary in a one-subject-one-fold manner. So that we can later
        # load them for whatever n-fold partition we want.
        # This is a long function. It basically does three tasks. First, for each subject, it generates all the
        # frame files to be loaded, with and without downsampling (i.e., the compacted_dict argument).
        # Second, it count the number of frames considered for each session with and without downsampling
        # (i.e., the sliced_length_dict). Third, it generates a dictionary saving the relationship
        # among subjects, video clips and samples (i.e., the subject_clip_sample_info).
        # The latter is crucial to restore the mini-batched output/labels to the shapes of complete session,
        # subject-wise, and partition-wise (train, validate & test are the so-called partition).
        compacted_dict, sliced_length_dict, subject_clip_sample_info = data_arranger.generate_data_filename_and_label_array(
            session_id_of_all_folds, downsampling_interval=self.downsampling_interval)

        # Save the compacted data. The reason to save the data is for time-saving. If we read the data
        # frame-by-frame one the fly, it is way too slow, whereas reading them as clip-by-clip (each as a
        # compacted mp4 file) is much faster.
        # A mp4 is called a video clip. It can contain one or more samples. A sample contains time_depth frames.
        # A sample is the minimum unit during the fitting and testing. Two samples may or may not have overlap.
        # A good overlapping ratio can improve the regression result, usually.
        self.save_compacted_data(
            subjects_having_feeltrace, sliced_length_dict, compacted_dict, subject_clip_sample_info)

    def save_compacted_data(
            self, subjects_having_feeltrace, sliced_length_dict, compacted_dict, subject_clip_sample_info):
        r"""
        This function load the frame files in the dictionary then save them to mp4 files.
        """
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        scale = "timedepth_" + str(self.time_depth) + "_stepsize_" + str(self.step_size)

        directory_old = os.path.join(
            self.root_directory, self.compact_folder, str(
                self.downsampling_interval), scale, str(self.frame_number_to_compact))

        # Save the session-wise length of a subject to a pkl file for later use.
        save_pkl_file(directory_old, 'subject_sliced_session_length.pkl', sliced_length_dict)

        # Save the sample to session map to a pkl file for later use. It will help to restore the output segment
        # during network training and testing. Only the restored output can be used to
        # visualize the output-to-continuous-label line graph.
        save_pkl_file(directory_old, 'subject_clip_sample_map.pkl', subject_clip_sample_info)

        for subject_relative_id, data_for_one_subject in compacted_dict.items():
            subject_id = subjects_having_feeltrace[subject_relative_id]

            directory = os.path.join(directory_old, str(subject_id))
            os.makedirs(directory, exist_ok=True)

            for batch, data_for_one_batch in enumerate(data_for_one_subject):
                for key, value in data_for_one_batch.items():

                    filename = key + "_" + str(batch).zfill(4) + "_" + str(len(value))
                    absolute_path = os.path.join(directory, filename)

                    if key == 'frame':
                        extension = ".mp4"
                        fullname = absolute_path + extension
                        if not os.path.isfile(fullname):

                            # Initialize the video writer.
                            writer = cv2.VideoWriter(fullname, codec, self.target_fps,
                                                     (self.frame_size, self.frame_size), isColor=True)

                            for index, path in enumerate(data_for_one_batch[key]):
                                image = cv2.imread(path)
                                writer.write(image)
                                print_progress(index, self.frame_number_to_compact)

                            writer.release()

                    else:
                        extension = ".h5"
                        fullname = absolute_path + extension
                        if not os.path.isfile(fullname):
                            data = value
                            h5f = h5py.File(fullname, 'w')
                            h5f.create_dataset(key, data=data, compression="gzip", compression_opts=9)
                            h5f.close()

class GenericPreprocessingSemaine:
    r""" A general preprocessing class for the Semaine dataset. Currently it has the following functions:
    1., recoding the video to a desired frame rate while preserving the video duration.
    2., trimming the video according to the continuous annotation;
    3., extracting the 68 facial fiducial points for each video frame;
    4., cropping each frame to a desired size and saving as images;
    5., saving the continuous labels as csv files.
    """

    def __init__(self, config):

        self.config = config

        # The root directory of the downloaded dataset.
        self.root_directory = config['root_directory']

        # The folder to store the output files.
        self.output_folder = config['data_folder']

        self.compact_folder = config['compact_folder']

        # The regular expression of the file name.
        self.filename_pattern = config['filename_pattern']

        # To how many frames is a label corresponds.
        self.label_to_video_ratio = config['label_to_video_ratio']

        # The emotional dimension of the dataset.
        self.emotion_dimension = config['emotion_dimension']

        # The intermediate folder to save the read continuous labels.
        # This is for time saving because there are too many labels!
        # Cannot read them everytime when debugging.
        self.intermediate_folder = config['intermediate_folder']

        # The config for the powerful openface.
        self.openface_config = config['openface_config']

        #
        self.frame_size = config['frame_size']
        self.frame_number_to_compact = config['frame_number_to_compact']

        #
        self.downsampling_interval = config['downsampling_interval']

        #
        self.fold_number = config['fold_number_preprocessing']

        self.time_depth = config['time_depth']

        self.step_size = config['step_size']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the continuous label for each trial.
        self.continuous_label = self.get_continuous_label()

        # Obtain the data length for each trial.
        self.dataset_info['continuous_label_length'] = self.get_continuous_label_length()

        # Obtain the trimmed length of the video for each trial.
        self.dataset_info['trim_length'] = self.get_video_trimmed_length()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        self.change_video_fps_config = config['change_video_fps']

        # Carry out the video preprocessing,
        # Alter the fps ---> Trim the video ---> Extract the facial fiducial points
        #  ---> crop the video frames ---> save to images
        self.video_preprocessing()

        # Carryout the label preprocessing.
        # Perform the CCC centering ---> Save both the continuous label
        # and the success indicator to csv files.
        self.label_preprocessing()

        # Save the dataset information. It is important because in can provide a consistent processing order.
        # For example, in Windows and Ubuntu, the sort can has different results, which results in a different
        # subject_id, trial_id orders. If the openface and sort are performed in Windows, and the sort is
        # performed in Ubuntu again, then the dataset information from the two system will be different, so that
        # the information will not correspond to the openface output.
        self.save_dataset_info()

        self.compact_data_for_n_fold_cross_validation()

    def count_session(self):
        r"""
        Count the total of the sessions.
        :return: (int), the total of the sessions.
        """
        session_number = len(self.dataset_info["session_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        Get the session_id, subject_id, subject_role and the feeltrace indicators.
        :return: (dict), the dictionary of the dataset information.
        """
        dataset_info = {"session_id": [],
                        "subject_id": [],
                        "subject_role": [],
                        "feeltrace_bool": []}

        directory = os.path.join(self.root_directory, "Sessions")

        for sub_folder in os.listdir(directory):
            xml_file = et.parse(os.path.join(directory, sub_folder, self.filename_pattern['session_log'])).getroot()

            user_info = read_semaine_xml_for_dataset_info(xml_file, "User")
            dataset_info = dict_combine(dataset_info, user_info)

            operator_info = read_semaine_xml_for_dataset_info(xml_file, "Operator")
            dataset_info = dict_combine(dataset_info, operator_info)

        dataset_info = {key: np.asarray(value) for key, value in dataset_info.items()}

        # Note, the sort_indices can be different between Windows and Ubuntu!!
        # So that the dataset_info has to be saved and copied to a different OS for consistency.
        sort_indices = np.argsort(dataset_info['subject_id'])
        dataset_info = {key: value[sort_indices] for key, value in dataset_info.items()}
        dataset_info = generate_trial_info(dataset_info)

        # If the dataset_info already exists, read it! This is for the above mentioned consistency!
        dataset_info_filename = os.path.join(self.root_directory, "dataset_info.pkl")
        if os.path.isfile(dataset_info_filename):
            with open(dataset_info_filename, 'rb') as f:
                existing_dataset_info = pickle.load(f)
            dataset_info.update(existing_dataset_info)

        return dataset_info

    def get_video_trimmed_length(self):
        r"""
        :return: the frame range of a video that corresponds to its annotation (dict).
        By default, the range starts from 0.
        """
        lengths = self.dataset_info['continuous_label_length'] * self.label_to_video_ratio
        return lengths

    def get_video_trimming_range(self):
        zero = np.zeros((len(self.dataset_info['trim_length']), 1), dtype=int)

        ranges = np.c_[zero, self.dataset_info['trim_length']]
        ranges = ranges[:, np.newaxis, :]

        return ranges

    def get_label_dict_by_pattern(self, pattern):
        r"""
        Get the dictionary storing the continuous labels.
        :param pattern: (string), the pattern of the files.
        :return: (dict), the filename dictionary of the continuous labels for users and operators.
        """
        folder = os.path.join(self.root_directory, "Sessions")
        dataset_info = self.dataset_info
        label_dict = {key: [] for key in self.emotion_dimension}
        role_dict = {0: "User", 1: "Operator"}

        for i in range(self.session_number):
            if dataset_info['feeltrace_bool'][i]:
                directory = os.path.join(folder, str(dataset_info['session_id'][i]))

                for emotion in self.emotion_dimension:

                    file_pattern = pattern[role_dict[dataset_info['subject_role'][i]]][emotion]
                    reg_compile = re.compile(file_pattern)

                    filename = sorted([os.path.join(directory, file) for file
                                       in os.listdir(directory) if reg_compile.match(file)])
                    if filename:
                        label_dict[emotion].append(filename)

        return label_dict

    def get_video_list_by_pattern(self, pattern):
        r"""
        Get the dictionary storing the videos having the continuous labels.
        :param pattern: (string), the pattern of the files.
        :return: (list), the filename list of the videos. Both the users and operators are taken as subjects,
            therefore a list not dictionary is suitable.
        """
        folder = os.path.join(self.root_directory, "Sessions")
        dataset_info = self.dataset_info
        video_list = []
        role_dict = {0: "User", 1: "Operator"}

        for index in range(self.session_number):
            if dataset_info['feeltrace_bool'][index]:
                directory = os.path.join(folder, str(dataset_info['session_id'][index]))
                file_pattern = pattern[role_dict[dataset_info['subject_role'][index]]]
                reg_compile = re.compile(file_pattern)
                filename = [os.path.join(directory, file) for file
                            in os.listdir(directory) if reg_compile.match(file)][0]
                video_list.append(filename)

        return video_list

    def get_intermediate_continuous_label(self):
        r"""
        Save the continuous label to disk from the memory. For time-saving during the dubbing.
        :return: (string), (string), the filename of the files saving the intermediate continuous labels and their length.
        """
        intermediate_label_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                   "intermediate_label.pkl")
        intermediate_label_length_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                          "intermediate_label_length.pkl")
        os.makedirs(os.path.join(self.root_directory, self.intermediate_folder), exist_ok=True)
        label_dict = {key: [] for key in self.emotion_dimension}
        label_length_dict = {key: [] for key in self.emotion_dimension}
        label_file = self.get_label_dict_by_pattern(self.filename_pattern['continuous_label'])

        if not os.path.isfile(intermediate_label_filename):
            for emotion in self.emotion_dimension:
                emotion_label_file = label_file[emotion]
                subject_level_list, subject_level_length_list = [], []

                for count, subject_level in enumerate(emotion_label_file):
                    rater_level_list = []
                    rater_level_list_length = []
                    for rater_level in subject_level:
                        label = np.loadtxt(rater_level)[:, 1]
                        rater_level_list_length.append(len(label))
                        rater_level_list.append(label)
                    min_length = min(rater_level_list_length)
                    subject_level_length_list.append(min_length)
                    rater_level_list = [label[:min_length] for label in rater_level_list]
                    subject_level_list.append(np.stack(rater_level_list))
                    print(count)

                label_dict[emotion] = subject_level_list
                label_length_dict[emotion] = subject_level_length_list

            with open(intermediate_label_filename, 'wb') as f:
                pickle.dump(label_dict, f)

            with open(intermediate_label_length_filename, 'wb') as f:
                pickle.dump(label_length_dict, f)

        return intermediate_label_filename, intermediate_label_length_filename

    def get_continuous_label_length(self):
        r"""
        Read the length of the continuous label.
        :return: (ndarray), the minimum length across the two emotional dimensions for each session.
        """
        _, intermediate_label_length_filename = self.get_intermediate_continuous_label()

        with open(intermediate_label_length_filename, 'rb') as f:
            intermediate_label_length = pickle.load(f)
        min_length = np.min(np.vstack([value for value in intermediate_label_length.values()]), axis=0)

        return min_length

    def get_continuous_label(self):
        r"""
        Read the continuous labels.
        :return: (dict), the dict saving the continuous labels. It is really fast when directly reads them from a pkl file.
        """
        intermediate_label_filename = os.path.join(self.root_directory, self.intermediate_folder,
                                                   "intermediate_label.pkl")
        label_dict = {key: [] for key in self.emotion_dimension}

        min_length = self.get_continuous_label_length()

        with open(intermediate_label_filename, 'rb') as f:
            intermediate_label = pickle.load(f)

        for emotion in self.emotion_dimension:
            mat_cell = np.squeeze(intermediate_label[emotion])

            # This block is for reading a pkl file saving a dictionary. A directly reading without forloop
            # can only obtain the first list.
            label_list = []
            for index in range(len(min_length)):
                label_list.append(mat_cell[index][:, :min_length[index]])

            label_dict[emotion] = label_list

        return label_dict

    def video_preprocessing(self):
        r"""
        Carry out the video preprocessing.
        """
        video_list = self.get_video_list_by_pattern(self.filename_pattern['video'])

        # # Change the fps of the video to a integer.
        # param = self.change_video_fps_config
        # video_list = change_video_fps(video_list, **param)

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=False, visualize=False)

        # Extract facial landmark, warp, crop, and save each frame.
        param = self.openface_config
        file_list = facial_video_preprocessing_by_openface(
            self.root_directory, self.output_folder, self.dataset_info, video_list, **param)

    def label_preprocessing(self):
        r"""
        Carry out the label preprocessing. Here, since multiple raters are available, therefore
            concordance_correlation_coefficient_centering has to be performed.
        """
        centered_continuous_label_dict = {key: [] for key in self.emotion_dimension}

        for emotion in self.emotion_dimension:
            continuous_labels = self.continuous_label[emotion]
            for continuous_label in continuous_labels:
                centered_continuous_label = concordance_correlation_coefficient_centering(continuous_label)
                centered_continuous_label_dict[emotion].append(np.float32(np.mean(centered_continuous_label, axis=0)))

        continuous_label_to_csv(
            self.root_directory, self.output_folder, centered_continuous_label_dict,
            self.dataset_info)

    def compact_data_for_n_fold_cross_validation(self):
        r"""
        To accelerate the data loading, instead of reading frame-by-frame, we seek to
            first compact the frame into one mp4 file, then load it in one go. This function is
            for this purpose.
        """
        folder_arranger = NFoldArranger(self.root_directory)

        # Get the session id for each folds.
        session_id_of_all_folds = folder_arranger.assign_session_to_subject()

        # Get the subjects having the continuous labels.
        data_arranger = DataArrangerSemaine(session_id_of_all_folds, self.config)
        subjects_having_feeltrace = data_arranger.get_subject_id_having_feeltrace()

        # Generate the dictionary in a one-subject-one-fold manner. So that we can later
        # load them for whatever n-fold partition we want.
        # This is a long function. It basically does three tasks. First, for each subject, it generates all the
        # frame files to be loaded, with and without downsampling (i.e., the compacted_dict argument).
        # Second, it count the number of frames considered for each session with and without downsampling
        # (i.e., the sliced_length_dict). Third, it generates a dictionary saving the relationship
        # among subjects, video clips and samples (i.e., the subject_clip_sample_info).
        # The latter is crucial to restore the mini-batched output/labels to the shapes of complete session,
        # subject-wise, and partition-wise (train, validate & test are the so-called partition).
        compacted_dict, sliced_length_dict, subject_clip_sample_info = data_arranger.generate_data_filename_and_label_array(
            session_id_of_all_folds, downsampling_interval=self.downsampling_interval)

        # Save the compacted data. The reason to save the data is for time-saving. If we read the data
        # frame-by-frame one the fly, it is way too slow, whereas reading them as clip-by-clip (each as a
        # compacted mp4 file) is much faster.
        # A mp4 is called a video clip. It can contain one or more samples. A sample contains time_depth frames.
        # A sample is the minimum unit during the fitting and testing. Two samples may or may not have overlap.
        # A good overlapping ratio can improve the regression result, usually.
        self.save_compacted_data(
            subjects_having_feeltrace, sliced_length_dict, compacted_dict, subject_clip_sample_info)

    def save_compacted_data(
            self, subjects_having_feeltrace, sliced_length_dict, compacted_dict, subject_clip_sample_info):
        r"""
        This function load the frame files in the dictionary then save them to mp4 files.
        """
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        scale = "timedepth_" + str(self.time_depth) + "_stepsize_" + str(self.step_size)

        directory_old = os.path.join(
            self.root_directory, self.compact_folder, str(
                self.downsampling_interval), scale, str(self.frame_number_to_compact))

        # Save the session-wise length of a subject to a pkl file for later use.
        save_pkl_file(directory_old, 'subject_sliced_session_length.pkl', sliced_length_dict)

        # Save the sample to session map to a pkl file for later use. It will help to restore the output segment
        # during network training and testing. Only the restored output can be used to
        # visualize the output-to-continuous-label line graph.
        save_pkl_file(directory_old, 'subject_clip_sample_map.pkl', subject_clip_sample_info)

        for subject_relative_id, data_for_one_subject in compacted_dict.items():
            subject_id = subjects_having_feeltrace[subject_relative_id]

            directory = os.path.join(directory_old, str(subject_id))
            os.makedirs(directory, exist_ok=True)

            for batch, data_for_one_batch in enumerate(data_for_one_subject):
                for key, value in data_for_one_batch.items():

                    filename = key + "_" + str(batch).zfill(4) + "_" + str(len(value))
                    absolute_path = os.path.join(directory, filename)

                    if key == 'frame':
                        extension = ".mp4"
                        fullname = absolute_path + extension
                        if not os.path.isfile(fullname):

                            # Initialize the video writer.
                            writer = cv2.VideoWriter(fullname, codec, self.change_video_fps_config['target_fps'],
                                                     (self.frame_size, self.frame_size), isColor=True)

                            for index, path in enumerate(data_for_one_batch[key]):
                                image = cv2.imread(path)
                                writer.write(image)
                                print_progress(index, self.frame_number_to_compact)

                            writer.release()

                    else:
                        extension = ".h5"
                        fullname = absolute_path + extension
                        if not os.path.isfile(fullname):
                            data = value
                            h5f = h5py.File(fullname, 'w')
                            h5f.create_dataset(key, data=data, compression="gzip", compression_opts=9)
                            h5f.close()

    def save_dataset_info(self):
        r"""
        Save the dataset information for cross-operation-system consistency.
            It is required that the pre-processing is done in the same operation
            system, or even at the same computer. If not, the dataset info may be
            varied. (Because some operations like sorting are different in Windows
            and Linux, for an array having some equal elements, the sorting can be different,
            and cause a wrong dataset info.
        """
        pkl_filename = os.path.join(self.root_directory, "dataset_info.pkl")

        if not os.path.isfile(pkl_filename):
            with open(pkl_filename, 'wb') as pkl_file:
                pickle.dump(self.dataset_info, pkl_file, pickle.HIGHEST_PROTOCOL)


class GenericPreprocessingRecola:
    r""" A general preprocessing class for the Semaine dataset. Currently it has the following functions:
    1., recoding the video to a desired frame rate while preserving the video duration.
    2., trimming the video according to the continuous annotation;
    3., extracting the 68 facial fiducial points for each video frame;
    4., cropping each frame to a desired size and saving as images;
    5., saving the continuous labels as csv files.
    """

    def __init__(self, config):

        # The root directory of the downloaded dataset.
        self.root_directory = config['root_directory']

        self.output_folder = config['output_folder']

        # The regular expression of the file name.
        self.filename_pattern = config['filename_pattern']

        # To how many frames is a label corresponds.
        self.label_to_video_ratio = config['label_to_video_ratio']

        self.emotion_dimension = config['emotion_dimension']

        self.intermediate_folder = config['intermediate_folder']

        self.openface_config = config['openface_config']

        # The dictionary to store the session_id, subject_id, trial_id, data length, trimming range, etc.
        self.dataset_info = self.get_subject_trial_info()

        # The total of the sessions of the dataset.
        self.session_number = self.count_session()

        # Obtain the continuous label for each trial.
        self.continuous_label = self.get_continuous_label()

        # Obtain the data length for each trial.
        self.dataset_info['continuous_label_length'] = self.get_continuous_label_length()

        # Obtain the trimmed length of the video for each trial.
        self.dataset_info['trim_length'] = self.get_video_trimmed_length()

        # Obtain the trimming range of the video for each trial.
        self.dataset_info['trim_range'] = self.get_video_trimming_range()

        self.change_video_fps_config = config['change_video_fps']

        # Carry out the video preprocessing,
        # Alter the fps ---> Trim the video ---> Extract the facial fiducial points
        #  ---> crop the video frames ---> save to images
        self.video_preprocessing()

        # Carryout the label preprocessing.
        # Perform the CCC centering ---> Save both the continuous label
        # and the success indicator to csv files.
        self.label_preprocessing()

        # Save the dataset information. It is important because in can provide a consistent processing order.
        # For example, in Windows and Ubuntu, the sort can has different results, which results in a different
        # subject_id, trial_id orders. If the openface and sort are performed in Windows, and the sort is
        # performed in Ubuntu again, then the dataset information from the two system will be different, so that
        # the information will not correspond to the openface output.
        self.save_dataset_info()

    def count_session(self):
        r"""
        Count the total of the sessions.
        :return: (int), the total of the sessions.
        """
        session_number = len(self.dataset_info["subject_id"])
        return session_number

    def get_subject_trial_info(self):
        r"""
        Get the session_id, subject_id, subject_role and the feeltrace indicators.
        :return: (dict), the dictionary of the dataset information.
        """
        directory = os.path.join(self.root_directory, "RECOLA-Video-recordings")
        dataset_info = {"subject_id": np.asarray(
            [int(file_string[1:3]) for file_string in os.listdir(directory) if len(file_string) == 7])}

        dataset_info_filename = os.path.join(self.root_directory, "dataset_info.pkl")
        if os.path.isfile(dataset_info_filename):
            with open(dataset_info_filename, 'rb') as f:
                existing_dataset_info = pickle.load(f)
            dataset_info.update(existing_dataset_info)

        return dataset_info

    def get_video_trimmed_length(self):
        r"""
        :return: the total trimming length of a video. I.e., the sum of the trimming ranges.
        """
        lengths = self.dataset_info['continuous_label_length'] * self.label_to_video_ratio
        return lengths

    def get_video_trimming_range(self):
        r"""
        :return: the frame range of a video that corresponds to its annotation (dict).
        By default, the range starts from 0.
        """
        zero = np.zeros((len(self.dataset_info['trim_length']), 1), dtype=int)

        ranges = np.c_[zero, self.dataset_info['trim_length']]
        ranges = ranges[:, np.newaxis, :]

        return ranges

    def get_video_list_by_pattern(self, pattern):
        r"""
        Get the dictionary storing the videos having the continuous labels.
        :param pattern: (string), the pattern of the files.
        :return: (list), the filename list of the videos.
        """
        directory = os.path.join(self.root_directory, "RECOLA-Video-recordings")
        video_list = []

        for subject_id in self.dataset_info['subject_id']:
            video_file = os.path.join(directory, pattern.format(str(subject_id)))
            video_list.append(video_file)

        return video_list

    def get_continuous_label_length(self):
        r"""
        Read the length of the continuous label.
        :return: (ndarray), the minimum length across the two emotional dimensions for each session.
        """
        length_list = np.asarray([len(continuous_label) for continuous_label in self.continuous_label['arousal']])
        return length_list

    def get_continuous_label(self):
        r"""
        Read the continuous labels.
        :return: (dict), the dict saving the continuous labels. It is really fast when directly reads them from a pkl file.
        """
        folder = os.path.join(self.root_directory, "RECOLA-Annotation", "emotional_behaviour")
        label_dict = {'arousal': [], 'valence': []}
        for emotion in self.emotion_dimension:
            directory = os.path.join(folder, emotion)

            for subject_id in self.dataset_info['subject_id']:
                csv_file = os.path.join(directory, self.filename_pattern['continuous_label'].format(str(subject_id)))
                data = pd.read_csv(csv_file, delimiter=';')
                continuous_label = data.iloc[:, 1:].to_numpy()
                label_dict[emotion].append(continuous_label)

        return label_dict

    def video_preprocessing(self):
        r"""
            Carry out the video preprocessing.
            """
        video_list = self.get_video_list_by_pattern(self.filename_pattern['video'])

        # Change the fps of the video to a integer.
        param = self.change_video_fps_config
        video_list = change_video_fps(video_list, **param)

        # Pick only the annotated clips from a complete video.
        video_list = combine_annotated_clips(
            video_list, self.dataset_info['trim_range'], direct_copy=True, visualize=False)

        # Extract facial landmark, warp, crop, and save each frame.
        param = self.openface_config
        file_list = facial_video_preprocessing_by_openface(
            self.root_directory, self.output_folder, self.dataset_info, video_list, **param)

    def label_preprocessing(self):
        r"""
        Carry out the label preprocessing. Here, since multiple raters are available, therefore
            concordance_correlation_coefficient_centering has to be performed.
        """
        centered_continuous_label_dict = {key: [] for key in self.emotion_dimension}

        for emotion in self.emotion_dimension:
            continuous_labels = self.continuous_label[emotion]
            for continuous_label in continuous_labels:
                centered_continuous_label = concordance_correlation_coefficient_centering(continuous_label)
                centered_continuous_label_dict[emotion].append(np.mean(centered_continuous_label, axis=0))

        continuous_label_to_csv(
            self.root_directory, self.output_folder, centered_continuous_label_dict,
            self.dataset_info)

    def save_dataset_info(self):
        r"""
        Save the dataset information for cross-OS consistency.
        """
        pkl_filename = os.path.join(self.root_directory, "dataset_info.pkl")
        with open(pkl_filename, 'wb') as pkl_file:
            pickle.dump(self.dataset_info, pkl_file, pickle.HIGHEST_PROTOCOL)

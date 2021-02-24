import subprocess
import h5py
import sys
from utils.base_class import *
from utils.helper import *

import pandas as pd


def change_video_fps(
        videos,
        target_fps
):
    r"""
    Alter the frame rate of a given video.
    :param videos:  (list),a list of video files to process.
    :param target_fps:  (float), the desired fps.
    :return: (list), the list of video files after the process.
    """
    output_video_list = []

    # Iterate over the video list.
    for video_name in videos:

        # Define the new file name by adding the fps string at the rear before the extension.
        output_video_name = video_name[:-4] + '_fps' + str(target_fps) + video_name[-4:]

        # If the new name already belongs to a file, then do nothing.
        if os.path.isfile(output_video_name):
            # print("Skipped fps conversion for video {}!".format(str(index+1)))
            pass

        # If not, call the ffmpeg tools to change the fps.
        # -qscale:v 0 can preserve the quality of the frame after the recoding.
        else:
            input_codec = " xvid "
            if ".mp4" in video_name:
                input_codec = " mp4v "
            command = "ffmpeg -i {} -filter:v fps=fps={} -c:v mpeg4 -vtag {} -qscale:v 0 {}".format(
                '"' + video_name + '"', str(target_fps), input_codec,
                '"' + output_video_name + '"')
            subprocess.call(command, shell=True)

        output_video_list.append(output_video_name)
    return output_video_list


def combine_annotated_clips(
        videos,
        clip_ranges,
        direct_copy=False,
        visualize=False
):
    output_video_list = []

    for video_idx, input_video_name in enumerate(videos):

        # Define the new file name by adding the tag at the rear before the extension.
        output_video_name = input_video_name[:-4] + '_combined' + input_video_name[-4:]

        # If the new name already belongs to a file, then do nothing.
        if os.path.isfile(output_video_name):
            print("Skipped video combination for video {}!".format(str(video_idx + 1)))
            pass

        # If not, call the video combiner.
        else:
            if not direct_copy:
                video_split = VideoSplit(input_video_name, output_video_name, clip_ranges[video_idx])
                video_split.combine(visualize)
            else:
                video_to_video_hauler(input_video_name, output_video_name)

            print("Finished video combination for video {}/{}!".format(str(video_idx + 1), str(len(videos))))

        output_video_list.append(output_video_name)
    return output_video_list

def facial_video_preprocessing_by_openface_for_AVEC19(
        root_directory,
        output_folder,
        output_filename,
        file,
        openface_directory,
        input_flag,
        output_features,
        output_action_unit,
        output_image_format,
        output_image_mask_flag,
        output_image_flag,
        output_image_size,
        output_filename_flag,
        output_directory_flag
):
    r"""
    A function to call the powerful OpenFace for almost everything!
    :param output_action_unit: the flat indicating that the action units are to be outputted.
    :param root_directory: the root directory of the data to be processed.
    :param output_folder: the output folder to save the processed data.
    :param dataset_info: the dictionary recording the dataset information.
    :param files: the files to be processed.
    :param openface_directory: the directory where the OpenFace executable is located.
    :param input_flag: (str), the flag indicating that the next string is a filename.
    :param output_features: (str), the flag indicating that which feature(s) are being outputted.
    :param output_image_format: the flag indicating the format to save the frames.
    :param output_image_flag: the flag telling the OpenFace to output the split video, in a specific format of image.
    :param output_image_size: the size of the outputted images.
    :param output_filename_flag: the flag indicating that the next string is the name of the output files.
    :param output_directory_flag: the flag indicating that the next string is the output directory.
    :return: (list), the list saving the processed files according to the subject/trial index.
    """

    output_directory = os.path.join(root_directory, output_folder)
    os.makedirs(output_directory, exist_ok=True)

    # Quote the file name if spaces occurred.
    if " " in file:
        file = '"' + file + '"'


    # The actual command to call.
    command = openface_directory + input_flag + file + output_features \
              + output_action_unit + output_image_flag + output_image_size \
              + output_image_format + output_filename_flag + output_filename \
              + output_directory_flag + output_directory + output_image_mask_flag

    # # If the file having the same name already exists, then skip the call.
    # if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
    if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
        subprocess.call(command, shell=True)


def facial_video_preprocessing_by_openface(
        root_directory,
        output_folder,
        dataset_info,
        files,
        openface_directory,
        input_flag,
        output_features,
        output_action_unit,
        output_image_format,
        output_image_flag,
        output_image_size,
        output_filename_flag,
        output_directory_flag,
        output_image_mask_flag
):
    r"""
    A function to call the powerful OpenFace for almost everything!
    :param output_action_unit: the flat indicating that the action units are to be outputted.
    :param root_directory: the root directory of the data to be processed.
    :param output_folder: the output folder to save the processed data.
    :param dataset_info: the dictionary recording the dataset information.
    :param files: the files to be processed.
    :param openface_directory: the directory where the OpenFace executable is located.
    :param input_flag: (str), the flag indicating that the next string is a filename.
    :param output_features: (str), the flag indicating that which feature(s) are being outputted.
    :param output_image_format: the flag indicating the format to save the frames.
    :param output_image_flag: the flag telling the OpenFace to output the split video, in a specific format of image.
    :param output_image_size: the size of the outputted images.
    :param output_filename_flag: the flag indicating that the next string is the name of the output files.
    :param output_directory_flag: the flag indicating that the next string is the output directory.
    :return: (list), the list saving the processed files according to the subject/trial index.
    """

    # If dataset_info has no key named feeltrace_bool.
    indices_having_continuous_label = range(len(dataset_info['subject_id']))
    # Otherwise, exclude those indices having no continuous label.
    if 'feeltrace_bool' in dataset_info:
        indices_having_continuous_label = np.where(dataset_info['feeltrace_bool'] == 1)[0]

    output_directory = os.path.join(root_directory, output_folder)
    os.makedirs(output_directory, exist_ok=True)

    processed_file_list = []

    for index, file in enumerate(files):
        # Quote the file name if spaces occurred.
        if " " in file:
            file = '"' + file + '"'

        session_id = indices_having_continuous_label[index]

        # If the session pattern has no trial information.
        output_filename = "P{}".format(dataset_info['subject_id'][session_id])
        # Otherwise consider the trial information.
        if 'trial_id' in dataset_info:
            output_filename = "P{}-T{}".format(dataset_info['subject_id'][session_id],
                                               dataset_info['trial_id'][session_id])

        # The actual command to call.
        command = openface_directory + input_flag + file + output_features \
                  + output_action_unit + output_image_flag + output_image_size \
                  + output_image_format + output_filename_flag + output_filename \
                  + output_directory_flag + output_directory + output_image_mask_flag

        # If the file having the same name already exists, then skip the call.
        if not os.path.isfile(os.path.join(output_directory, output_filename + ".csv")):
            subprocess.call(command, shell=True)

        print_progress(index, len(files))

        # Record the processed filename.
        processed_file_list.append(os.path.join(output_directory, output_filename))

    return processed_file_list


def continuous_label_to_csv(
        root_directory,
        output_folder,
        continuous_labels,
        dataset_info,
):
    r"""
    Save the continuous label to csv files.
    :param root_directory: (str), the root directory of the dataset.
    :param output_folder: (str), the output folder.
    :param continuous_labels: (dict), the dictionary saving the continuous labels.
    :param dataset_info: (dict), the dictionary saving the dataset information.
    """

    # If feeltrace_bool is not contained in dataset_info
    indices_having_continuous_label = range(len(dataset_info['subject_id']))
    # Otherwise, exclude all indices having no continuous trace.
    if 'feeltrace_bool' in dataset_info:
        indices_having_continuous_label = np.where(dataset_info['feeltrace_bool'] == 1)[0]

    for index, session_id in enumerate(indices_having_continuous_label):

        # If no trial information is contained.
        csv_recording_file = "P{}".format(dataset_info['subject_id'][session_id])
        # Otherwise, fill in the trial_id.
        if 'trial_id' in dataset_info:
            csv_recording_file = "P{}-T{}".format(dataset_info['subject_id'][session_id],
                                                  dataset_info['trial_id'][session_id])

        csv_recording_filename = os.path.join(
            root_directory, output_folder, csv_recording_file + ".csv")

        output_csv_continuous_label_filename = os.path.join(
            root_directory, output_folder, csv_recording_file + "_continuous_label.csv")

        output_csv_success_indices_filename = os.path.join(
            root_directory, output_folder, csv_recording_file + "_success_indices.csv")

        # If the output file does not exist, save the csv file indicating the face
        # detection success of each frame, and also the csv file saving the continuous labels.
        if not os.path.isfile(output_csv_continuous_label_filename) \
                and not os.path.isfile(output_csv_continuous_label_filename):
            frame_indices = pd.read_csv(csv_recording_filename,
                                        skipinitialspace=True, usecols=["success"],
                                        index_col=False).values.squeeze()
            success_frame_indices = np.where(frame_indices == 1)[0]

            # Saving the continuous label for the successful and failed frames.
            continuous_labels_for_this_subject_trial = {emotion: data[index]
                                                        for emotion, data in continuous_labels.items()}

            data_frame = pd.DataFrame(data=continuous_labels_for_this_subject_trial)
            data_frame.to_csv(output_csv_continuous_label_filename, index=False)

            # Saving the indices indicating the successful frames.
            data_frame = pd.DataFrame(success_frame_indices, columns=["success"])
            data_frame.to_csv(output_csv_success_indices_filename, index=False)

        print_progress(index, len(indices_having_continuous_label))


def read_start_end_from_mahnob_tsv(tsv_file_list):
    r"""
    Get the start and end indices of the stimulated video frames by
        reading the tsv file from MAHNOB dataset.
    :param tsv_file_list: (list), the tsv file list.
    :return: (ndarray), the start and end indices of the stimulated frames,
        for later video trimming.
    """
    start_end_array = np.zeros((len(tsv_file_list), 1, 2), dtype=int)
    for index, tsv_file in enumerate(tsv_file_list):
        data = pd.read_csv(tsv_file, sep='\t', skiprows=23)
        end = data[data['Event'] == 'MovieEnd'].index[0]
        start_end_array[index, :, 1] = end
    return start_end_array


def eeg_independent_component_analysis(bdf_file_list, log_save_directory):
    r"""
    The eeg preprocessing.
    :param bdf_file_list: (list), the bdf file containing the physiological data to preprocess.
    :param log_save_directory: (str), the directory to save the log file.
    :return: (none), the preprocessed EEG signal will be saved in a specific h5 file for each session.
    """
    log_txt_filename = os.path.join(log_save_directory, "eeg_ica_log.txt")
    for index, bdf_file in enumerate(bdf_file_list):

        new_filename = bdf_file[:-4] + "_preprocessed" + ".h5"

        if not os.path.isfile(new_filename):

            # Carry out the EEG preprocessing.
            eeg = GenericEegPreprocessing(bdf_file)

            if os.path.isfile(log_txt_filename):
                # Read the log file.
                with open(log_txt_filename, 'a') as log_file:
                    # Write the interactive operations in a log file for further check.
                    log_file.write(eeg.log)
            else:
                with open(log_txt_filename, 'w') as log_file:
                    # Write the interactive operations in a log file for further check.
                    log_file.write(eeg.log)

            # Save the landmarks and bounding boxesin h5 format, with the largest compression ratio.
            h5f = h5py.File(new_filename, 'w')
            h5f.create_dataset('eeg', data=eeg.repaired_data[0], compression="gzip", compression_opts=9)
            h5f.create_dataset('time', data=eeg.repaired_data[1], compression="gzip", compression_opts=9)
            h5f.close()


def concordance_correlation_coefficient_centering(data):
    r"""
    Perform the CCC centering.
    :param data: (ndarray), the continuous labels to be centered.
    :return: (ndarray), the centered data.
    """
    CCC = ConcordanceCorrelationCoefficient(data)
    centered_data = CCC.centered_data
    return centered_data


# def facial_landmark_extraction_by_dlib_lk(
#         root_directory,
#         files,
#         info,
#         landmark_lib_config,
#         output_directory,
#         folder,
#         detection_interval=1,
#         detection_method='dlib',
#         tracking_method='lk',
#         output_landmark_file_extension='.h5',
#         visualize=False
# ):
#     r"""
#     Extract facial fiducial points using dlib and Lucas-Kanade algorithm from opencv.
#     :param landmark_lib_config: (dict), the config for the class landmark_lib.
#     :param root_directory:  (str), the root directory of the downloaded dataset.
#     :param files:  (list), the video list to be processed.
#     :param info:  (dict) the session_id, subject_id, trial_id, label_length, video_length, trim_length of the dataset.
#     :param output_directory: (str) the directory to store the output.
#     :param folder: (str) the folder of output_directory to store the landmark file.
#     :param detection_interval: (int) run the face detection method for each `detection_interval` frame(s),
#         tracking the others using lk algorithm. If detection_interval=1, then the tracking is not used, which produce the
#         best landmark locations at the great cost of time.
#     :param detection_method:  (str), the method for facial landmark localization.
#     :param tracking_method: (str), the method for tracking the points.
#     :param output_landmark_file_extension: (str), the format to store the extracted facial landmarks.
#     :param visualize: (binary), whether to show the landmarks for each frame.
#     :return: (list), the list of landmark files.
#     """
#     # Define then create the directory.
#     directory = os.path.join(root_directory, output_directory, folder)
#     os.makedirs(directory, exist_ok=True)
#
#     landmark_file_list = []
#
#     landmark_lib = GenericLandmarkLib(landmark_lib_config)
#
#     # Iterate over the video files
#     for file_idx, video_file in enumerate(files):
#
#         # Obtain the info for subject and trial.
#         subject = info['subject_id'][file_idx]
#         trial = info['trial_id'][file_idx]
#
#         # Define the file name of the landmark for this video.
#         landmark_filename = os.path.join(
#             directory,
#             "P{}-T{}".format(subject, trial) +
#             output_landmark_file_extension)
#
#         # If the file name already belongs to an existing file, then do nothing.
#         if os.path.isfile(landmark_filename):
#             pass
#
#         # Otherwise
#         else:
#
#             # The relevant part (corresponding to the annotation) of this video. The irrelevant part will be discard.
#             video_length = info['trim_length'][file_idx]
#
#             # Initialization
#             # ndarray for string the landmarks and bounding boxes.
#             landmark_tosave = np.zeros((68, 2, video_length), dtype=np.float32)
#             bbox_tosave = np.zeros((2, 2, video_length))
#
#             # Old frame and landmarks of the last frame.
#             frame_gray_old, point_old = 0, 0
#
#             # Call the opencv to read the video.
#             video = cv2.VideoCapture(video_file)
#
#             # Iterate over the frames of this video.
#             for frame_idx in range(video_length):
#
#                 # The string to show the progress.
#                 print_progress(frame_idx, video_length, file_idx, len(files))
#
#                 # Read one frame
#                 ret, frame = video.read()
#
#                 # If this frame is successfully read
#                 if ret:
#
#                     # Convert the frame to gray scale because the detection and tracking require so.
#                     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#                     # For every 'detection_interval'  frame(s), carry out the landmark detection method.
#                     if frame_idx % detection_interval == 0:
#                         landmark_new, bounding_box_new = landmark_lib.extract_landmark(gray_frame)
#
#                         if landmark_new is None:
#                             if frame_idx == 0:
#                                 sys.exit('Failed to detect the face at the 1st frame.\n'
#                                          'This video needs to be checked.')
#                             else:
#                                 landmark_new = landmark_lib.track_landmark(gray_frame_old, gray_frame, landmark_old)
#
#                     # For other frames, carry out the tracking for speeding up, at the cost of localization precision.
#                     else:
#                         # Call the Lucas-Kanade Algorithm.
#                         landmark_new = landmark_lib.track_landmark(gray_frame_old, gray_frame, landmark_old)
#
#                     # Collect useful variables.
#                     # Store the landmarks and bounding boxes detected.
#                     landmark_tosave[:, :, frame_idx] = landmark_new
#                     # For the bounding boxes, it is from the last detected frame.
#                     bbox_tosave[:, :, frame_idx] = bounding_box_new
#                     landmark_old = landmark_new
#
#                     # Show the landmarks for each frame.
#                     if visualize:
#                         visualize_frame(frame, landmark_new)
#                     # Save the last gray-scale frame.
#
#                     gray_frame_old = gray_frame.copy()
#
#                 # If opencv failed to read this frame
#                 else:
#                     sys.exit("Failed to read videos for Subject {} Trial {} Frame {}".format(subject, trial, frame_idx))
#
#             # Close the opencv visualization windows if any, and release the video reader.
#             cv2.destroyAllWindows()
#             video.release()
#
#             # Save the landmarks and bounding boxesin h5 format, with the largest compression ratio.
#             h5f = h5py.File(landmark_filename, 'w')
#             h5f.create_dataset('landmark', data=landmark_tosave, compression="gzip", compression_opts=9)
#             h5f.create_dataset('bbox', data=bbox_tosave, compression="gzip", compression_opts=9)
#             h5f.close()
#
#         # Save the h5 filename
#         landmark_file_list.append(landmark_filename)
#
#     return landmark_file_list
#
#
# def landmark_alignment(
#         root_directory,
#         video_files,
#         landmark_files,
#         info,
#         landmark_lib_config,
#         output_directory,
#         folder,
#         output_extension,
#         output_video_resolution=250,
#         visualize=False
# ):
#     r"""
#     Crop the videos to a target resolution which contain the facial region only. The cropping is done
#     by using affine transformation to align key landmarks onto the key template points and then
#     resampling the frame.
#     :param landmark_lib_config:
#     :param root_directory: (str), the root directory of the downloaded dataset.
#     :param video_files:  (list), the video files to be processed.
#     :param landmark_files:  (list), the landmark files to be processed.
#     :param info:  (dict) the session_id, subject_id, trial_id, label_length, video_length, trim_length of the dataset.
#     :param output_directory: (str) the directory to store the output.
#     :param folder: (dict) the folder of output_directory to store the landmark file.
#     :param output_extension:  (dict), the format of the output.
#     :param output_video_resolution:  (int), the desired cropping size.
#     :param visualize:  (boolean), whether to
#     :return:
#     """
#     video_folder = folder['video']
#     landmark_folder = folder['landmark']
#     output_video_extension = output_extension['video']
#     output_landmark_extension = output_extension['landmark']
#
#     # Directory to store the video files.
#     video_directory = os.path.join(root_directory, output_directory, video_folder)
#     os.makedirs(video_directory, exist_ok=True)
#     video_file_list = []
#
#     # Directory to store the landmark files.
#     landmark_directory = os.path.join(root_directory, output_directory, landmark_folder)
#     os.makedirs(landmark_directory, exist_ok=True)
#     landmark_file_list = []
#
#     # The resolution of the output video.
#     desired_frame_size = output_video_resolution
#
#     Landmark_lib = GenericLandmarkLib(landmark_lib_config)
#
#     # Define the codex of the video writer
#     codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#
#     for file_idx, (video_file, landmark_file) in enumerate(zip(video_files, landmark_files)):
#
#         # Read the subject and trial id.
#         subject = info['subject'][file_idx]
#         trial = info['trial'][file_idx]
#
#         # Define the output video and landmark file.
#         output_video_filename = os.path.join(
#             video_directory, "P{}-T{}".format(subject, trial) +
#                              output_video_extension)
#
#         output_landmark_filename = os.path.join(
#             landmark_directory, "P{}-T{}".format(subject, trial) +
#                                 output_landmark_extension)
#
#         # If the filename already belongs to an existing file, then do nothing.
#         if os.path.isfile(output_video_filename) and os.path.isfile(output_landmark_filename):
#             pass
#         # Otherwise, create and write it by the following operations.
#         else:
#
#             # Read the landmarks.
#             h5 = h5py.File(landmark_file, 'r')
#             landmarks = h5['landmark'][()]
#             h5.close()
#
#             # Initialize the landmark ndarray to save.
#             aligned_landmark_tosave = np.zeros_like(landmarks)
#
#             # Initialize the video reading and get the frame rate.
#             video = cv2.VideoCapture(video_file)
#             fps = video.get(cv2.CAP_PROP_FPS)
#
#             # Initialize the video writer.
#             writer = cv2.VideoWriter(output_video_filename, codec, fps,
#                                      (desired_frame_size, desired_frame_size), isColor=True)
#
#             video_length = landmarks.shape[2]
#             for frame_idx in range(video_length):
#
#                 # Read one frame
#                 ret, frame = video.read()
#
#                 # If read it successfully.
#                 if ret:
#
#                     # Get the affine transformation matrix H.
#                     affine_matrix = Landmark_lib.get_affine_matrix(landmarks[:, :, frame_idx])
#
#                     # Warp the frame using H. The frame will be resized to desired frame size.
#                     aligned_frame = Landmark_lib.align_frame(frame, affine_matrix)
#
#                     # Manually warp the landmarks using H.
#                     aligned_landmark = Landmark_lib.align_landmark(landmarks[:, :, frame_idx], affine_matrix)
#
#                     # Find the pixel indices located within the convex hull of landmark outline points.
#                     cropped_frame = Landmark_lib.crop_facial_region(aligned_frame, aligned_landmark)
#
#                     # Save the aligned landmarks.
#                     aligned_landmark_tosave[:, :, frame_idx] = aligned_landmark
#
#                     # Write the frame
#                     writer.write(cropped_frame)
#
#                     # The string to show the progress.
#                     print_progress(frame_idx, landmarks.shape[2], file_idx, len(video_files))
#
#                     # Show the cropped and resized frame, and the re-scaled landmarks.
#                     if visualize:
#                         visualize_frame(cropped_frame, aligned_landmark)
#
#             # Save the aligned landmarks.
#             h5f = h5py.File(output_landmark_filename, 'w')
#             h5f.create_dataset('landmark', data=aligned_landmark_tosave, compression="gzip", compression_opts=9)
#             h5f.close()
#
#             # Release the video reader and writer.
#             # the writer must be released for each video it wrote or the output cannot be opened!!
#             cv2.destroyAllWindows()
#             video.release()
#             writer.release()
#
#         # Record the files processed.
#         video_file_list.append(output_video_filename)
#         landmark_file_list.append(output_landmark_filename)
#
#     return video_file_list, landmark_file_list

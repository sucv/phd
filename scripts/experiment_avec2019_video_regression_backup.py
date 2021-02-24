from utils.initial_setting import initial_setting, initialize_emotion_spatial_temporal_model
from utils.trainer import EmotionalVideoRegressionTrainer
from utils.base_class import Avec2019Arranger
from utils.helper import load_pkl_file, save_pkl_file

import numpy as np
from operator import itemgetter
import json
import os
import sys


def main_avec2019(tc="all", vc="all", gpu_index=None, cpu_thread_number=None):

    # Set the random seed for reproducing the result.
    # Detect the available device (gpu available? How many of it?)
    # Set one of the gpu to use.
    # Set the thread number of cpu.
    device = initial_setting(seed=0, gpu_index=gpu_index, cpu_thread_number=cpu_thread_number)

    # Load the configs.
    with open("configs/config_avec2019_backup") as config_file:
        config = json.load(config_file)
        config['train_country'] = tc
        config['validate_country'] = vc

    # Where to save the metrics and trained models.
    result_directory = os.path.join('result', config['program_name'], config['model_name'])
    os.makedirs(result_directory, exist_ok=True)

    # Where to save the output-to-continuous label plots, and the checkpoint.
    model_directory = os.path.join('load', config['program_name'], config['model_name'])
    os.makedirs(model_directory, exist_ok=True)

    # Allocate subjects for N-fold partition. Each fold tends to have equal trials.
    folder_arranger = Avec2019Arranger(config['root_directory'], config)
    subject_id_of_all_folds, b = folder_arranger.assign_subject_to_fold()

    # Where to save the session-wise plot. These plots will be used for
    # debugging, analysis, and showcase.
    directory_to_save_checkpoint_and_plot = os.path.join(model_directory)

    # Load the checkpoint.
    checkpoint = {}
    checkpoint_filename = os.path.join(directory_to_save_checkpoint_and_plot, "checkpoint.pkl")

    # If checkpoint file exists, then read it.
    if os.path.isfile(checkpoint_filename):
        print("Loading checkpoint. Are you sure it is intended?")
        checkpoint = {**checkpoint, **load_pkl_file(checkpoint_filename)}
        print("Checkpoint loaded!")
        print("Fitting completed?", str(checkpoint['fit_finished']))
        print("Start epoch:", str(checkpoint['start_epoch']))

    # Here we generate three crucial arguments. "file_of_all_partitions" is a dictionary saving the filename to load for training,
    # validation, and test. "length_of_all_partitions" is a dictionary saving the length (frame count) for every sessions of the subjects.
    # "clip_sample_map_of_all_partitions" is a dictionary saving the session index from which a video clip comes from. The latter
    # is used to correctly place the output from clipped and shuffled video clips in the session-wise, subject-wise, and partition-wise
    # manners, which is necessary for plotting and metric calculation.
    file_of_all_partitions, length_of_all_partitions, clip_sample_map_of_all_partitions = folder_arranger.get_partition_list(
        subject_id_of_all_folds, clip_number_to_load=config['clip_number_to_load'],
        downsampling_interval=config['downsampling_interval'],
        compact_folder=config['compact_folder'], time_depth=config['time_depth'], step_size=config['step_size'],
        frame_number_to_load=config['frame_number_to_compact'], shuffle=config['data_arranger_shuffle'], debug=False)

    # Assign the arguments to different partitions.
    data_to_load_for_training = {
        'train': file_of_all_partitions['train'],
        'validate': file_of_all_partitions['validate']
    }

    length_to_track_for_training = {
        'train': length_of_all_partitions['train'],
        'validate': length_of_all_partitions['validate']
    }

    clip_sample_map_to_track_for_training = {
        'train': clip_sample_map_of_all_partitions['train'],
        'validate': clip_sample_map_of_all_partitions['validate']
    }

    # Here we initialize the model. It contains the spatial block and temporal block.
    FRAME_DIM = 224
    TIME_DEPTH = 16
    SHARED_LINEAR_DIM1 = 1024
    SHARED_LINEAR_DIM2 = 512
    EMBEDDING_DIM = SHARED_LINEAR_DIM2
    HIDDEN_DIM = 512
    OUTPUT_DIM = 2
    N_LAYERS = 1
    DROPOUT_RATE_1 = 0.5
    DROPOUT_RATE_2 = 0.5
    model = initialize_emotion_spatial_temporal_model(
        device, frame_dim=FRAME_DIM, time_depth=TIME_DEPTH,
        shared_linear_dim1=SHARED_LINEAR_DIM1,
        shared_linear_dim2=SHARED_LINEAR_DIM2,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, n_layers=N_LAYERS,
        dropout_rate_1=DROPOUT_RATE_1, dropout_rate_2=DROPOUT_RATE_2
    )

    # Next we initialize the trainer. It carries out tasks including training, validation, evaluation, and visualization.
    trainer = EmotionalVideoRegressionTrainer(model, time_depth=config['time_depth'], step_size=config['step_size'],
                                              metrics=config['metrics'], batch_size=config['batch_size'],
                                              emotional_dimension=config['emotion_dimension'],
                                              device=device, shuffle=config['data_loader_shuffle'], verbose=True,
                                              print_training_metric=True)

    # If the fitting is not finished, then continuing it from the epoch recorded in the checkpoint.
    # Epoch is the smallest unit to continue from the checkpoint. Within an epoch, there are multiple groups
    # to iterate, which is not recorded. Therefore, once stopped, we can continue only from the first group of
    # that epoch.
    if 'fit_finished' not in checkpoint.keys() or not checkpoint['fit_finished']:
        combined_record_dict, checkpoint = trainer.fit(
            data_to_load_for_training, length_to_track_for_training,
            clip_sample_map_to_track_for_training, epoch_number=config['epochs'],
            early_stopping=config['early_stopping'],
            clipwise_frame_number=config['frame_number_to_compact'],
            checkpoint=checkpoint,
            directory_to_save_checkpoint_and_plot=directory_to_save_checkpoint_and_plot)

    # If fitting is done but not the testing, then load the fitting result before continuing.
    else:
        combined_record_dict = {'train': checkpoint['combined_train_record_dict'],
                                'validate': checkpoint['combined_validate_record_dict']}




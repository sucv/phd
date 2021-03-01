from utils.base_class import NFoldMahnobArrangerNPY
from utils.dataset import MAHNOBDataset
from utils.trainer import MAHNOBRegressionTrainer
from utils.initial_setting import initial_setting
from utils.helper import load_pkl_file
from model.model import CFER, InceptResV1
from model.prototype import my_2d1d, my_2dlstm
from utils.initial_setting import initialize_emotion_spatial_temporal_model
from utils.loss import ccc_loss

from datetime import datetime
import json
import os
import numpy as np
from operator import itemgetter

import torch
import torch.nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class generic_experiment_mahnob_video_regression(object):
    def __init__(self, param):

        self.n_fold = param.n_fold
        self.folds_to_run = param.folds_to_run
        self.dataset_name = "mahnob"
        self.model = param.m
        self.stamp = param.s
        self.modality = param.modal
        self.job = param.j
        if param.j == 0:
            self.job = "reg_v"

        self.model_name = param.m + "_" + self.dataset_name + "_" + self.job
        self.config = self.load_config()
        self.learning_rate = param.lr
        self.patience = param.p
        self.time_delay = param.d
        self.model_load_path = param.model_load_path
        self.model_save_path = param.model_save_path

        if param.model_load_path == '':
            self.gpu = param.gpu
            self.cpu = param.cpu
        else:
            self.gpu = None
            self.cpu = None

        self.device = initial_setting(seed=0, gpu_index=self.gpu, cpu_thread_number=self.cpu)

        self.experiment()

    def get_train_emotion(self, option):
        if option == "a":
            emotion = "arousal"
        elif option == "v":
            emotion = "valence"
        elif option == "b":
            emotion = "both"
        else:
            raise ValueError("Unknown emotion dimension to train!")

        return emotion

    def load_config(self):
        # Load the config.
        with open("configs/config_mahnob") as config_file:
            config = json.load(config_file)
        return config

    def init_model(self, model_name):
        # # Here we initialize the model. It contains the spatial block and temporal block.
        # FRAME_DIM = 96
        # TIME_DEPTH = 300
        # SHARED_LINEAR_DIM1 = 1024
        # SHARED_LINEAR_DIM2 = 512
        # EMBEDDING_DIM = SHARED_LINEAR_DIM2
        # HIDDEN_DIM = 512
        # OUTPUT_DIM = 2
        # N_LAYERS = 1
        # DROPOUT_RATE_1 = 0.5
        # DROPOUT_RATE_2 = 0.5
        # model = initialize_emotion_spatial_temporal_model(
        #     self.device, frame_dim=FRAME_DIM, time_depth=TIME_DEPTH,
        #     shared_linear_dim1=SHARED_LINEAR_DIM1,
        #     shared_linear_dim2=SHARED_LINEAR_DIM2,
        #     embedding_dim=EMBEDDING_DIM,
        #     hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, n_layers=N_LAYERS,
        #     dropout_rate_1=DROPOUT_RATE_1, dropout_rate_2=DROPOUT_RATE_2
        # )

        if self.model == "2d1d":
            model = my_2d1d(backbone_model_name="my_res50_fer+_backup", feature_dim=512,
                            channels_1D=[128, 128, 128, 128, 128], output_dim=1, kernel_size=5, dropout=0.1, root_dir=self.model_load_path)
        elif self.model == "2dlstm":
            model = my_2dlstm(backbone_model_name="my_res50_fer+_backup", feature_dim=512,
                               output_dim=1, dropout=0.4, root_dir=self.model_load_path)
        else:
            raise ValueError("Unknown model!")
        return model

    def init_dataloader(self, subject_id_of_all_folds, fold_arranger):

        # Set the fold-to-partition configuration.
        # Each fold have approximately the same number of sessions.
        if self.n_fold == 3:
            partition_dictionary = {'train': 1, 'validate': 1, 'test': 1}
        elif self.n_fold == 5:
            partition_dictionary = {'train': 3, 'validate': 1, 'test': 1}
        elif self.n_fold == 9:
            partition_dictionary = {'train': 6, 'validate': 2, 'test': 1}
        elif self.n_fold == 10:
            partition_dictionary = {'train': 7, 'validate': 2, 'test': 1}
        else:
            raise ValueError("The fold number is not supported or realistic!")

        data_dict = fold_arranger.make_data_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)
        length_dict = fold_arranger.make_length_dict(subject_id_of_all_folds, partition_dictionary=partition_dictionary)

        dataloaders_dict = {}
        for partition in partition_dictionary.keys():
            dataset = MAHNOBDataset(self.config, data_dict[partition], modality=self.modality, time_delay=self.time_delay, mode=partition)
            dataloaders_dict[partition] = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.config['batch_size'], shuffle=True if partition == "train" else False)

        return dataloaders_dict, length_dict

    def experiment(self):

        checkpoint_directory = os.path.join("load", self.model_name + "_" + self.stamp, str(self.n_fold))
        if self.model_save_path:
            checkpoint_directory = os.path.join(self.model_save_path, self.model_name + "_" + self.stamp, str(self.n_fold))

        # Allocate subjects for N-fold partition. Each fold tends to have equal trials.
        fold_arranger = NFoldMahnobArrangerNPY(self.config, job=self.job, modality=self.modality)
        subject_id_of_all_folds, _ = fold_arranger.assign_subject_to_fold(self.n_fold)

        # Here goes the N-fold training.
        for fold in iter(self.folds_to_run):
            print("Running fold:", str(fold))
            print("How many folds?", str(self.n_fold))
            # Where to save the session-wise plot. These plots will be used for
            # debugging, analysis, and showcase.
            directory_to_save_checkpoint_and_plot = os.path.join(checkpoint_directory, "fold_" + str(fold))

            # Load the checkpoint.
            checkpoint = {}
            checkpoint_filename = os.path.join(directory_to_save_checkpoint_and_plot, "checkpoint.pkl")

            # If checkpoint file exists, then read it.
            if os.path.isfile(checkpoint_filename):
                print("Loading checkpoint. Are you sure it is intended?")
                checkpoint = {**checkpoint, **load_pkl_file(checkpoint_filename)}
                print("Checkpoint loaded!")
                print("Fold completed?", str(checkpoint['fold_finished']))
                print("Fitting completed?", str(checkpoint['fit_finished']))
                print("Start epoch:", str(checkpoint['start_epoch']))

            if 'fold_finished' in checkpoint and checkpoint['fold_finished']:
                continue

            # If it is not the first fold, then we shall roll the fold list, so that data from
            # subjects of next N-fold partition can be successfully processed.
            if fold > 0:
                fold_index = np.roll(np.arange(self.n_fold), fold)
                subject_id_of_all_folds = list(itemgetter(*fold_index)(subject_id_of_all_folds))

            # This line below forces the three partitions to use data from subject '5'. It is for debugging use.
            # Subject 5 has the least data, which takes the least time to finish, by which we can find out whether
            # the code works as desired.
            # subject_id_of_all_folds = [[18], [18], [18], [18], [18]]

            # Here we generate three crucial arguments. "file_of_all_partitions" is a dictionary saving the filename to load for training,
            # validation, and test. "length_of_all_partitions" is a dictionary saving the length (frame count) for every sessions of the subjects.
            # "clip_sample_map_of_all_partitions" is a dictionary saving the session index from which a video clip comes from. The latter
            # is used to correctly place the output from clipped and shuffled video clips in the session-wise, subject-wise, and partition-wise
            # manners, which is necessary for plotting and metric calculation.
            dataloaders_dict, lengths_dict = self.init_dataloader(subject_id_of_all_folds, fold_arranger)


            criterion = ccc_loss()
            model = self.init_model("my_res50_fer+_backup")


            milestone = [1000]
            trainer = MAHNOBRegressionTrainer(model, stamp=self.stamp, model_name=self.model_name, learning_rate=self.learning_rate, metrics=self.config['metrics'],
                                    train_emotion='valence', patience=self.patience, emotional_dimension=['Valence'],
                                    milestone=milestone, criterion=criterion, verbose=True, device=self.device)
            trainer.fit(dataloaders_dict, lengths_dict, num_epochs=200, early_stopping=50, min_num_epoch=0,
                        directory_to_save_checkpoint_and_plot=directory_to_save_checkpoint_and_plot, save_model=True, checkpoint=checkpoint)


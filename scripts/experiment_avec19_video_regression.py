from utils.base_class import AVEC19ArrangerNPY
from utils.dataset import AVEC19Dataset
from utils.trainer import AVEC19Trainer
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


class generic_experiment_avec19_video_regression(object):
    def __init__(self, param):

        self.train_country = param.tc
        self.validate_country = param.vc
        self.model = param.m
        self.gpu = param.gpu
        self.cpu = param.cpu
        self.stamp = param.s
        self.train_emotion = self.get_train_emotion(param.e)
        self.model_name = param.m + "_" + self.train_emotion
        self.device = initial_setting(seed=0, gpu_index=self.gpu, cpu_thread_number=self.cpu)
        self.config = self.load_config()
        self.learning_rate = param.lr
        self.patience = param.p
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
        with open("configs/config_avec2019") as config_file:
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
                            channels_1D=[128, 128, 128], output_dim=2, kernel_size=3, dropout=0.1)
        elif self.model == "2dlstm":
            model = my_2dlstm(backbone_model_name="my_res50_fer+_backup", feature_dim=512,
                               output_dim=2, dropout=0.4)
        else:
            raise ValueError("Unknown model!")
        return model

    def init_dataloader(self):
        arranger = AVEC19ArrangerNPY(self.config)
        train_dict, validate_dict = arranger.make_data_dict(train_country=self.train_country, validate_country=self.validate_country)

        train_dataset = AVEC19Dataset(self.config, train_dict, mode='train')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        validate_dataset = AVEC19Dataset(self.config, validate_dict, mode='validate')
        validate_loader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=self.config['batch_size'], shuffle=False)
        return train_loader, validate_loader

    def init_length_dict(self):
        arranger = AVEC19ArrangerNPY(self.config)
        train_length_dict, validate_length_dict = arranger.make_length_dict(train_country=self.train_country, validate_country=self.validate_country)
        return train_length_dict, validate_length_dict

    def experiment(self):
        directory_to_save_checkpoint_and_plot = os.path.join("load", self.model_name + "_" + self.stamp)

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

        criterion = ccc_loss()
        model = self.init_model("my_res50_fer+_backup")
        train_loader, validate_loader = self.init_dataloader()
        train_length_dict, validate_length_dict = self.init_length_dict()

        dataloaders_dict = {'train': train_loader, 'validate': validate_loader}
        lengths_dict = {'train': train_length_dict, 'validate': validate_length_dict}

        milestone = [1000]
        trainer = AVEC19Trainer(model, stamp=self.stamp, model_name=self.model_name, learning_rate=self.learning_rate, metrics=self.config['metrics'],
                                train_emotion=self.train_emotion, patience=self.patience, emotional_dimension=self.config['emotion_dimension'],
                                milestone=milestone, criterion=criterion, verbose=True, device=self.device)
        trainer.fit(dataloaders_dict, lengths_dict, num_epochs=200, early_stopping=50, min_num_epoch=0,
                    directory_to_save_checkpoint_and_plot=directory_to_save_checkpoint_and_plot, save_model=True, checkpoint=checkpoint)


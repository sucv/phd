from utils.base_class import CKplusArrangerStatic, OuluArrangerStatic, RafdArrangerStatic
from utils.dataset import EmotionalStaticImgClassificationDataset
from utils.dataset_github import Fer2013PlusDataset
from utils.trainer import EmotionalStaticImgClassificationTrainer
from utils.initial_setting import initial_setting
from model.model import CFER, InceptResV1, InceptResV2
from model.prototype import my_res50

from torch.utils.data import WeightedRandomSampler
import json
import os
import numpy as np
from operator import itemgetter

import torch
import torch.nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class generic_experiment_static_image_classification(object):
    def __init__(self, param):

        self.num_folds = param.n_fold
        if param.fold_to_run is None:
            self.fold_to_run = np.arange(0, self.num_folds)

        if not param.cv:
            self.fold_to_run = [0]

        self.dataset = param.d
        self.model = param.m
        self.cross_validation = param.cv
        self.gpu = param.gpu
        self.cpu = param.cpu
        self.model_name = param.m + "_" + param.d
        self.config = self.load_config()

    def load_config(self):
        # Load the config.
        with open("configs/config_" + self.dataset) as config_file:
            config = json.load(config_file)
        return config

    def init_model(self):
        if self.model == "cfer":
            model = CFER(num_classes=self.config['num_classes'])
        elif self.model == "inceptresv1":
            model = InceptResV1(num_classes=self.config['num_classes'])
        elif self.model == "my_inceptresv2":
            model = InceptResV2(num_classes=self.config['num_classes'])
        elif self.model == "my_res50":
            model = my_res50(num_classes=self.config['num_classes'], use_pretrained=self.config['use_pretrained'])
        else:
            raise ValueError('Model not supported!')
        return model

    def init_transform(self):

        transform = []
        transform.append(T.RandomHorizontalFlip())
        # if self.dataset == "fer2013" or self.dataset == "fer+":
        transform.append(T.Resize(self.config['resize']))
        transform.append(T.RandomCrop(self.config['center_crop']))
        transform.append(T.ColorJitter())
        transform.append(T.RandomAffine(degrees=10))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=self.config['mean'], std=self.config['std']))
        transform = T.Compose(transform)

        transform_val = []
        # if self.dataset == "fer2013" or self.dataset == "fer+":
        transform_val.append(T.Resize(self.config['resize']))
        transform_val.append(T.CenterCrop(self.config['center_crop']))
        transform_val.append(T.ToTensor())
        transform_val.append(T.Normalize(mean=self.config['mean'], std=self.config['std']))
        transform_val = T.Compose(transform_val)


        return transform, transform_val

    def init_arranger(self):

        if self.dataset == "ckplus":
            arranger = CKplusArrangerStatic(self.config, self.num_folds)
        elif self.dataset == "oulu":
            arranger = OuluArrangerStatic(self.config, self.num_folds)
        elif self.dataset == "rafd":
            arranger = RafdArrangerStatic(self.config, self.num_folds)
        else:
            arranger = None

        return arranger

    def experiment(self):
        device = initial_setting(seed=0, gpu_index=self.gpu, cpu_thread_number=self.cpu)

        if self.cross_validation:
            arranger = self.init_arranger()
            fold_list_origin = arranger.establish_fold()

        model = self.init_model()
        transform, transform_val = self.init_transform()

        for fold in iter(self.fold_to_run):

            if self.cross_validation:
                fold_index = np.roll(self.fold_to_run, fold)
                fold_list = list(itemgetter(*fold_index)(fold_list_origin))

                train_list = np.vstack(fold_list[1:])
                val_list = np.vstack(fold_list[0])

                train_dataset = EmotionalStaticImgClassificationDataset(train_list, transform)
                val_dataset = EmotionalStaticImgClassificationDataset(val_list, transform_val)

            # elif self.dataset == "fer+":
            #     train_dataset = Fer2013PlusDataset(self.config['remote_root_directory'], 'train', transform=transform, include_train=True)
            #     val_dataset = Fer2013PlusDataset(self.config['remote_root_directory'], 'val', transform=transform_val,
            #                                        include_train=False)
            else:
                train_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'train'),
                                            transform=transform)

                val_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'validate'),
                                          transform=transform_val)
                # test_dataset = ImageFolder(os.path.join(self.config['remote_root_directory'], 'test'),
                #                           transform=transform_val)

            # class_sample_count = np.array([len(np.where(train_dataset.targets == t)[0]) for t in np.unique(train_dataset.targets)])
            class_sample_count = np.unique(train_dataset.targets, return_counts=True)[1]
            weight = 1. / class_sample_count
            # samples_weight = np.array([weight[t] for t in train_dataset.targets])
            samples_weight = weight[train_dataset.targets]
            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

            train_loader = data.DataLoader(dataset=train_dataset, batch_size=self.config['batch_size'], sampler=sampler)
            val_loader = data.DataLoader(dataset=val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            # test_loader = data.DataLoader(dataset=test_dataset, batch_size=self.config['batch_size'], shuffle=False)
            dataloaders_dict = {'train': train_loader, 'val': val_loader}


            milestone = [0]
            trainer = EmotionalStaticImgClassificationTrainer(model, model_name=self.model_name,
                                                              num_classes=self.config['num_classes'], device=device,
                                                              fold=fold, milestone=milestone, patience=5, samples_weight=samples_weight)
            trainer.fit(dataloaders_dict, num_epochs=2000, early_stopping=100, topk_accuracy=1, min_num_epoch=0,
                        save_model=True)
            # trainer.validate(test_loader, topk_accuracy=1)


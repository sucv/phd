from utils.base_class import CKplusArrangerStatic, OuluArrangerStatic
from utils.dataset import EmotionalStaticImgClassificationDataset
from utils.trainer import EmotionalStaticImgClassificationTrainer
from utils.initial_setting import initial_setting
from model.model import CFER, InceptResV1
from model.inception_resnet_v1 import InceptionResnetV1

import json
import numpy as np
from operator import itemgetter

import torch
import torch.nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import vgg11_bn


def main_k_fold_static_image_classification(n_fold=10, folds_to_run=None, gpu_index=None, cpu_thread_number=None):

    device = initial_setting(seed=0, gpu_index=gpu_index, cpu_thread_number=cpu_thread_number)
    torch.cuda.set_device(1)

    with open("configs/config_ckplus") as config_file:
        config = json.load(config_file)

    a = CKplusArrangerStatic(config)
    fold_list_origin = a.establish_fold()

    if folds_to_run is None:
        folds_to_run = np.arange(0, n_fold)

    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(96))
    transform.append(T.ColorJitter())
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    for k in range(1, n_fold):
        fold_index = np.roll(folds_to_run, k)
        fold_list = list(itemgetter(*fold_index)(fold_list_origin))

        train_list = np.vstack(fold_list[1:])
        val_list = np.vstack(fold_list[0])

        train_dataset = EmotionalStaticImgClassificationDataset(train_list, transform, "train")
        val_dataset = EmotionalStaticImgClassificationDataset(val_list, transform, "train")
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=48, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=48, shuffle=False)
        dataloaders_dict = {'train': train_loader,
                            'val': val_loader}

        model = CFER()

        trainer = EmotionalStaticImgClassificationTrainer(model, device=device)

        trainer.fit(dataloaders_dict, num_epochs=2000, early_stopping=500, topk_accuracy=1, min_num_epoch=10, save_model=False)


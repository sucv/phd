from utils.dataset import EmotionalSequentialDataset
from utils.base_class import variable_length_collate_fn
from utils.helper import length_to_mask
from model.model import custom_mse

import os
import numpy as np
import torch
import torch.utils.data
from utils.dataset import *


def n_fold_train(
        train_session_id,
        validate_session_id,
        data_arranger,
        optimizer,
        criterion,
        config,
        transform,
        model1,
        model2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generated_data_label_list_train = \
        data_arranger.generate_data_filename_and_label_array(
            train_session_id, random_interval=False, shuffle=True, training=True)
    generated_data_label_list_validate = \
        data_arranger.generate_data_filename_and_label_array(
            validate_session_id, random_interval=False, shuffle=True, training=True)

    for generated_data_label in generated_data_label_list_train:
        data = generated_data_label['data']
        continuous_label = generated_data_label['continuous_label']

        dataset = EmotionalFramewiseDataset(data, continuous_label, config, transform)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=4, shuffle=True)

        running_loss = 0
        num_batches = 0

        early_stopping_flag = False
        early_stopping_counter = 0
        best_loss_train = 1000

        for epoch in range(config['epoch']):

            if early_stopping_flag:
                break

            for data, target in train_loader:
                optimizer.zero_grad()

                # ===
                data = data.view(-1, 3, 224, 224).to(device)
                target = target.type(torch.float).to(device)
                prob = model1(data)
                data = prob.view(-1, 64, 1000)
                prob = model2(data, "regression")
                # ===

                loss = criterion(prob, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            epoch_loss = running_loss / num_batches
            print('epoch=%d, \t loss=%f' % (int(epoch + 1), epoch_loss))

            if epoch_loss < best_loss_train:
                best_loss = epoch_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == config['early_stopping_bound']:
                    early_stopping_flag = True

    with torch.no_grad():
        model1.eval()
        model2.eval()

        best_loss_validate = 1000
        for generated_data_label in generated_data_label_list_validate:
            data = generated_data_label['data']
            continuous_label = generated_data_label['continuous_label']

            dataset = EmotionalFramewiseDataset(data, continuous_label, config, transform)
            validate_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=4, shuffle=True)


def train(net, optimizer, subjects, config, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_working_dir = os.getcwd()
    path_to_save_model = os.path.join(current_working_dir, config['load_folder'], "weights_best_model.hdf5")

    subjects_for_train = subjects[:-4]
    train_dataset = EmotionalSequentialDataset(subjects_for_train, config, transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'], shuffle=True,
                                               collate_fn=variable_length_collate_fn)

    subjects_for_validate = subjects[-4:-1]
    validate_dataset = EmotionalSequentialDataset(subjects_for_validate, config, transform)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                                  batch_size=config['batch_size'], shuffle=True,
                                                  collate_fn=variable_length_collate_fn)

    best_loss = 10000

    for epoch in range(300):

        running_loss = 0
        num_batches = 0
        train_probs = []
        net.train()
        for data in train_loader:
            optimizer.zero_grad()

            minibatch_train_feature, minibatch_train_feature_length = data[0].to(device), data[1]
            minibatch_train_label, minibatch_train_label_length = data[2].to(device), data[3]

            inference = net(minibatch_train_feature, minibatch_train_feature_length, 'regression')
            mask = length_to_mask(minibatch_train_feature_length, device)
            loss = custom_mse(minibatch_train_label, inference, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            num_batches += 1

        total_loss = running_loss / num_batches

        # validation
        num_batches = 0
        val_losses = 0

        with torch.no_grad():
            net.eval()
            for data in validate_loader:
                minibatch_validate_feature, minibatch_valdate_validate_length = data[0].to(device), data[1]
                minibatch_validate_label, minibatch_validate_label_length = data[2].to(device), data[3]

                val_inference = net(minibatch_validate_feature, minibatch_valdate_validate_length, 'regression')
                mask = length_to_mask(minibatch_valdate_validate_length, device)
                val_loss = custom_mse(minibatch_validate_label, val_inference, mask)

                num_batches += 1
                val_losses += val_loss
            val_losses = val_losses / num_batches

            if val_losses < best_loss:
                best_loss = val_losses
                early_stopping_counter = 0
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path_to_save_model)
            else:
                early_stopping_counter += 1

            print('epoch=%d, \t loss=%.5f, \t val_loss=%.9f' % (int(epoch + 1), total_loss, val_losses))

            if early_stopping_counter == 10:
                break


def test(net, optimizer, subjects, config, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_working_dir = os.getcwd()
    path_to_load_checkpoint = os.path.join(current_working_dir, config['load_folder'], "weights_best_model.hdf5")
    checkpoint = torch.load(path_to_load_checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    subjects_for_test = [subjects[-1]]
    test_dataset = EmotionalSequentialDataset(subjects_for_test, config, transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config['batch_size'], shuffle=True,
                                              collate_fn=variable_length_collate_fn)

    subject_result_list = [subjects_for_test]
    with torch.no_grad():
        net.eval()
        batch_result_list = []
        for batch_index, data in enumerate(test_loader):
            single_batch_result = [batch_index]
            minibatch_test_feature, minibatch_test_feature_length = data[0].to(device), data[1]
            minibatch_test_label, minibatch_test_label_length = data[2].to(device), data[3]

            test_inference = net(minibatch_test_feature, minibatch_test_feature_length, 'regression')
            mask = length_to_mask(minibatch_test_feature_length, device)

            single_batch_result.append(test_inference.tolist())
            single_batch_result.append(minibatch_test_label.tolist())
            single_batch_result.append(mask.tolist())
            batch_result_list.append(single_batch_result)

    subject_result_list.append(batch_result_list)
    return subject_result_list

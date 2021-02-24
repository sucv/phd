import time
import copy
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from sklearn.metrics import accuracy_score


from utils.helper import save_pkl_file, print_progress
from utils.dataset import EmotionalDataset
from utils.parameter_control import GenericParamControl, GenericReduceLROnPlateau
from utils.base_class import ContinuousOutputHandler, ContinuousMetricsCalculator, PlotHandler, \
    ContinuousOutputHandlerNPY


class AVEC19Trainer(object):
    def __init__(
            self,
            model,
            stamp,
            model_name='2d1d',
            model_path=None,
            train_emotion='both',
            max_epoch=100,
            optimizer=None,
            criterion=None,
            scheduler=None,
            milestone=[0],
            patience=10,
            learning_rate=0.00001,
            device='cpu',
            emotional_dimension=None,
            metrics=None,
            verbose=False,
            print_training_metric=False
    ):

        # The device to use.
        self.device = device

        self.stamp = stamp
        # Whether to show the information strings.
        self.verbose = verbose

        # Whether print the metrics for training.
        self.print_training_metric = print_training_metric

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension
        self.train_emotion = train_emotion

        self.metrics = metrics

        # The learning rate, and the patience of schedule.
        self.learning_rate = learning_rate
        self.patience = patience

        # The networks.
        self.model_path = model_path
        if model_path is None:
            self.model_path = 'load/' + str(model_name) + "_" + self.stamp + '.pth'
        self.model = model.to(device)

        # Get the parameters to update, to check whether the false parameters
        # are included.
        parameters_to_update = self.get_parameters()

        # Initialize the optimizer.
        if optimizer:
            self.optimizer = optimizer
        else:
            # self.optimizer = optim.Adam(parameters_to_update, lr=learning_rate, weight_decay=0.01)
            self.optimizer = optim.SGD(parameters_to_update, lr=learning_rate, weight_decay=0.001, momentum=0.9)

        # Initialize the loss function.
        if criterion:
            # Use custom ccc loss
            self.criterion = criterion
        else:
            self.criterion = torch.nn.MSELoss()

        # Initialize the scheduler.
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        # parameter_control
        self.milestone = milestone
        self.parameter_control = GenericParamControl(model)

        self.lr_control = GenericReduceLROnPlateau(patience=patience, min_epoch=0, learning_rate=learning_rate,
                                                   milestone=self.milestone, num_release=8)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        # if self.verbose:
        #     print("Layers with params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        #         if self.verbose:
        #             print("\t", name)
        # if self.verbose:
        #     print('\t', len(params_to_update), 'layers')
        return params_to_update

    def train(self, data_loader, length_to_track, directory_to_save_checkpoint_and_plot, epoch):
        self.model.train()
        return self.loop(data_loader, length_to_track, directory_to_save_checkpoint_and_plot, epoch, train_mode=True)

    def validate(self, data_loader, length_to_track, directory_to_save_checkpoint_and_plot, epoch):
        self.model.eval()
        return self.loop(data_loader, length_to_track, directory_to_save_checkpoint_and_plot, epoch, train_mode=False)

    def fit(
            self,
            data_to_load,
            length_to_track,
            num_epochs=100,
            early_stopping=20,
            min_num_epoch=10,
            clipwise_frame_number=160,
            checkpoint=None,
            directory_to_save_checkpoint_and_plot=None,
            save_model=False
    ):
        r"""
        The function to carry out training and validation.
        :param directory_to_save_checkpoint_and_plot:
        :param clip_sample_map_to_track:
        :param data_to_load: (dict), the data in training and validation partitions.
        :param length_to_track: (dict), the corresponding length of the subjects' sessions.
        :param fold: the current fold index.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param epoch_number: (int), how many epochs to run.
        :param early_stopping: (int), how many epochs to tolerate before stopping early.
        :param min_epoch_number: the minimum epochs to run before calculating the early stopping.
        :param checkpoint: (dict), to save the information once an epoch is done
        :return: (dict), the metric dictionary recording the output and its associate continuous labels
            as a long array for each subject.
        """

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, validate_losses = [], []
        early_stopping_counter = early_stopping
        start_epoch = 0

        best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'ccc': -1e10
        }

        combined_train_record_dict = {}
        combined_validate_record_dict = {}
        combined_record_dict = {'train': [], 'validate': []}

        df = pd.DataFrame(
            columns=['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr', 'plateau_count',
                     'tr_loss', 'tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf', 'tr_ccc_a',
                     'tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
                     'val_loss', 'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a',
                     'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])

        csv_filename = self.model_path[:-4] + ".csv"
        df.to_csv(csv_filename, index=False)

        if len(checkpoint.keys()) > 1:
            time_fit_start = checkpoint['time_fit_start']
            csv_filename = checkpoint['csv_filename']
            start_epoch = checkpoint['start_epoch']
            early_stopping_counter = checkpoint['early_stopping_counter']
            best_epoch_info = checkpoint['best_epoch_info']
            combined_train_record_dict = checkpoint['combined_train_record_dict']
            combined_validate_record_dict = checkpoint['combined_validate_record_dict']
            train_losses = checkpoint['train_losses']
            validate_losses = checkpoint['validate_losses']
            self.model.load_state_dict(checkpoint['current_model_weights'])
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.parameter_control = checkpoint['param_control']
            self.lr_control = checkpoint['lr_control']
            self.model.load_state_dict(best_epoch_info['model_weights'])

        # Loop the epochs
        for epoch in np.arange(start_epoch, num_epochs):
            time_epoch_start = time.time()

            if epoch in self.milestone or self.lr_control.to_release:
                self.parameter_control.release_parameters_to_update()
                self.lr_control.released = True
                self.lr_control.update_lr()
                self.lr_control.to_release = False
                self.milestone = self.lr_control.update_milestone(epoch, add_milestone=20)
                params_to_update = self.get_parameters()
                self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                # self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                #                            momentum=0.9)

            num_layers_to_update = len(self.optimizer.param_groups[0]['params'])
            print("There are {} layers to update.".format(num_layers_to_update))
            # Get the losses and the record dictionaries for training and validation.
            train_loss, train_record_dict = self.train(data_to_load['train'], length_to_track['train'],
                                                       directory_to_save_checkpoint_and_plot, epoch)

            # Combine the record to a long array for each subject.
            combined_train_record_dict = self.combine_record_dict(
                combined_train_record_dict, train_record_dict)

            validate_loss, validate_record_dict = self.validate(data_to_load['validate'], length_to_track['validate'],
                                                                directory_to_save_checkpoint_and_plot, epoch)

            combined_validate_record_dict = self.combine_record_dict(
                combined_validate_record_dict, validate_record_dict)

            # Calculate the mean metrics for a whole partition for information showcase.
            mean_train_record = train_record_dict['overall']
            mean_validate_record = validate_record_dict['overall']

            train_losses.append(train_loss)
            validate_losses.append(validate_loss)

            improvement = False

            if self.train_emotion == "both":
                validate_ccc = np.mean([mean_validate_record[emotion]['ccc'] for emotion in self.emotional_dimension])
            elif self.train_emotion == "arousal":
                validate_ccc = np.mean(mean_validate_record['Arousal']['ccc'])
            elif self.train_emotion == "valence":
                validate_ccc = np.mean(mean_validate_record['Valence']['ccc'])
            else:
                raise  ValueError("Unknown emotion dimension!")


            # If a lower validate loss appears.
            if validate_ccc > best_epoch_info['ccc']:
                if save_model:
                    torch.save(self.model.state_dict(), self.model_path)

                improvement = True
                best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                    'scalar_metrics': {
                        'train_loss': train_loss,
                        'validate_loss': validate_loss,
                    },
                    'array_metrics': {
                        'train_metric_record': mean_train_record,
                        'validate_metric_record': mean_validate_record
                    }
                }

            # Early stopping controller.
            if early_stopping and epoch > min_num_epoch:
                if improvement:
                    early_stopping_counter = early_stopping
                else:
                    early_stopping_counter -= 1

                if early_stopping_counter <= 0:
                    if self.verbose:
                        print("\nEarly Stop!!")
                    break

            if validate_loss < 0:
                print('validate loss negative')

            if self.verbose:
                print(
                    "\n Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        self.lr_control.learning_rate,
                        self.lr_control.release_count,
                        int(best_epoch_info['epoch']) + 1,
                        improvement,
                        early_stopping_counter))

                print(mean_train_record)
                print(mean_validate_record)
                print("------")

            # df = pd.DataFrame(
            #     columns=['time', 'epoch', 'best_epoch', 'layer_to_update', 'lr', 'plateau_count',
            #              'tr_loss', 'tr_rmse_a', 'tr_pcc_a_v', 'tr_pcc_a_conf','tr_ccc_a', 'tr_rmse_v', 'tr_pcc_v_v', 'tr_pcc_v_conf', 'tr_ccc_v',
            #              'val_loss', 'val_rmse_a', 'val_pcc_a_v', 'val_pcc_a_conf', 'val_ccc_a', 'val_rmse_v', 'val_pcc_v_v', 'val_pcc_v_conf', 'val_ccc_v'])

            csv_records = [
                time.time(), epoch, int(best_epoch_info['epoch']), num_layers_to_update, self.optimizer.param_groups[0]['lr'], self.lr_control.plateau_count,
                train_loss, mean_train_record['Arousal']['rmse'][0], mean_train_record['Arousal']['pcc'][0][0], mean_train_record['Arousal']['pcc'][0][1], mean_train_record['Arousal']['ccc'][0],
                mean_train_record['Valence']['rmse'][0], mean_train_record['Valence']['pcc'][0][0],
                mean_train_record['Valence']['pcc'][0][1], mean_train_record['Valence']['ccc'][0],
                validate_loss, mean_validate_record['Arousal']['rmse'][0], mean_validate_record['Arousal']['pcc'][0][0], mean_validate_record['Arousal']['pcc'][0][1], mean_validate_record['Arousal']['ccc'][0],
                mean_validate_record['Valence']['rmse'][0], mean_validate_record['Valence']['pcc'][0][0],
                mean_validate_record['Valence']['pcc'][0][1], mean_validate_record['Valence']['ccc'][0],
            ]

            row_df = pd.DataFrame(data=csv_records)
            row_df.T.to_csv(csv_filename, mode='a', index=False, header=False)

            # if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.scheduler.step(validate_loss)
            # else:
            #     self.scheduler.step()

            self.lr_control.step(epoch, 1 - validate_ccc)

            if self.lr_control.updated:
                params_to_update = self.get_parameters()
                self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                # self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                #                            momentum=0.9)
                self.lr_control.updated = False

            if self.lr_control.halt:
                break

            checkpoint['time_fit_start'] = time_fit_start
            checkpoint['start_epoch'] = epoch + 1
            checkpoint['early_stopping_counter'] = early_stopping_counter
            checkpoint['best_epoch_info'] = best_epoch_info
            checkpoint['combined_train_record_dict'] = combined_train_record_dict
            checkpoint['combined_validate_record_dict'] = combined_validate_record_dict
            checkpoint['train_losses'] = train_losses
            checkpoint['validate_losses'] = validate_losses
            checkpoint['csv_filename'] = csv_filename
            checkpoint['optimizer'] = self.optimizer
            checkpoint['scheduler'] = self.scheduler
            checkpoint['param_control'] = self.parameter_control
            checkpoint['current_model_weights'] = copy.deepcopy(self.model.state_dict())
            checkpoint['lr_control'] = self.lr_control
            checkpoint['fit_finished'] = False
            checkpoint['fold_finished'] = False
            if directory_to_save_checkpoint_and_plot:
                print("Saving checkpoint.")
                save_pkl_file(directory_to_save_checkpoint_and_plot, "checkpoint.pkl", checkpoint)
                print("Checkpoint saved.")

        checkpoint['fit_finished'] = True

        combined_record_dict['train'] = combined_train_record_dict
        combined_record_dict['validate'] = combined_validate_record_dict

        self.model.load_state_dict(best_epoch_info['model_weights'])

        if self.print_training_metric:
            print("------")
            time_elapsed = time.time() - time_fit_start
            print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

            print("Epoch with lowest val loss:", best_epoch_info['epoch'])
            for m in best_epoch_info['scalar_metrics']:
                print('{}: {:.5f}'.format(m, best_epoch_info['scalar_metrics'][m]))

            print("Train metric:", best_epoch_info['array_metrics']['train_metric_record'])
            print("Validate metric:", best_epoch_info['array_metrics']['validate_metric_record'])
            print("------")

        if save_model:
            torch.save(self.model.state_dict(), self.model_path)


        # return combined_record_dict, checkpoint

    def loop(self, data_loader, length_to_track, directory_to_save_checkpoint_and_plot, epoch, train_mode=True):
        running_loss = 0.0

        output_handler = ContinuousOutputHandlerNPY(length_to_track, self.emotional_dimension)
        continuous_label_handler = ContinuousOutputHandlerNPY(length_to_track, self.emotional_dimension)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculator(self.metrics, self.emotional_dimension,
                                                     output_handler, continuous_label_handler)
        total_batch_counter = 0
        for batch_index, (X, Y, indices, sessions) in tqdm(enumerate(data_loader), total=len(data_loader)):
            total_batch_counter += len(sessions)

            inputs = X.to(self.device)

            if train_mode:
                labels = torch.squeeze(Y.float().to(self.device))
            else:
                labels = Y.float().to(self.device)

            # Determine the weight for loss function
            if train_mode:
                loss_weights = torch.ones([labels.shape[0], labels.shape[1], 2]).to(self.device)
                self.optimizer.zero_grad()

                if self.train_emotion == "both":
                    loss_weights[:, :, 0] *= 0.5
                    loss_weights[:, :, 1] *= 0.5
                elif self.train_emotion == "arousal":
                    loss_weights[:, :, 0] *= 0.8
                    loss_weights[:, :, 1] *= 0.2
                elif self.train_emotion == "valence":
                    loss_weights[:, :, 0] *= 0.2
                    loss_weights[:, :, 1] *= 0.8
                else:
                    raise ValueError("Unknown emotion dimention to train!")

            outputs = self.model(inputs)

            output_handler.place_clip_output_to_subjectwise_dict(outputs.detach().cpu().numpy(), indices, sessions)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(labels.detach().cpu().numpy(), indices,
                                                                           sessions)
            loss = self.criterion(outputs, labels) * outputs.size(0)

            running_loss += loss.mean().item()

            if train_mode:
                loss.backward(loss_weights, retain_graph=True)
                self.optimizer.step()

            print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / total_batch_counter

        # Restore the output and continuous labels to its original shape, which is session-wise.
        # By which the metrics can be calculated for each session.
        # The metrics is the average across the session and subjects of a partition.
        output_handler.get_sessionwise_dict()
        continuous_label_handler.get_sessionwise_dict()

        # Restore the output and continuous labels to partition-wise, i.e., two very long
        # arrays.  It is used for calculating the metrics.
        output_handler.get_partitionwise_dict()
        continuous_label_handler.get_partitionwise_dict()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        # This object plot the figures and save them.
        plot_handler = PlotHandler(self.metrics, self.emotional_dimension, epoch_result_dict,
                                   output_handler.sessionwise_dict, continuous_label_handler.sessionwise_dict,
                                   epoch=epoch, train_mode=train_mode,
                                   directory_to_save_plot=directory_to_save_checkpoint_and_plot)
        plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict

    def combine_record_dict(self, main_record_dict, epoch_record_dict):
        r"""
        Append the metric recording dictionary of an epoch to a main record dictionary.
            Each single term from epoch_record_dict will be appended to the corresponding
            list in min_record_dict.
        Therefore, the minimum terms in main_record_dict are lists, whose element number
            are the epoch number.
        """

        # If the main record dictionary is blank, then initialize it by directly copying from epoch_record_dict.
        # Since the minimum term in epoch_record_dict is list, it is available to append further.
        if not bool(main_record_dict):
            main_record_dict = epoch_record_dict
            return main_record_dict

        # Iterate the dict and append each terms from epoch_record_dict to
        # main_record_dict.
        for (subject_id, main_subject_record), (_, epoch_subject_record) \
                in zip(main_record_dict.items(), epoch_record_dict.items()):

            # Go through emotions, e.g., Arousal and Valence.
            for emotion in self.emotional_dimension:
                # Go through metrics, e.g., rmse, pcc, and ccc.
                for metric in self.metrics:
                    # Go through the sub-dictionary belonging to each subject.
                    if subject_id != "overall":
                        session_dict = epoch_record_dict[subject_id][emotion][metric]
                        for session_id in session_dict.keys():
                            main_record_dict[subject_id][emotion][metric][session_id].append(
                                epoch_record_dict[subject_id][emotion][metric][session_id][0]
                            )

                    # In addition to subject-wise records, there are one extra sub-dictionary
                    # used to store the overall metrics, which is actually the partition-wise metrics.
                    # In this sub-dictionary, the results are obtained by first concatenating all the output
                    # and continuous labels to two long arraies, and then calculate the metrics.
                    else:
                        main_record_dict[subject_id][emotion][metric].append(
                            epoch_record_dict[subject_id][emotion][metric][0]
                        )

        return main_record_dict


class EmotionalStaticImgClassificationTrainer(object):
    def __init__(
            self,
            model,
            milestone=[0, 10, 20, 30, 40, 50],
            fold=0,
            model_name='CFER',
            max_epoch=2000,
            optimizer=None,
            criterion=None,
            scheduler=None,
            learning_rate=0.0001,
            device='cpu',
            num_classes=6,
            patience=20,
            samples_weight=0,
            verbose=True,
            print_training_metric=False,
            warmup_learning_rate=False,
    ):
        self.fold = fold
        self.device = device
        self.verbose = verbose
        self.model_name = model_name
        self.print_training_metric = print_training_metric
        self.num_classes = num_classes
        self.max_epoch = max_epoch
        self.model_path = 'load/' + str(model_name) + '.pth'
        self.learning_rate = learning_rate
        self.warmup_learning_rate = warmup_learning_rate
        self.model = model.to(device)
        params_to_update = self.get_parameters()

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(params_to_update, lr=learning_rate, weight_decay=0.001, momentum=0.9)

        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience)

        # parameter_control
        self.milestone = milestone
        self.parameter_control = GenericParamControl(model)

        self.lr_control = GenericReduceLROnPlateau(patience=patience, min_epoch=0, learning_rate=learning_rate,
                                                   milestone=self.milestone, num_release=8)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        # if self.verbose:
        #     print("Layers with params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        #         if self.verbose:
        #             print("\t", name)
        # if self.verbose:
        #     print('\t', len(params_to_update), 'layers')
        return params_to_update

    def compute_accuracy(self, outputs, targets, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))
        correct_k = correct[:k].view(-1).float()
        return correct_k

    def get_preds(self, outputs, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        return preds[0]

    def train(self, data_loader, topk_accuracy):
        self.model.train()
        return self.loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)

    def validate(self, data_loader, topk_accuracy):
        self.model.eval()
        return self.loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)

    def fit(self, dataloaders_dict, num_epochs=10, early_stopping=5, topk_accuracy=1, min_num_epoch=0,
            save_model=False):
        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
        early_stopping_counter = early_stopping

        best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'acc': 0,
        }

        for epoch in range(num_epochs):
            time_epoch_start = time.time()

            if epoch in self.milestone or self.lr_control.to_release:
                self.parameter_control.release_parameters_to_update()
                self.lr_control.released = True
                self.lr_control.update_lr()
                self.lr_control.to_release = False
                self.milestone = self.lr_control.update_milestone(epoch, add_milestone=50)
                params_to_update = self.get_parameters()
                # self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                                           momentum=0.9)
            print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            train_loss, train_acc = self.train(dataloaders_dict['train'], topk_accuracy)
            val_loss, val_acc = self.validate(dataloaders_dict['val'], topk_accuracy)

            train_losses.append(train_loss)
            test_losses.append(val_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(val_acc)

            mean_loss = np.mean(train_losses)

            improvement = False
            if val_acc > best_epoch_info['acc']:
                improvement = True
                best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': val_loss,
                    'acc': val_acc,
                    'epoch': epoch,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }
                }

            if early_stopping and epoch > min_num_epoch:
                if improvement:
                    early_stopping_counter = early_stopping
                else:
                    early_stopping_counter -= 1

                if early_stopping_counter <= 0:
                    if self.verbose:
                        print("\nEarly Stop!\n")
                    break

            if val_loss < 0:
                print('\nVal loss negative!\n')
                break

            if self.verbose:
                print(
                    "Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f}, | Val loss={:.3f}, acc={:.3f}, | LR={:.1e} | LR={:.1e} | best={} | best_acc={} | plateau_count={:2} | improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        self.optimizer.param_groups[0]['lr'],
                        self.lr_control.learning_rate,
                        int(best_epoch_info['epoch']) + 1,
                        best_epoch_info['acc'],
                        self.lr_control.plateau_count,
                        improvement,
                        early_stopping_counter)
                )

            # if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #     self.scheduler.step(val_loss)
            #
            # else:
            #     self.scheduler.step()

            self.model.load_state_dict(best_epoch_info['model_weights'])

            if self.print_training_metric:
                print()
                time_elapsed = time.time() - time_fit_start
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

                print('Epoch with lowest val loss:', best_epoch_info['epoch'])
                for m in best_epoch_info['metrics']:
                    print('{}: {:.5f}'.format(m, best_epoch_info['metrics'][m]))
                print()

            if save_model:
                torch.save(self.model.state_dict(), self.model_path)

            self.lr_control.step(epoch, val_loss)

            if self.lr_control.updated:
                params_to_update = self.get_parameters()
                # self.optimizer = optim.Adam(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001, betas=(0.9, 0.999))
                self.optimizer = optim.SGD(params_to_update, lr=self.lr_control.learning_rate, weight_decay=0.001,
                                           momentum=0.9)
                self.lr_control.updated = False

            if self.lr_control.halt:
                break

    def loop(self, data_loader, train_mode=True, topk_accuracy=1):

        running_loss = 0.0
        running_corrects = 0
        total_data_count = 0
        y_true = []
        y_pred = []

        # self.model.load_state_dict(state_dict=torch.load(self.model_path))

        for batch_index, (X, Y) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = X.to(self.device)

            labels = torch.squeeze(Y.long().to(self.device))

            if train_mode:
                self.optimizer.zero_grad()



            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels) * outputs.size(0)

            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(self.get_preds(outputs, topk_accuracy).cpu().numpy())

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

            # print_progress(batch_index, len(data_loader))

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)

        return epoch_loss, np.round(epoch_acc.item(), 3)


class EmotionalVideoRegressionTrainer:
    r"""
    A key class for training and testing of the continuous emotion regression.
    """

    def __init__(
            self,
            model,
            time_depth=64,
            step_size=64,
            batch_size=4,
            shuffle=False,
            optimizer=None,
            criterion=None,
            scheduler=None,
            patience=3,
            learning_rate=0.00001,
            device='cpu',
            emotional_dimension=None,
            metrics=None,
            verbose=False,
            print_training_metric=False
    ):

        # The device to use.
        self.device = device

        # The length of a sample.
        self.time_depth = time_depth

        self.step_size = step_size

        # The number of video to load at a time.
        self.batch_size = batch_size

        # Whether to show the information strings.
        self.verbose = verbose

        # Whether print the metrics for training.
        self.print_training_metric = print_training_metric

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension

        self.metrics = metrics

        # The learning rate, and the patience of schedule.
        self.learning_rate = learning_rate
        self.patience = patience

        # Whether to shuffle the sampling index of the data loader.
        self.shuffle = shuffle

        # The networks.
        self.model = model.to(device)

        # Get the parameters to update, to check whether the false parameters
        # are included.
        parameters_to_update = self.get_parameters()

        # Initialize the optimizer.
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(parameters_to_update, lr=learning_rate, weight_decay=0.05)

        # Initialize the loss function.
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = torch.nn.MSELoss()

        # Initialize the scheduler.
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

            torch.set_num_threads(1)

    def get_parameters(self):
        r"""
        Get the parameters to update.
        :return:
        """
        if self.verbose:
            print("Layers with params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                if self.verbose:
                    print("\t", name)
        if self.verbose:
            print('\t', len(params_to_update), 'layers')
        return params_to_update

    def train(self, data_to_load, length_to_track, clip_sample_map_to_track, directory_to_save_plot,
              clipwise_frame_number, epoch=None, train_mode=True):
        r"""
        The training function.
        :param clip_sample_map_to_track: (dict), a crucial argument, which records that for
            each sample having time_depth frames, which session it comes from. It will help to
            restore the output from clipped and shuffled data to the correct position for
            session-wise, subject-wise, and partition-wise.
        :param epoch: (int), the epoch index.
        :param directory_to_save_plot: (str), the directory to save the session-wise plot.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param data_to_load: (dict), the data in training and validation partitions.
        :param length_to_track: (dict), the corresponding length of the subjects' sessions.
        :param train_mode: (boolean), the flag for toggle the train /eval mode of the model.
        :return: (tensor), (dict), the train loss and the metric dictionary.
        """
        self.model.train()
        return self.epoch_loop(data_to_load, length_to_track, clip_sample_map_to_track, clipwise_frame_number,
                               epoch, directory_to_save_plot, train_mode=train_mode)

    def validate(self, data_to_load, length_to_track, clip_sample_map_to_track,
                 directory_to_save_plot, clipwise_frame_number=160, epoch=None, train_mode=False):
        r"""
        The validation or testing function.
        :param clip_sample_map_to_track: (dict), a crucial argument, which records that for
            each sample having time_depth frames, which session it comes from. It will help to
            restore the output from clipped and shuffled data to the correct position for
            session-wise, subject-wise, and partition-wise.
        :param epoch: (int), the epoch index.
        :param directory_to_save_plot: (str), the directory to save the session-wise plot.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param data_to_load: (dict), the data in validation or test partitions.
        :param length_to_track: (dict), the corresponding length of the subjects' sessions.
        :param train_mode: (boolean), the flag for toggle the train /eval mode of the model.
        :return: (tensor), (dict), the loss and the metric dictionary.
        """
        self.model.eval()
        return self.epoch_loop(data_to_load, length_to_track, clip_sample_map_to_track,
                               clipwise_frame_number, epoch, directory_to_save_plot, train_mode=train_mode)

    def fit(
            self,
            data_to_load,
            length_to_track,
            clip_sample_map_to_track,
            fold=None,
            epoch_number=10,
            early_stopping=5,
            min_epoch_number=10,
            clipwise_frame_number=160,
            checkpoint=None,
            directory_to_save_checkpoint_and_plot=None
    ):
        r"""
        The function to carry out training and validation.
        :param directory_to_save_checkpoint_and_plot:
        :param clip_sample_map_to_track:
        :param data_to_load: (dict), the data in training and validation partitions.
        :param length_to_track: (dict), the corresponding length of the subjects' sessions.
        :param fold: the current fold index.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param epoch_number: (int), how many epochs to run.
        :param early_stopping: (int), how many epochs to tolerate before stopping early.
        :param min_epoch_number: the minimum epochs to run before calculating the early stopping.
        :param checkpoint: (dict), to save the information once an epoch is done
        :return: (dict), the metric dictionary recording the output and its associate continuous labels
            as a long array for each subject.
        """

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, validate_losses = [], []
        early_stopping_counter = early_stopping
        start_epoch = 0

        best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10
        }

        combined_train_record_dict = {}
        combined_validate_record_dict = {}
        combined_record_dict = {'train': [], 'validate': []}

        if len(checkpoint.keys()) > 1:
            time_fit_start = checkpoint['time_fit_start']
            start_epoch = checkpoint['start_epoch']
            early_stopping_counter = checkpoint['early_stopping_counter']
            best_epoch_info = checkpoint['best_epoch_info']
            combined_train_record_dict = checkpoint['combined_train_record_dict']
            combined_validate_record_dict = checkpoint['combined_validate_record_dict']
            train_losses = checkpoint['train_losses']
            validate_losses = checkpoint['validate_losses']
            self.optimizer = checkpoint['optimizer']
            self.scheduler = checkpoint['scheduler']
            self.model.load_state_dict(best_epoch_info['model_weights'])

        # Loop the epochs
        for epoch in np.arange(start_epoch, epoch_number):
            time_epoch_start = time.time()

            # Get the losses and the record dictionaries for training and validation.
            train_loss, train_record_dict = self.train(
                data_to_load['train'], length_to_track['train'],
                clip_sample_map_to_track['train'],
                directory_to_save_checkpoint_and_plot, clipwise_frame_number, epoch)

            # Combine the record to a long array for each subject.
            combined_train_record_dict = self.combine_record_dict(
                combined_train_record_dict, train_record_dict)

            validate_loss, validate_record_dict = self.validate(
                data_to_load['validate'],
                length_to_track['validate'], clip_sample_map_to_track['validate'],
                directory_to_save_checkpoint_and_plot, clipwise_frame_number, epoch)

            combined_validate_record_dict = self.combine_record_dict(
                combined_validate_record_dict, validate_record_dict)

            # Calculate the mean metrics for a whole partition for information showcase.
            mean_train_record = train_record_dict['overall']
            mean_validate_record = validate_record_dict['overall']

            train_losses.append(train_loss)
            validate_losses.append(validate_loss)

            improvement = False

            # If a lower validate loss appears.
            if validate_loss < best_epoch_info['loss']:
                improvement = True
                best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'epoch': epoch,
                    'scalar_metrics': {
                        'train_loss': train_loss,
                        'validate_loss': validate_loss,
                    },
                    'array_metrics': {
                        'train_metric_record': mean_train_record,
                        'validate_metric_record': mean_validate_record
                    }
                }

            # Early stopping controller.
            if early_stopping and epoch > min_epoch_number:
                if improvement:
                    early_stopping_counter = early_stopping
                else:
                    early_stopping_counter -= 1

                if early_stopping_counter <= 0:
                    if self.verbose:
                        print("\nEarly Stop!!")
                    break

            if validate_loss < 0:
                print('validate loss negative')

            if self.verbose and fold is not None:
                print(
                    "\n Fold {:2}: Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | best={} | "
                    "improvement={}-{}".format(
                        fold + 1,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        int(best_epoch_info['epoch']) + 1,
                        improvement,
                        early_stopping_counter))
            else:
                print(
                    "\n Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | best={} | "
                    "improvement={}-{}".format(
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        int(best_epoch_info['epoch']) + 1,
                        improvement,
                        early_stopping_counter))

                print(mean_train_record)
                print(mean_validate_record)
                print("------")

            checkpoint['time_fit_start'] = time_fit_start
            checkpoint['start_epoch'] = epoch + 1
            checkpoint['early_stopping_counter'] = early_stopping_counter
            checkpoint['best_epoch_info'] = best_epoch_info
            checkpoint['combined_train_record_dict'] = combined_train_record_dict
            checkpoint['combined_validate_record_dict'] = combined_validate_record_dict
            checkpoint['train_losses'] = train_losses
            checkpoint['validate_losses'] = validate_losses
            checkpoint['optimizer'] = self.optimizer
            checkpoint['scheduler'] = self.scheduler
            checkpoint['fit_finished'] = False
            checkpoint['fold_finished'] = False
            if directory_to_save_checkpoint_and_plot:
                print("Saving checkpoint.")
                save_pkl_file(directory_to_save_checkpoint_and_plot, "checkpoint.pkl", checkpoint)
                print("Checkpoint saved.")

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(validate_loss)
            else:
                self.scheduler.step()

        checkpoint['fit_finished'] = True

        combined_record_dict['train'] = combined_train_record_dict
        combined_record_dict['validate'] = combined_validate_record_dict

        self.model.load_state_dict(best_epoch_info['model_weights'])

        if self.print_training_metric:
            print("------")
            time_elapsed = time.time() - time_fit_start
            print("Training completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

            print("Epoch with lowest val loss:", best_epoch_info['epoch'])
            for m in best_epoch_info['scalar_metrics']:
                print('{}: {:.5f}'.format(m, best_epoch_info['scalar_metrics'][m]))

            print("Train metric:", best_epoch_info['array_metrics']['train_metric_record'])
            print("Validate metric:", best_epoch_info['array_metrics']['validate_metric_record'])
            print("------")

        return combined_record_dict, checkpoint

    def combine_record_dict(self, main_record_dict, epoch_record_dict):
        r"""
        Append the metric recording dictionary of an epoch to a main record dictionary.
            Each single term from epoch_record_dict will be appended to the corresponding
            list in min_record_dict.
        Therefore, the minimum terms in main_record_dict are lists, whose element number
            are the epoch number.
        """

        # If the main record dictionary is blank, then initialize it by directly copying from epoch_record_dict.
        # Since the minimum term in epoch_record_dict is list, it is available to append further.
        if not bool(main_record_dict):
            main_record_dict = epoch_record_dict
            return main_record_dict

        # Iterate the dict and append each terms from epoch_record_dict to
        # main_record_dict.
        for (subject_id, main_subject_record), (_, epoch_subject_record) \
                in zip(main_record_dict.items(), epoch_record_dict.items()):

            # Go through emotions, e.g., Arousal and Valence.
            for emotion in self.emotional_dimension:
                # Go through metrics, e.g., rmse, pcc, and ccc.
                for metric in self.metrics:
                    # Go through the sub-dictionary belonging to each subject.
                    if subject_id != "overall":
                        session_dict = epoch_record_dict[subject_id][emotion][metric]
                        for session_id in session_dict.keys():
                            main_record_dict[subject_id][emotion][metric][session_id].append(
                                epoch_record_dict[subject_id][emotion][metric][session_id][0]
                            )

                    # In addition to subject-wise records, there are one extra sub-dictionary
                    # used to store the overall metrics, which is actually the partition-wise metrics.
                    # In this sub-dictionary, the results are obtained by first concatenating all the output
                    # and continuous labels to two long arraies, and then calculate the metrics.
                    else:
                        main_record_dict[subject_id][emotion][metric].append(
                            epoch_record_dict[subject_id][emotion][metric][0]
                        )

        return main_record_dict

    def initialize_subject_wise_output(self, length_to_track):
        r"""
        Initialize the dictionary to record subject wise output as a long array.
            The array will then be truncated according to the length dictionary.
        :param length_to_track: (dict), the session-wise length of subjects, will be used to
            calculate the length sum.
        """
        return {key: torch.zeros(int(np.sum(value)), len(self.emotional_dimension), dtype=torch.float32)
                for key, value in length_to_track.items()}

    def epoch_loop(self, data_to_load, length_to_track, clip_sample_map_to_track,
                   clipwise_frame_number, epoch, directory_to_save_plot, train_mode=True):
        r"""
        The function to carry out operations of a loop. Since our files cannot be loaded completely
            for a epoch, therefore we have to loop the data list in an epoch.
        :param clip_sample_map_to_track: (dict), a crucial argument, which records that for
            each sample having time_depth frames, which session it comes from. It will help to
            restore the output from clipped and shuffled data to the correct position for
            session-wise, subject-wise, and partition-wise.
        :param directory_to_save_plot: (str), the path to save the plot.
        :param clipwise_frame_number: (int), how many frames contained in a mp4 file.
        :param epoch: (int), the current epoch.
        :param data_to_load: (dict), the video filename and the associated continuous label to load for this epoch.
        :param length_to_track: (dict), the associated length dictionary. It helps to restore the output to a session-wise
            manner.
        :param train_mode: (boolean) , the flag to control the train/eval mode of the model.
        :return: (tensor), (dict), the epoch loss and the epoch metric record.
        """

        # How many group we have to load for an epoch.
        group_number = len(data_to_load)

        # These two objects focus on deal with the output and the corresponding continuous
        # labels. The reason to use them is as follows. The training first split the data of a
        # partition to N groups, each contains M video clips. It then split each M video clips
        # to K mini-batches, and obtain the output for each mini-batch. However, in order to
        # debug, analyze, evaluate, and visualize, it is required to restore the output to
        # partition-wise, subject-wise and session-wise manners. This is where the two objects
        # come to picture.
        output_handler = ContinuousOutputHandler(
            length_to_track, clip_sample_map_to_track, self.emotional_dimension,
            clipwise_frame_number, self.time_depth, self.step_size)

        continuous_label_handler = ContinuousOutputHandler(
            length_to_track, clip_sample_map_to_track, self.emotional_dimension,
            clipwise_frame_number, self.time_depth, self.step_size)

        # This object calculate the metrics, usually by root mean square error, pearson correlation
        # coefficient, and concordance correlation coefficient.
        metric_handler = ContinuousMetricsCalculator(self.metrics, self.emotional_dimension,
                                                     output_handler, continuous_label_handler)

        # For tracking the ongoing fitting and testing in the console.
        running_loss = 0.0
        total_batch_count = 0

        # Loop the groups.
        for index in range(group_number):
            # Load the data. Each time we load a group only. A group contains M video clips, maybe 32 or 64.
            # The larger the group number is, the more video clips are to be loaded,
            # and the more memory is required.
            # Note, graphic memory is not influenced by group number.
            dataset = EmotionalDataset(data_to_load[index], time_depth=self.time_depth)
            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            # Loop the loaded data of a group.
            running_loss, total_batch_count, output_handler, continuous_label_handler = self._loop(
                data_loader, output_handler, continuous_label_handler, data_to_load[index],
                running_loss=running_loss, total_batch_count=total_batch_count, train_mode=train_mode)

            # Show how many groups are processed.
            print_progress(index, group_number)

        # Calculate the running loss for this epoch.
        epoch_loss = running_loss / total_batch_count

        # Restore the output and continuous labels to its original shape, which is session-wise.
        # By which the metrics can be calculated for each session.
        # The metrics is the average across the session and subjects of a partition.
        output_handler.get_sessionwise_dict()
        continuous_label_handler.get_sessionwise_dict()

        # Restore the output and continuous labels to partition-wise, i.e., two very long
        # arrays.  It is used for calculating the metrics.
        output_handler.get_partitionwise_dict()
        continuous_label_handler.get_partitionwise_dict()

        # Compute the root mean square error, pearson correlation coefficient and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the metrics.
        metric_handler.calculate_metrics()
        epoch_result_dict = metric_handler.metric_record_dict

        # This object plot the figures and save them.
        plot_handler = PlotHandler(self.metrics, self.emotional_dimension, epoch_result_dict,
                                   output_handler.sessionwise_dict, continuous_label_handler.sessionwise_dict,
                                   epoch=epoch, train_mode=train_mode, directory_to_save_plot=directory_to_save_plot)
        plot_handler.save_output_vs_continuous_label_plot()

        return epoch_loss, epoch_result_dict

    def _loop(
            self, data_loader, output_handler, continuous_label_handler,
            data_to_load, running_loss=0, total_batch_count=0, train_mode=True):
        r"""
        One loop of mini_batch clips.
        :param data_loader: (dataloader), the dataloader containing the data of the mini-batch.
        :param data_to_load: (dict), the video files and their associate continuous labels.
        :param running_loss: (tensor), the loss in a loop.
        :param total_batch_count: (int), the count of how many mini-batch in an epoch.
        :param train_mode: (boolean), the flag to control the train/eval mode of a model.
        :return:
            (tensor), the running loss for showcase.
            (int), the mini-batch count for an epoch.
            (dict), the subject-wise output dictionary.
            (dict), the subject-wise continuous label dictionary.
        """

        # A typical model/optimizer updating forloop.
        for X, Y, sample_id in data_loader:

            if train_mode:
                self.optimizer.zero_grad()

            # Get the data.
            inputs = X.to(self.device)

            # Get the labels.
            labels = Y.to(self.device)

            # Feed the data to the model.
            # outputs = self.model(inputs, labels)
            outputs = self.model(inputs)

            # The mean square error loss.
            loss = self.criterion(outputs, labels)

            # Keep filling in the subject-wise dictionary the output from a mini-batch.
            output_handler.place_clip_output_to_subjectwise_dict(
                data_to_load['frame'], outputs.detach().cpu().numpy(), sample_id)
            continuous_label_handler.place_clip_output_to_subjectwise_dict(
                data_to_load['frame'], labels.detach().cpu().numpy(), sample_id)

            running_loss += loss.item()
            total_batch_count += 1

            # Update the model and optimizer for train mode.
            if train_mode:
                loss.backward()
                self.optimizer.step()

        return running_loss, total_batch_count, output_handler, continuous_label_handler


if __name__ == "__main__":
    length_to_track = [[1300, 1400], [1200, 1300]]
    output_handler = ContinuousOutputHandlerNPY(length_to_track, ['Arousal', 'Valence'])
    continuous_label_handler = ContinuousOutputHandlerNPY(length_to_track, ['Arousal', 'Valence'])

    outputs = np.ones((2, 300, 2))

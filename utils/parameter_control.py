import torch
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import utils
from model.prototype import my_res50
from torch import nn, optim


class GenericParamControl(object):
    def __init__(self, model):
        self.model = model
        self.module_list = self.init_module_list()
        self.module_stack = self.init_module_list()
        self.module_to_layer_mapping_index_mapping = self.init_module_to_layer_index_mapping()
        self.modules_to_release = []

    @staticmethod
    def init_module_list():
        # return ['input_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'output_layer']
        return ['0', '1', '2', '3', '4', '5', '6', '7']

    @staticmethod
    def init_module_to_layer_index_mapping():
        # return {'input_layer': slice(0, 4), 'layer1': slice(10, 37),
        #                                      'layer2': slice(37, 76),
        #                                      'layer3': slice(76, 205), 'layer4': slice(205, 235),
        #                                      'output_layer': slice(4, 10)}

        return {'0': slice(151, 160), '1': slice(160, 169), '2': slice(169, 178),
                '3': slice(178, 187), '4': slice(187, 196), '5': slice(196, 205),
                '6': slice(205, 235), '7': slice(4, 10)}

    def get_updating_module_names(self):
            self.modules_to_release.append(self.module_stack.pop())

    def release_parameters_to_update(self):

        self.get_updating_module_names()

        if self.modules_to_release:
            for module in self.modules_to_release:
                indices = self.module_to_layer_mapping_index_mapping[module]

                for param in list(self.model.parameters())[indices]:
                    # parameters.append({'params': param})
                    # parameters[-1]['params'].requires_grad = True
                    param.requires_grad = True

class GenericParamControl2(object):
    def __init__(self, model):
        self.model = model
        self.module_list = self.init_module_list()
        self.module_stack = self.init_module_list()
        self.module_to_layer_mapping_index_mapping = self.init_module_to_layer_index_mapping()
        self.modules_to_release = []

    @staticmethod
    def init_module_list():
        # return ['input_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'output_layer']
        return ['0', '1', '2', '3', '4', '5', '6', '7']

    @staticmethod
    def init_module_to_layer_index_mapping():
        # return {'input_layer': slice(0, 4), 'layer1': slice(10, 37),
        #                                      'layer2': slice(37, 76),
        #                                      'layer3': slice(76, 205), 'layer4': slice(205, 235),
        #                                      'output_layer': slice(4, 10)}

        return {'0': slice(0, 4), '1': slice(160, 165), '2': slice(165, 170),
                '3': slice(170, 180), '4': slice(180, 190), '5': slice(190, 205),
                '6': slice(205, 235), '7': slice(4, 10)}

    def get_updating_module_names(self):
            self.modules_to_release.append(self.module_stack.pop())

    def release_parameters_to_update(self):

        self.get_updating_module_names()

        if self.modules_to_release:
            for module in self.modules_to_release:
                indices = self.module_to_layer_mapping_index_mapping[module]

                for param in list(self.model.parameters())[indices]:
                    # parameters.append({'params': param})
                    # parameters[-1]['params'].requires_grad = True
                    param.requires_grad = True


class Avec19ParamControl(GenericParamControl):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def init_module_to_layer_index_mapping():
        # return {'input_layer': slice(0, 4), 'layer1': slice(10, 37),
        #                                      'layer2': slice(37, 76),
        #                                      'layer3': slice(76, 205), 'layer4': slice(205, 235),
        #                                      'output_layer': slice(4, 10)}

        return {'0': slice(151, 160), '1': slice(160, 169), '2': slice(169, 178),
                '3': slice(178, 187), '4': slice(187, 196), '5': slice(196, 205),
                '6': slice(205, 235), '7': slice(4, 10)}

        # return {'0': slice(151, 160), '1': slice(160, 169), '2': slice(169, 178),
        #         '3': slice(178, 187), '4': slice(187, 196), '5': slice(196, 205),
        #         '6': slice(205, 235), '7': slice(4, 10)}


class GenericReduceLROnPlateau(object):
    def __init__(self, patience, min_epoch, learning_rate, num_release, milestone):
        self.learning_rate = learning_rate
        self.original_learning_rate = learning_rate
        self.patience = patience
        self.milestone = milestone
        self.min_epoch = min_epoch
        self.plateau_count = 0
        self.release_count = num_release
        self.lowest_loss = 1e7
        self.updated = False
        self.to_release = False
        self.released = False
        self.halt = False

    def count_plateau(self, epoch, current_loss):
        if epoch > self.min_epoch:
            if current_loss < self.lowest_loss:
                self.lowest_loss = current_loss
                self.plateau_count = 0
            else:
                self.plateau_count += 1

    def update_lr(self):

        if self.released:
            self.learning_rate = self.original_learning_rate
            self.release_count -= 1
            self.released = False
        else:
            self.learning_rate *= 0.1

        self.plateau_count = 0

        if self.learning_rate < 1e-7 and self.release_count == 0:
            self.halt = True
        elif self.learning_rate < 1e-7 and self.release_count > 0:
            self.to_release = True

    def step(self, epoch, current_loss):
        self.count_plateau(epoch, current_loss)
        if self.plateau_count == self.patience:
            self.update_lr()
            self.updated = True

    def update_milestone(self, epoch, add_milestone):
        self.milestone[-1] = epoch
        self.milestone.append(epoch + add_milestone)
        return self.milestone


def get_parameters(model):
    r"""
    Get the parameters to update.
    :return:
    """
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)

    model = my_res50(num_classes=8, use_pretrained=True)
    params_to_update = get_parameters(model)
    optimizer = optim.SGD(params_to_update, lr=0.01, weight_decay=0.001, momentum=0.9)

    milestone = [0]
    module_list = ['input_layer', 'layer1', 'layer2', 'layer3', 'layer4', 'output_layer']
    module_to_layer_index_mapping = {'input_layer': slice(0, 4), 'layer1': slice(10, 37), 'layer2': slice(37, 76), 'layer3': slice(76, 205), 'layer4': slice(205, 235), 'output_layer': slice(4, 10)}
    a = GenericParamControl(model)
    loss = torch.ones(100)
    b = GenericReduceLROnPlateau(patience=2, min_epoch=0, learning_rate=0.001, num_release=4, milestone=milestone)

    for index, epoch in enumerate(range(100)):

        if epoch in milestone or b.to_release:
            a.release_parameters_to_update()
            b.released = True
            b.update_lr()
            b.to_release = False
            milestone = b.update_milestone(epoch, add_milestone=5)
            params_to_update = get_parameters(model)
            optimizer = optim.SGD(params_to_update, lr=b.learning_rate, weight_decay=0.001, momentum=0.9)

        b.step(epoch, loss[index])

        if b.updated:
            b.updated = False
            params_to_update = get_parameters(model)
            optimizer = optim.SGD(params_to_update, lr=b.learning_rate, weight_decay=0.001, momentum=0.9)


        if b.halt:
            print(0)





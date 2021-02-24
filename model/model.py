import os

import torch
import torch.nn as nn
from torchvision.models import vgg11_bn
import torch.nn.functional as F
from model.inception_resnet_v1 import InceptionResnetV1
from model.inception_resnet_v2 import InceptionResNetV2
import torch.nn.init as init
from model.arcface_model import Backbone


def init_weight_cfer(layer):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(layer, nn.Conv1d):
        init.normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.Conv2d):
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)


class CAModule(nn.Module):
    '''Channel Attention Module'''
    def __init__(self, channels, reduction):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        x = self.sigmoid(x)
        return input * x


class SAModule(nn.Module):
    '''Spatial Attention Module'''
    def __init__(self):
        super(SAModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return input * x

class CFER(nn.Module):
    def __init__(self, num_classes):
        super(CFER, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).apply(init_weight_cfer)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).apply(init_weight_cfer)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(36864, 64).apply(init_weight_cfer)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64).apply(init_weight_cfer)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes).apply(init_weight_cfer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x)
        x = self.fc2(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x)
        x = self.fc3(x)
        return x


class VideoEmbedding(nn.Module):
    def __init__(self, frame_dim, time_depth, dropout_rate, embedding_dim, shared_linear_dim1, shared_linear_dim2):
        super().__init__()
        self.frame_dim = frame_dim
        self.time_depth = time_depth
        self.embedding_dim = embedding_dim
        self.shared_linear_dim1 = shared_linear_dim1
        self.shared_linear_dim2 = shared_linear_dim2
        self.drop = nn.Dropout(dropout_rate)

        self.spatial_model = self.init_spatial_model()
        self.intermediate_size = self.determine_size()

        self.shared_fc_layers = self.init_shared_fc_layers()

    def determine_size(self):
        pseudo_data = torch.zeros(self.time_depth, 3, self.frame_dim, self.frame_dim)
        pseudo_intermediate = self.spatial_model(pseudo_data)
        _, dim1, dim2, dim3 = pseudo_intermediate.shape
        return dim1 * dim2 * dim3

    def init_spatial_model(self):
        spatial_model = vgg11_bn(pretrained=True)
        for param in spatial_model.features[:11].parameters():
            param.requires_grad = False
        spatial_model.features[11:].apply(init_weight)
        return spatial_model.features

    def init_shared_fc_layers(self):
        shared_fc_layers = nn.Sequential(
            nn.Linear(self.intermediate_size, self.shared_linear_dim1),
            nn.ReLU(),
            nn.Linear(self.shared_linear_dim1, self.shared_linear_dim2),
            nn.ReLU()).apply(init_weight)
        return shared_fc_layers

    def forward(self, x):
        x = x.view(-1, 3, self.frame_dim, self.frame_dim)
        x = self.spatial_model(x)
        x = self.drop(x)
        x = x.view(x.shape[0], -1)
        x = self.shared_fc_layers(x)
        x = self.drop(x)
        x = x.view(-1, self.time_depth, self.embedding_dim)
        return x


class EmotionEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True).apply(init_weight)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        embedded = self.dropout(x)
        output, (hidden, cell) = self.rnn(embedded)

        return output, hidden, cell


class EmotionDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, dropout_rate):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(output_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True).apply(init_weight)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.fc_out(output)

        return prediction, hidden, cell


class EmotionSeq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, device):
        super().__init__()

        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target):
        # Dimension of source = [batch, sequence, channel, width, height]
        # Dimension of output = [batch, sequence, output_dim]
        batch_size = target.shape[0]
        target_length = target.shape[1]
        target_emotion_dimension = self.decoder.output_dim

        outputs = torch.zeros(batch_size, target_length, target_emotion_dimension).to(self.device)

        embedding = self.embedding(source)
        _, hidden, cell = self.encoder(embedding)

        input = target[:, 0, :]

        for t in range(1, target_length):

            output, hidden, cell = self.decoder(input[:, None, :], hidden, cell)

            outputs[:, t, :] = output[:, 0, :]

        return outputs


class EmotionLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate):
        super().__init__()

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout_rate, batch_first=True, bidirectional=True).apply(init_weight)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim).apply(init_weight)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.drop(x)
        x = self.fc_out(x)
        return x


class EmotionSpatialTemporalModel(nn.Module):
    def __init__(self, spatial_embedding, temporal_block):
        super().__init__()

        self.spatial_embedding = spatial_embedding
        self.temporal_block = temporal_block

    def forward(self, x):
        x = self.spatial_embedding(x)
        x = self.temporal_block(x)
        return x






def init_weight(layer):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(layer, nn.Conv1d):
        init.normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.Conv2d):
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.Conv3d):
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.ConvTranspose1d):
        init.normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.ConvTranspose2d):
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.ConvTranspose3d):
        init.xavier_normal_(layer.weight.data)
        if layer.bias is not None:
            init.normal_(layer.bias.data)
    elif isinstance(layer, nn.BatchNorm1d):
        init.normal_(layer.weight.data, mean=1, std=0.02)
        init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.BatchNorm2d):
        init.normal_(layer.weight.data, mean=1, std=0.02)
        init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.BatchNorm3d):
        init.normal_(layer.weight.data, mean=1, std=0.02)
        init.constant_(layer.bias.data, 0)
    elif isinstance(layer, nn.Linear):
        init.xavier_normal_(layer.weight.data)
        init.normal_(layer.bias.data)
    elif isinstance(layer, nn.LSTM):
        for param in layer.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(layer, nn.LSTMCell):
        for param in layer.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(layer, nn.GRU):
        for param in layer.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(layer, nn.GRUCell):
        for param in layer.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class InceptResV1(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.spatial_model =  InceptionResnetV1(num_classes=num_classes)
        # for param in self.spatial_model.parameters():
        #     param.requires_grad = False
        # self.spatial_model = torch.nn.Sequential(*(list(self.spatial_model.children())[:-1]))
        self.fc1 = nn.Linear(7168, 512)
        self.activation1 = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.spatial_model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation1(x)
        x = self.bn1(x)
        x = self.classifier(x)
        return x

class InceptResV2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.spatial_model =  InceptionResNetV2()
        # for param in self.spatial_model.parameters():
        #     param.requires_grad = False
        # self.spatial_model = torch.nn.Sequential(*(list(self.spatial_model.children())[:-1]))
        self.fc1 = nn.Linear(1001, 512)
        self.activation1 = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.spatial_model(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation1(x)
        x = self.bn1(x)
        x = self.classifier(x)
        return x

class Net(nn.Module):
    def __init__(self, out_ns, device, bidirectional, n_layers, n_dropout=0.3):
        super(Net, self).__init__()
        self.out_ns = out_ns
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        """
        self.sharedcvlayer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),#224*224*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#112*112*64
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#56*56*128
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        """

        spatial_model = vgg11_bn(pretrained=True).eval()
        # spatial_model = InceptionResnetV1(pretrained='vggface2').eval()
        for param in spatial_model.parameters():
            param.requires_grad = False

        BackBone = spatial_model.features[:11]
        self.BackBone = BackBone
        add_block_cv = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28*28*256
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*512
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)).apply(init_weight)  # 7*7*512

        self.add_block_cv = add_block_cv

        self.sharedfclayer = nn.Sequential(
            nn.Linear(5 * 5 * 512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()).apply(init_weight)

        self.rnn = nn.LSTM(input_size=1024,
                           hidden_size=512,
                           num_layers=self.n_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional).apply(init_weight)

        if self.bidirectional:
            self.dropout = nn.Dropout(0.5)
            self.fe_out = nn.Linear(in_features=512 * 2, out_features=self.out_ns).apply(init_weight)  # AU predictions
        else:
            self.dropout = nn.Dropout(0.5)
            self.fe_out = nn.Linear(in_features=512, out_features=self.out_ns).apply(init_weight)  # AU predictions

    def forward(self, x):
        x = x.view(-1, 3, 160, 160)
        backbone = self.BackBone(x)
        blk_cv = self.add_block_cv(backbone)
        h_sharedcv_flatten = blk_cv.view(blk_cv.size(0), -1)
        h_sharedfc = self.sharedfclayer(h_sharedcv_flatten)
        fe_fcs1 = h_sharedfc.view(-1, 64, 1024)

        # fe_fcs = []
        # for t in range(x.shape[1]):  # x.shape[1]: video length
        #     backbone = self.BackBone(x[:, t, :, :, :])
        #     blk_cv = self.add_block_cv(backbone)
        #     # blk_cv = backbone
        #     h_sharedcv_flatten = blk_cv.view(blk_cv.size(0), -1)  # flatten
        #     h_sharedfc = self.sharedfclayer(h_sharedcv_flatten)
        #     fe_fcs.append(h_sharedfc)
        # fe_fcs1 = torch.stack(fe_fcs, dim=0).transpose_(1, 0)  # fe_fcs1: shape=(batch, time_step, input_size)

        output, (h_n, c_n) = self.rnn(fe_fcs1)

        # if self.bidirectional:
        #     output1 = output.reshape(-1, 512 * 2 * self.n_layers)
        #     output2 = self.dropout(output1)
        # else:
        #     output1 = output.reshape(-1, 512 * self.layers)
        #     output2 = self.dropout(output1)

        fe_out = self.fe_out(output)
        return fe_out

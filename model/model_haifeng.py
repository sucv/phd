import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(nn.Module):
    '''Squeeze and Excitation Module'''
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x

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

class BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       #nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class SE_BottleNeck_IR(nn.Module):
    '''Improved Residual Bottlenecks with Squeeze and Excitation Module'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(SE_BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       SEModule(out_channel, 16))
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res

class CBAM_BottleNeck_IR(nn.Module):
    '''Improved Residual Bottleneck with Channel Attention and Spatial Attention'''
    def __init__(self, in_channel, out_channel, stride, dim_match):
        super(CBAM_BottleNeck_IR, self).__init__()
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       CAModule(out_channel, 16),
                                       SAModule()
                                       )
        if dim_match:
            self.shortcut_layer = None
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)

        if self.shortcut_layer is not None:
            shortcut = self.shortcut_layer(x)

        return shortcut + res


filter_list = [64, 64, 128, 256, 512]
def get_layers(num_layers):
    if num_layers == 18:
        return [2, 2, 2, 2]
    elif num_layers == 50:
        return [3, 4, 14, 3]
    elif num_layers == 100:
        return [3, 13, 30, 3]
    elif num_layers == 152:
        return [3, 8, 36, 3]


class CBAMResNet_IR(nn.Module):
    def __init__(self, num_layers, feature_dim=512, conv1_stride=1, drop_ratio=0.4, mode='cbam_ir',filter_list=filter_list):
        super(CBAMResNet_IR, self).__init__()
        assert num_layers in [18, 50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'se_ir', 'cbam_ir'], 'mode should be ir, se_ir or cbam_ir'
        layers = get_layers(num_layers)
        if mode == 'ir':
            block = BottleNeck_IR
        elif mode == 'se_ir':
            block = SE_BottleNeck_IR
        elif mode == 'cbam_ir':
            block = CBAM_BottleNeck_IR

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=conv1_stride)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, out_channel, blocks, stride):
        layers = []
        layers.append(block(in_channel, out_channel, stride, False))
        for i in range(1, blocks):
            layers.append(block(out_channel, out_channel, 1, True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)

        return x


class FerIRResnet50(nn.Module):

    def __init__(self, opts):
        super(FerIRResnet50, self).__init__()

        self.num_classes = opts.num_classes
        self.feature_dim = opts.feature_dim
        self.drop_ratio = opts.drop_ratio

        self.backbone_block_expansion = 1
        self.backbone = CBAMResNet_IR(50, feature_dim=opts.feature_dim, conv1_stride=opts.conv1_stride, mode='ir')

        layer_index = {
            'body.0': 'layer1.0',
            'body.1': 'layer1.1',
            'body.2': 'layer1.2',
            'body.3': 'layer2.0',
            'body.4': 'layer2.1',
            'body.5': 'layer2.2',
            'body.6': 'layer2.3',
            'body.7': 'layer3.0',
            'body.8': 'layer3.1',
            'body.9': 'layer3.2',
            'body.10': 'layer3.3',
            'body.11': 'layer3.4',
            'body.12': 'layer3.5',
            'body.13': 'layer3.6',
            'body.14': 'layer3.7',
            'body.15': 'layer3.8',
            'body.16': 'layer3.9',
            'body.17': 'layer3.10',
            'body.18': 'layer3.11',
            'body.19': 'layer3.12',
            'body.20': 'layer3.13',
            'body.21': 'layer4.0',
            'body.22': 'layer4.1',
            'body.23': 'layer4.2',
        }

        if opts.pretrain_backbone:
            print('use pretrain_backbone from face recognition model...')
            if device.type == 'cuda':
                ckpt = torch.load(opts.pretrain_backbone)
            if device.type == 'cpu':
                ckpt = torch.load(opts.pretrain_backbone, map_location=lambda storage, loc: storage)
            # self.backbone.load_state_dict(ckpt['net_state_dict'])

            net_dict = self.backbone.state_dict()
            pretrained_dict = ckpt

            new_state_dict = {}
            num = 0
            for k, v in pretrained_dict.items():
                num = num + 1
                print('num: {}, {}'.format(num, k))
                for k_r, v_r in layer_index.items():
                    if k_r in k:
                        if k[k_r.__len__()] == '.':
                            k = k.replace(k_r, v_r)
                            break

                if (k in net_dict) and (v.size() == net_dict[k].size()):
                    new_state_dict[k] = v
                else:
                    print('error...')

            un_init_dict_keys = [k for k in net_dict.keys() if k not in new_state_dict]
            print("un_init_num: {}, un_init_dict_keys: {}".format(un_init_dict_keys.__len__(), un_init_dict_keys))
            print("\n------------------------------------")

            for k in un_init_dict_keys:
                new_state_dict[k] = torch.DoubleTensor(net_dict[k].size()).zero_()
                if 'weight' in k:
                    if 'bn' in k:
                        print("{} init as: 1".format(k))
                        constant_(new_state_dict[k], 1)
                    else:
                        print("{} init as: xavier".format(k))
                        xavier_uniform_(new_state_dict[k])
                elif 'bias' in k:
                    print("{} init as: 0".format(k))
                    constant_(new_state_dict[k], 0)

            print("------------------------------------")

            self.backbone.load_state_dict(new_state_dict)

        # self.backbone.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.backbone.output_layer = nn.Sequential(nn.BatchNorm2d(512 * self.backbone_block_expansion),
                                          nn.Dropout(self.drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * self.backbone_block_expansion * opts.spatial_size * opts.spatial_size,
                                                    self.feature_dim),
                                          nn.BatchNorm1d(self.feature_dim))

        self.output = nn.Linear(self.feature_dim, self.num_classes)

        for m in self.backbone.output_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.output.weight = xavier_uniform_(self.output.weight)
        self.output.bias = constant_(self.output.bias, 0)


    def forward(self, x):
        feature_2D = self.backbone(x)

        return feature_2D



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='AVEC2019_2D_Pretrain')
    parser.add_argument('--pretrain_backbone', type=str, default='../load/model_ir_se50.pth')  # ./data/backbone_ir50_ms1m_epoch63.pth
    parser.add_argument('--pretrain_backbone_2D', type=str, default='./data/0110.ckpt')
    parser.add_argument('--spatial_size', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--drop_ratio', type=float, default=0.4)
    parser.add_argument('--output_num', type=int, default=2)
    parser.add_argument('--conv1_stride', type=int, default=1)
    opts = parser.parse_args()

    input = torch.Tensor(2, 3, 40, 40)
    net = FerIRResnet50(opts)
    #print(net)

    x, feature_2D = net(input)
    print(x.shape)

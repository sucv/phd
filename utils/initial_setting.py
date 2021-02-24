import random
import numpy as np
import torch
import torch.backends.cudnn

from model.model import VideoEmbedding, EmotionLSTM, EmotionSpatialTemporalModel, EmotionEncoder, EmotionDecoder, EmotionSeq2Seq


def initialize_emotion_spatial_temporal_model(
    device,
    frame_dim,
    time_depth,
    shared_linear_dim1,
    shared_linear_dim2,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    dropout_rate_1,
    dropout_rate_2
):

    spatial_block = VideoEmbedding(frame_dim=frame_dim, time_depth=time_depth, embedding_dim=embedding_dim,
                                   shared_linear_dim1=shared_linear_dim1, shared_linear_dim2=shared_linear_dim2, dropout_rate=dropout_rate_1)
    # encoder = EmotionEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
    #                          n_layers=n_layers, dropout_rate=dropout_rate)
    # decoder = EmotionDecoder(hidden_dim=hidden_dim, output_dim=output_dim,
    #                          n_layers=n_layers, dropout_rate=dropout_rate)
    temporal_block = EmotionLSTM(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                 output_dim=output_dim, n_layers=n_layers, dropout_rate=dropout_rate_2)

    model = EmotionSpatialTemporalModel(spatial_block, temporal_block).to(device)
    # model = EmotionSeq2Seq(spatial_block, encoder, decoder, device)
    return model


def initial_setting(seed=0, gpu_index=None, cpu_thread_number=None):
    initialize_random_seed(seed)
    device = detect_device()
    # if gpu_index is not None:
    #     select_gpu(gpu_index)

    if cpu_thread_number is not None:
        set_cpu_thread(cpu_thread_number)
    return device


def initialize_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def detect_device():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cuda:1')
    return device


def select_gpu(index):
    r"""
    Choose which gpu to use.
    :param index: (int), the index corresponding to the desired gpu. For example,
        0 means the 1st gpu.
    """
    torch.cuda.set_device(index)


def set_cpu_thread(number):
    r"""
    Set the maximum thread of cpu for torch module.
    :param number: (int), the number of thread allowed, usually 1 is enough.
    """
    torch.set_num_threads(number)

import torch


class ccc_loss(object):
    def __init__(self):
        pass

    def __call__(self, gold, pred):
        # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
        # input (num_batches, seq_len, 1)
        # x = gold.view(-1, 1)
        # y = pred.view(-1, 1)
        #
        # x_mean = torch.mean(x)
        # y_mean = torch.mean(y)
        #
        # covariance = torch.mean((x - x_mean) * (y - y_mean))
        #
        # x_var = 1.0 / (len(x) - 1) * torch.sum(
        #     (x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
        # y_var = 1.0 / (len(y) - 1) * torch.sum((y - y_mean) ** 2)
        #
        # ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)


        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * ( pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2.*covariance / ((gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean))+ 1e-07)
        ccc_loss = 1. - ccc
        return ccc_loss
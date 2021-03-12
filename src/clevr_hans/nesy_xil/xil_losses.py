"""
Code for rrr loss.
@author: Wolfgang Stammer and Patrick Schramowski
@date: 11.05.2020
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from captum.attr import IntegratedGradients


def resize_mask_to_convoutput(mask, output):
    return torch.tensor(ndimage.zoom(mask.cpu(), (1, 1, output.shape[-2]/mask.shape[-2], output.shape[-1]/mask.shape[-1])))


log_softmax = torch.nn.LogSoftmax(dim=1).cuda()


def rrr_loss_function(A, X, y, logits, criterion, l2_grads=1000, reduce_func=torch.sum):
    log_prob_ys = log_softmax(logits)
    right_answer_loss = criterion(log_prob_ys, y)

    gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys).cuda(), create_graph=True)[0]

    for _ in range(len(gradXes.shape) - len(A.shape)):
        A = A.unsqueeze(dim=1)
    expand_list = [-1]*len(A.shape)
    expand_list[-3] = gradXes.shape[-3]
    A = A.expand(expand_list)
    A_gradX = torch.mul(A, gradXes) ** 2

    # right_reason_loss = l2_grads * torch.sum(A_gradX)
    right_reason_loss = torch.sum(A_gradX, dim=list(range(1, len(A_gradX.shape))))
    right_reason_loss = reduce_func(right_reason_loss)
    right_reason_loss *= l2_grads

    res = right_answer_loss + right_reason_loss
    return res, right_answer_loss, right_reason_loss


class LexiLoss:
    def __init__(self, class_weights, args):
        self.criterion = torch.nn.MSELoss()
        self.args = args

    def norm_saliencies(self, saliencies):
        saliencies_norm = saliencies.clone()

        for i in range(saliencies.shape[0]):
            if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
                saliencies_norm[i] = saliencies[i]
            else:
                saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                     (torch.max(saliencies[i]) - torch.min(saliencies[i]))

        return saliencies_norm

    def generate_intgrad_captum_table(self, net, input, labels):
        labels = labels.to("cuda")
        explainer = IntegratedGradients(net)
        saliencies = explainer.attribute(input, target=labels)
        # remove negative attributions
        saliencies[saliencies < 0] = 0.
        return self.norm_saliencies(saliencies)

    def __call__(self, net, target_set, class_ids, masks, epoch, batch_id, writer=None, writer_prefix=""):

        masks = masks.squeeze(dim=1).double()
        saliencies = self.generate_intgrad_captum_table(net, target_set, class_ids).squeeze(dim=1)

        assert len(saliencies.shape) == 3
        assert masks.shape == saliencies.shape

        loss = self.criterion(saliencies, masks)

        return loss

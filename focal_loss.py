import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = inputs
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return torch.mean(loss)
#
# def sigmoid_focal_loss(
#     inputs: torch.Tensor,
#     targets: torch.Tensor,
#     alpha: float = 0.25,
#     gamma: float = 2,
#     reduction: str = "none",
# ) -> torch.Tensor:
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#
#     Args:
#         inputs (Tensor): A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha (float): Weighting factor in range (0,1) to balance
#                 positive vs negative examples or -1 for ignore. Default: ``0.25``.
#         gamma (float): Exponent of the modulating factor (1 - p_t) to
#                 balance easy vs hard examples. Default: ``2``.
#         reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
#                 ``'none'``: No reduction will be applied to the output.
#                 ``'mean'``: The output will be averaged.
#                 ``'sum'``: The output will be summed. Default: ``'none'``.
#     Returns:
#         Loss tensor with the reduction option applied.
#     """
#     # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
#
#     p = inputs
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = p * targets + (1 - p) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)
#
#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss
#
#     return loss


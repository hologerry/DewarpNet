import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_label):
        n, c, w, h = pred_label.size()
        loss = torch.tensor(0.0, device=pred_label.device)
        for i in range(w-1):
            for j in range(h-1):
                loss += F.l1_loss(pred_label[:, :, i, j], pred_label[:, :, i+1, j], reduction='sum')
                loss += F.l1_loss(pred_label[:, :, i, j], pred_label[:, :, i, j+1], reduction='sum')

        loss /= w * h
        return loss

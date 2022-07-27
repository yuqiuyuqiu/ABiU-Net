import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import compute_grad_mag


def get_loss():
    criterion = JointSegLoss().cuda()
    return criterion

class JointSegLoss(nn.Module):
    def __init__(self):
        super(JointSegLoss, self).__init__()
        self.seg_weight = 1 #args.seg_weight
        self.dual_weight = 1#args.dual_weight
        self.ignore_label = 255
        self.batch_weighting = True

        self.seg_loss = CrossEntropyLoss2d(class_balance=self.batch_weighting, ignore_label=self.ignore_label)
        self.dual_task = DualTaskLoss(ignore_label=self.ignore_label)

    def forward(self, preds, target):
        self.seg_loss = self.seg_loss.cuda()
        losses = {}
        losses['seg_loss'] = self.seg_weight * self.seg_loss(preds, target)
        losses['dual_loss'] = self.dual_weight * self.dual_task(preds, target)
        return losses


class DualTaskLoss(nn.Module):
    def __init__(self, cuda=True, deep_weight=0.4, ignore_label=255):
        super(DualTaskLoss, self).__init__()
        self.cuda = cuda
        self.deep_weight = deep_weight
        self.ignore_label = ignore_label

    def forward(self, preds, target):
        pred = preds[0].squeeze(1)
        N, H, W = pred.shape
        ignore_mask = (target == self.ignore_label).detach()
        pred_mask = torch.where(ignore_mask, torch.zeros(N, H, W).cuda(), pred)
        gt_mask = torch.where(ignore_mask, torch.zeros(N, H, W).cuda(), target.float())

        pred_mask = compute_grad_mag(pred_mask.unsqueeze(1), cuda=self.cuda)
        gt_mask = compute_grad_mag(gt_mask.unsqueeze(1), cuda=self.cuda)
        loss_ewise = [F.l1_loss(pred_mask, gt_mask)]
        for idx in range(1, len(preds)):
            pred = preds[idx].squeeze(1)
            pred_mask = torch.where(ignore_mask, torch.zeros(N, H, W).cuda(), pred)
            pred_mask = compute_grad_mag(pred_mask.unsqueeze(1), cuda=self.cuda)
            loss = F.l1_loss(pred_mask, gt_mask)
            loss_ewise.append(self.deep_weight * loss)
        return sum(loss_ewise)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, class_balance=False, deep_weight=0.4, ignore_label=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.class_balance = class_balance
        self.deep_weight = deep_weight
        self.ignore_label = ignore_label

    def _cross_entropy_loss_weights(self, target):
        if self.class_balance:
            weights = torch.ones_like(target, dtype=torch.float)
            for idx in range(target.shape[0]):
                num_neg = torch.sum((target[idx] == 0).float())
                num_pos = torch.sum((target[idx] == 1).float())
                weights[idx][target[idx] == 0] = num_pos / (num_pos + num_neg)
                weights[idx][target[idx] == 1] = num_neg / (num_pos + num_neg)
                weights[idx][target[idx] == self.ignore_label] = 0
        else:
            weights = torch.ones_like(target, dtype=torch.float)
            weights[target == self.ignore_label] = 0
        return weights

    def forward(self, preds, target):
        #preds, target = inputs[:-1], inputs[-1]
        weights = self._cross_entropy_loss_weights(target)
        losses = [F.binary_cross_entropy(preds[0].squeeze(1), target.float(), weights)]
        for idx in range(1, len(preds)):
            loss = F.binary_cross_entropy(preds[idx].squeeze(1), target.float(), weights)
            losses.append(self.deep_weight * loss)
        return sum(losses)

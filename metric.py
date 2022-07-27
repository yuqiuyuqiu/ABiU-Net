import torch
import numpy as np

# adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class SalEval(object):
    def __init__(self, nthresh=49):
        self.nthresh = nthresh
        self.thresh = np.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh)
        self.EPSILON = np.finfo(np.float).eps

        self.recall = np.zeros((nthresh,))
        self.precision = np.zeros((nthresh,))
        self.mae = 0
        self.num = 0

    def add_batch(self, predict, gth):
        assert len(predict.shape) == 3 and len(gth.shape) == 3
        for t in range(self.nthresh):
            bi_res = predict > self.thresh[t]
            intersection = np.sum(np.sum(np.logical_and(gth == bi_res, gth), axis=1), axis=1)
            self.recall[t] += np.sum(intersection * 1. / (np.sum(np.sum(gth, axis=1), axis=1) + np.finfo(np.float).eps))
            self.precision[t] += np.sum(intersection * 1. / (np.sum(np.sum(bi_res, axis=1), axis=1) + np.finfo(np.float).eps))

        self.mae += np.sum(np.fabs(gth - predict)) * 1. / (gth.shape[1] * gth.shape[2])
        self.num += gth.shape[0]

    def get_metric(self):
        tr = self.recall / self.num
        tp = self.precision / self.num
        MAE = self.mae / self.num
        F_beta = (1 + 0.3) * tp * tr / (0.3 * tp + tr + np.finfo(np.float).eps)

        return np.max(F_beta), MAE

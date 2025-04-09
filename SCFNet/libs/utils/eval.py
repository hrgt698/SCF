# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np
import torch
from torchmetrics.functional.classification.f_beta import binary_f1_score

def db_eval_iou_multi(annotations, segmentations):
    iou = 0.0
    batch_size = annotations.shape[0]

    for i in range(batch_size):
        annotation = annotations[i, 0, :, :]
        segmentation = segmentations[i, 0, :, :]

        iou += db_eval_iou(annotation, segmentation)

    iou /= batch_size
    return iou


def db_eval_iou(annotation,segmentation,threshold=0.5):

    annotation = annotation > threshold
    segmentation = segmentation > threshold

    if np.isclose(np.sum(annotation), 0) and\
            np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation), dtype=np.float32)
                
def db_eval_F1(annotation,segmentation,threshold=0.5):
    annotation = annotation > threshold
    segmentation = segmentation > threshold
    #F1
    pp=torch.tensor(segmentation)
    gt=torch.tensor(annotation)
    pp_neg = ~ pp
    f1_pos = binary_f1_score(pp, gt)
    f1_neg = binary_f1_score(pp_neg, gt)
    if f1_neg > f1_pos:
        return float(f1_neg)
    else:
        return float(f1_pos)

def db_eval_MCC(mask1,mask2,threshold=0.5): #gt,pred
    mask1 = mask1 > threshold
    mask2 = mask2 > threshold
    TP = np.float64(np.sum(np.logical_and(mask1 == 1, mask2 == 1)))
    TN = np.float64(np.sum(np.logical_and(mask1 == 0, mask2 == 0)))
    FP = np.float64(np.sum(np.logical_and(mask1 == 0, mask2 == 1)))
    FN = np.float64(np.sum(np.logical_and(mask1 == 1, mask2 == 0)))
    
    mcc = np.abs(TP*TN - FP*FN) / np.maximum(np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) ), 1e-32)
    return mcc

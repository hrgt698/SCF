
""" Configuration file."""

import yaml
import numpy as np
from enum import Enum
import os.path as osp
from easydict import EasyDict as edict

# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from FAST-RCNN (Ross Girshick)
# ----------------------------------------------------------------------------

class phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TESTDEV = 'test-dev'
    TRAINVAL = 'train-val'


__C = edict()

# Public access to configuration settings
cfg = __C

# Number of CPU cores used to parallelize evaluation.
__C.N_JOBS = 32

# Paths to dataset folders
__C.PATH = edict()

# Dataset resolution: ("480p","1080p")
__C.RESOLUTION="480p"

# Dataset year: ("2016","2017")
__C.YEAR = "2017"

__C.PHASE = phase.VAL

# Multiobject evaluation (Set to False only when evaluating DAVIS 2016)
__C.MULTIOBJECT = True

# Root folder of project
__C.PATH.ROOT = osp.abspath('/users/u202220081200014/video-inpainting/video_forgery/HCPN_VOS')

# Data folder
__C.PATH.DATA = osp.abspath(('/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/train'))

# Path to input images
__C.PATH.SEQUENCES = osp.join(__C.PATH.DATA, "videos")

# Path to annotations
__C.PATH.ANNOTATIONS = osp.join(__C.PATH.DATA, "masks")

__C.PATH.ANNOTATIONS_CTR = osp.join(__C.PATH.DATA, "Annotations_ctr")

__C.PATH.FLOW = osp.join(__C.PATH.DATA, 'Flow')
__C.PATH.HED = osp.join(__C.PATH.DATA, 'davis-hed/')

# Color palette
__C.PATH.PALETTE = osp.abspath(osp.join(__C.PATH.ROOT, 'libs/dataset/palette.txt'))

# Paths to files
__C.FILES = edict()

# Path to property file, holding information on evaluation sequences.
__C.FILES.DB_INFO = osp.abspath(osp.join(__C.PATH.ROOT,
                                         "libs/dataset/db_info.yaml"))

# Measures and Statistics
__C.EVAL = edict()

# Metrics: J: region similarity, F: contour accuracy, T: temporal stability
__C.EVAL.METRICS = ['J', 'F']

# Statistics computed for each of the metrics listed above
__C.EVAL.STATISTICS = ['mean', 'recall', 'decay']


def db_read_info():
    """ Read dataset properties from file."""
    with open(cfg.FILES.DB_INFO, 'r') as f:
        return edict(yaml.load(f,Loader=yaml.FullLoader))


def db_read_attributes():
    """ Read list of sequences. """
    return db_read_info().attributes


def db_read_years():
    """ Read list of sequences. """
    return db_read_info().years


def db_read_sequences(year=None,db_phase=None):
    """ Read list of sequences. """

    sequences = db_read_info().sequences

    if year is not None:
        sequences = filter(
            lambda s: int(s.year) <= int(year), sequences)

    if db_phase is not None:
        if db_phase == phase.TRAINVAL:
            sequences = filter(lambda s: ((s.set == phase.VAL) or
                                          (s.set == phase.TRAIN)), sequences)
        else:
            sequences = filter(lambda s: s.set == db_phase, sequences)
    # return list(sequences)
    return sequences

# Load all sequences
__C.SEQUENCES = dict([(sequence.name, sequence)
                      for sequence in db_read_sequences()])

__C.palette = np.loadtxt(__C.PATH.PALETTE, dtype=np.uint8).reshape(-1, 3)

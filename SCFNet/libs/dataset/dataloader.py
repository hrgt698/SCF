# 中间分支为光流
import os
import cv2
import glob
import lmdb
import numpy as np
from PIL import Image
import os.path as osp
from skimage.transform import resize as imresize

from torch.utils import data

from torchvision import transforms
from .base import Sequence, Annotation
from libs.dataset import transform as tr

from libs.utils.config_standard_db import db_video_list
from libs.utils.config_standard_db import cfg
# from libs.utils.config_davis import cfg as cfg_davis
# from libs.utils.config_davis import db_read_sequences as db_read_sequences_davis
#from libs.utils.config_youtubevos import cfg as cfg_youtubevos
#from libs.utils.config_youtubevos import db_read_sequences_train as db_read_sequences_train_youtubevos


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def print_list_davis(imagefile):
    temp = []
    imagefiles = []
    temp.extend(imagefile[: :])
    temp.extend(imagefile[1: -1:])#帧组排列顺序：12 23 34 45 
    temp.sort()
    li = func(temp, 2)
    for i in li:
        imagefiles.append(i)
    return imagefiles


class DataLoader(data.Dataset):

    def __init__(self, args, split, input_size, augment=False,
                 transform=None, target_transform=None, pre_train=False):
        self._phase = split
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.augment = augment
        self.augment_transform = None
        self.pre_train = pre_train
        self._single_object = False
        
        if augment:
            self.augment_transform = transforms.Compose([
                tr.RandomHorizontalFlip(),
                tr.ScaleNRotate(rots=(-args.rotation, args.rotation),
                                scales=(.75, 1.25))])

        self.image_files = []
        self.mask_files = []
        self.flow_files = []
        self.contour_files = []
        self.hed_files = []

        if split == 'train':
            self.load_davis(args)
        else:
            self.load_davis(args)

    def __len__(self):
        return len(self.flow_files)

    def __getitem__(self, index):

        image_file1 = self.image_files[index][0]
        image_file2 = self.image_files[index][1]
        mask_file1 = self.mask_files[index][0]
        mask_file2 = self.mask_files[index][1]
        flow_file = self.flow_files[index]
       

        image1 = Image.open(image_file1).convert('RGB')
        image2 = Image.open(image_file2).convert('RGB')
        flow = Image.open(flow_file).convert('RGB')

        mask1 = cv2.imread(mask_file1, 0)
        mask1[mask1 > 0] = 255
        mask2 = cv2.imread(mask_file2, 0)
        mask2[mask2 > 0] = 255

        mask1 = Image.fromarray(mask1)
        mask2 = Image.fromarray(mask2)

        if self.input_size is not None:
            image1 = imresize(image1, self.input_size)
            flow = imresize(flow, self.input_size)
            mask1 = imresize(mask1, self.input_size, interp='nearest')

            image2 = imresize(image2, self.input_size)
            mask2 = imresize(mask2, self.input_size, interp='nearest')

        sample = {'image1': image1, 'image2': image2, 'flow': flow,
                  'mask1': mask1, 'mask2': mask2}

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image1, image2, flow, mask1, mask2 =\
            sample['image1'], sample['image2'], sample['flow'],\
            sample['mask1'], sample['mask2']

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            flow = self.transform(flow)

        if self.target_transform is not None:
            mask1 = mask1[:, :, np.newaxis]
            mask2 = mask2[:, :, np.newaxis]
          
            mask1 = self.target_transform(mask1)
            mask2 = self.target_transform(mask2)
            ctr1 = self.target_transform(ctr1)
            ctr2 = self.target_transform(ctr2)
            negative_pixels1 = self.target_transform(negative_pixels1)
            negative_pixels2 = self.target_transform(negative_pixels2)

        return image1, image2, flow, mask1, mask2, ctr1, ctr2, negative_pixels1, negative_pixels2

    # def load_youtubevos(self, args):

    #     self._db_sequences = db_read_sequences_train_youtubevos()

    #     # Check lmdb existance. If not proceed with standard dataloader.
    #     lmdb_env_seq_dir = osp.join(cfg_youtubevos.PATH.DATA, 'lmdb_seq')
    #     lmdb_env_annot_dir = osp.join(cfg_youtubevos.PATH.DATA, 'lmdb_annot')

    #     if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
    #         lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
    #         lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
    #     else:
    #         lmdb_env_seq = None
    #         lmdb_env_annot = None
    #         print('LMDB not found. This could affect the data loading time.'
    #               ' It is recommended to use LMDB.')

    #     # Load sequences
    #     self.sequences = [Sequence(self._phase, s, lmdb_env=lmdb_env_seq)
    #                       for s in self._db_sequences]

    #     # Load sequences
    #     videos = []
    #     for seq, s in zip(self.sequences, self._db_sequences):
    #         videos.append(s)

    #     for _video in videos:
    #         image_file = sorted(glob.glob(os.path.join(
    #             cfg_youtubevos.PATH.SEQUENCES_TRAIN, _video, '*.jpg')))
    #         mask_file = sorted(glob.glob(os.path.join(
    #             cfg_youtubevos.PATH.ANNOTATIONS_TRAIN, _video, '*.png')))
    #         flow_file = sorted(glob.glob(os.path.join(
    #             cfg_youtubevos.PATH.FLOW, _video, '*.png')))
    #         contour_file = sorted(glob.glob(os.path.join(
    #             cfg_youtubevos.PATH.ANNOTATIONS_TRAIN_CTR, _video, '*.png')))
    #         hed_file = sorted(glob.glob(os.path.join(
    #             cfg_youtubevos.PATH.HED, _video, '*.jpg')))

    #         self.image_files.extend(print_list_davis(image_file))
    #         self.mask_files.extend(print_list_davis(mask_file))
    #         self.flow_files.extend(flow_file)
    #         self.contour_files.extend(print_list_davis(contour_file))
    #         self.hed_files.extend(print_list_davis(hed_file))

    #     print('images: ', len(self.image_files))
    #     print('masks: ', len(self.mask_files))
    #     print('heds: ', len(self.hed_files))
    #     print('flows: ', len(self.flow_files))
    #     print('contours: ', len(self.contour_files))

    #     assert(len(self.image_files) == len(self.mask_files) ==
    #            len(self.flow_files) == len(self.contour_files) ==
    #            len(self.hed_files))

    def load_davis(self, args):

        self._db_sequences = list(db_read_sequences_davis(args.year, self._phase))

        # Check lmdb existance. If not proceed with standard dataloader.
        lmdb_env_seq_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_seq')
        lmdb_env_annot_dir = osp.join(cfg_davis.PATH.DATA, 'lmdb_annot')

        if osp.isdir(lmdb_env_seq_dir) and osp.isdir(lmdb_env_annot_dir):
            lmdb_env_seq = lmdb.open(lmdb_env_seq_dir)
            lmdb_env_annot = lmdb.open(lmdb_env_annot_dir)
        else:
            lmdb_env_seq = None
            lmdb_env_annot = None
            print('LMDB not found. This could affect the data loading time.'
                  ' It is recommended to use LMDB.')

        self.sequences = [Sequence(self._phase, s.name, lmdb_env=lmdb_env_seq)
                          for s in self._db_sequences] # 视频名称
        self._db_sequences = db_read_sequences_davis(args.year, self._phase)

        # Load annotations
        # self.annotations = [Annotation(
        #     self._phase, s.name, self._single_object, lmdb_env=lmdb_env_annot)
        #     for s in self._db_sequences]
        # self._db_sequences = db_read_sequences_davis(args.year, self._phase)

    def load_data(self, args):
        # Load Videos
        videos = []
        videos =os.listdir(cfg_davis.PATH.DATA)
        for seq, s in zip(self.sequences, self._db_sequences):
            if s['set'] == self._phase:
                videos.append(s['name'])

        #以视频为单位载入
        for _video in videos:
            image_file = sorted(glob.glob(os.path.join(
                cfg.PATH.data,self._phase,cfg.SEQUENCES, _video, '*.jpg'))+glob.glob(os.path.join(cfg.PATH.data,self._phase,cfg.SEQUENCES, _video, '*.png')))
            mask_file = sorted(glob.glob(os.path.join(
                cfg.PATH.data,self._phase,cfg.ANNOTATIONS, _video, '*.png')))
            flow_file = sorted(glob.glob(os.path.join(
                cfg.PATH.data,self._phase,cfg.Flow, _video, '*.png')))
           
            self.image_files.extend(print_list_davis(image_file))
            self.mask_files.extend(print_list_davis(mask_file))
            self.flow_files.extend(flow_file)


        print('images: ', len(self.image_files))
        print('masks: ', len(self.mask_files))
        print('flows: ', len(self.flow_files))

        # assert(len(self.image_files) == len(self.mask_files) ==
        #        len(self.flow_files) == len(self.contour_files) ==
        #        len(self.hed_files))

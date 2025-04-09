# 三分支，均为视频帧
import os
import cv2
import glob
import lmdb
import numpy as np
from PIL import Image
import os.path as osp
from skimage.transform import resize as imresize

from torch.utils import data
import torch
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
    imagefiles = []
    for i in range(0,len(imagefile)-2):
        sub_list=imagefile[i:i+3]
        imagefiles.append(sub_list)
   
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
        self.contour_files = []


        self.load_data(args)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_file1 = self.image_files[index][0]
        image_file2 = self.image_files[index][2]
        image_file3 = self.image_files[index][1] # mid frame
        
        mask_file1 = self.mask_files[index][0]
        mask_file2 = self.mask_files[index][2]
        

        image1 = cv2.imread(image_file1)  # 使用cv2读取图像
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB模式
        image2 = cv2.imread(image_file2)  # 使用cv2读取图像
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image3 = cv2.imread(image_file3)  # 使用cv2读取图像
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

        mask1 = cv2.imread(mask_file1, 0)
        mask1[mask1 > 0] = 255
        mask2 = cv2.imread(mask_file2, 0)
        mask2[mask2 > 0] = 255



        if self.input_size is not None:
            image1 = imresize(image1, self.input_size) #(height, width)
            mask1 = imresize(mask1, self.input_size,order=0)

            image2 = imresize(image2, self.input_size)
            mask2 = imresize(mask2, self.input_size, order=0)

            image3 = imresize(image3, self.input_size)
        sample = {'image1': image1, 'image2': image2, 'image3':image3 ,
                  'mask1': mask1, 'mask2': mask2
                  }

        if self.augment_transform is not None:
            sample = self.augment_transform(sample)

        image1, image2, image3, mask1, mask2=\
            sample['image1'], sample['image2'], sample['image3'],\
            sample['mask1'], sample['mask2']

        if self.transform is not None:
            image1 = self.transform(image1).to(torch.float32)
            image2 = self.transform(image2).to(torch.float32)
            image3 = self.transform(image3).to(torch.float32)

        if self.target_transform is not None:
            mask1 = mask1[:, :, np.newaxis]
            mask2 = mask2[:, :, np.newaxis]

            mask1 = self.target_transform(mask1).to(torch.float32)
            mask2 = self.target_transform(mask2).to(torch.float32)


        return image1, image2, image3, mask1, mask2


    def load_data(self, args):

        # Load Videos
        videos = []
        videos=db_video_list(self._phase)

        #以视频为单位载入
        for _video in videos:
            image_file = sorted(glob.glob(os.path.join(
                cfg.PATH.DATA,self._phase,cfg.SEQUENCES, _video, '*.jpg'))+glob.glob(os.path.join(cfg.PATH.DATA,self._phase,cfg.SEQUENCES, _video, '*.png')))
            mask_file = sorted(glob.glob(os.path.join(
                cfg.PATH.DATA,self._phase,cfg.ANNOTATIONS, _video, '*.png')))

            assert(len(image_file) == len(mask_file)  )

            self.image_files.extend(print_list_davis(image_file))
            self.mask_files.extend(print_list_davis(mask_file))


        print('images: ', len(self.image_files))
        print('masks: ', len(self.mask_files))

        assert(len(self.image_files) == len(self.mask_files) )

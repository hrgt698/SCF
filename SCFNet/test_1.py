# 保存mask
import os
import glob

import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm
from skimage.transform import resize as imresize
from torchvision import transforms
from libs.utils.utils import check_parallel
from libs.utils.utils import load_checkpoint_epoch
from libs.utils.eval import db_eval_iou,db_eval_F1,db_eval_MCC

from libs.model.HCPN import EncoderNet, DecoderNet


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


#4帧一组
def print_data_list(imagefile):
    imagefiles = []
    left=len(imagefile)%4
    for i in range(0,len(imagefile)-left,4):
        sub_list1=imagefile[i:i+3]
        sub_list2=imagefile[i+1:i+4]
        imagefiles.append(sub_list1)
        imagefiles.append(sub_list2)
   
    return imagefiles


def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).\
                                  long().cuda())
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).\
                                  long())

def test():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr = transforms.ToTensor()
    image_transforms = transforms.Compose([tr, normalize])
    
    encoder_dict, decoder_dict = load_checkpoint_epoch(model_path, epoch, True, False)
    encoder = EncoderNet()
    decoder = DecoderNet()
    encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)

    encoder.cuda()
    decoder.cuda()

    encoder.train(False)
    decoder.train(False)

    iou_all=[]
    F1_all=[]
    MCC_all=[]
    for video in os.listdir(video_dir):

        im_dir = os.path.join(video_dir, video)
        mk_dir = os.path.join(mask_dir, video)

        imagefile = sorted(glob.glob(os.path.join(im_dir, '*.jpg'))+glob.glob(os.path.join(im_dir, '*.png')))
        imagefiles = []
        imagefiles.extend(print_data_list(imagefile))
        
        maskfile = sorted(glob.glob(os.path.join(mk_dir, '*.jpg'))+glob.glob(os.path.join(mk_dir, '*.png')))
        maskfiles = []
        maskfiles.extend(print_data_list(maskfile))

        better_mask1=0
        better_mask2=0
        iou=[]
        F1=[]
        MCC=[]
       # flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))

        with torch.no_grad():
            for index in range(0,min(len(imagefiles),len(maskfiles))):
                
                image1 = cv2.imread(imagefiles[index][0])  # 使用cv2读取图像
                im1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB模式
                image2 = cv2.imread(imagefiles[index][2])  # 使用cv2读取图像
                im2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                mid = cv2.imread(imagefiles[index][1])  # 使用cv2读取图像
                mid= cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)
                
                ori_shape = im1.shape[:2] #[height,width]
                
                mask1 = cv2.imread(maskfiles[index][0], 0).astype(float)/255.0
                mask1[mask1 > 0] = 1.0
                mask2 = cv2.imread(maskfiles[index][2], 0).astype(float)/255.0
                mask2[mask2 > 0] = 1.0

                # only test forged frame
                if np.max(mask1)==0 or np.max(mask2)==0:
                    continue
                
                if img_size is not None:
                    im1 = imresize(im1, img_size)
                    im2 = imresize(im2, img_size)
                    mid = imresize(mid, img_size)


                im1 = image_transforms(im1).to(torch.float32)
                im2 = image_transforms(im2).to(torch.float32)
                mid = image_transforms(mid).to(torch.float32)
                
                

                im1 = im1.unsqueeze(0)
                im2 = im2.unsqueeze(0)
                mid = mid.unsqueeze(0)

                im1, im2, mid = im1.cuda(), im2.cuda(), mid.cuda()

                h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2 = encoder(im1, im2, mid)
                mask_1, c_1, mask_2, c_2 = decoder(h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2)

                if use_flip:
                    im1_flip = flip(im1, 3)
                    im2_flip = flip(im2, 3)
                    mid_flip = flip(mid, 3)
                    h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2 = encoder(im1_flip, im2_flip, mid_flip)
                    mask_flip_1, c_flip_1, mask_flip_2, c_flip_2 = \
                        decoder(h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2)

                    mask_flip_1 = flip(mask_flip_1, 3)
                    mask_flip_2 = flip(mask_flip_2, 3)
                    mask_1 = (mask_1 + mask_flip_1) / 2.0
                    mask_2 = (mask_2 + mask_flip_2) / 2.0

                mask_1 = mask_1[0, 0, :, :].cpu().detach().numpy()
                mask_2 = mask_2[0, 0, :, :].cpu().detach().numpy()
                mask_1 = imresize(mask_1,ori_shape)
                mask_2 = imresize(mask_2,ori_shape)
                
                if mask1.shape!= ori_shape:
                    mask1 = imresize(mask1,ori_shape)
                    mask2 = imresize(mask2,ori_shape)  
                # mask_1_np = mask_1.cpu().detach().numpy() * 255
                # mask_1 = imresize(mask_1_np,ori_shape)
                # mask_1 = mask_1.astype(np.uint8)
                # mask_2_np = mask_2.cpu().detach().numpy() * 255              
                # mask_2 = imresize(mask_2_np,ori_shape) 
                # mask_2 = mask_2.astype(np.uint8)      
                
                ### save pred_mask
                save_mask_folder = os.path.join(mask_result_dir,video)#'{}/{}_epoch{}/{}'.format(davis_mask_dir, model_name, epoch, video)
                if not os.path.exists(save_mask_folder):
                    os.makedirs(save_mask_folder)

                save_file1 = os.path.join(save_mask_folder,
                                          os.path.basename(imagefiles[index][0])[:-4] + '.png')
                save_file2 = os.path.join(save_mask_folder,
                                           os.path.basename(imagefiles[index][2])[:-4] + '.png')
                save_mask_1 = (mask_1*255).astype(np.uint8)
                save_mask_2 = (mask_2*255).astype(np.uint8)
                cv2.imwrite(save_file1,save_mask_1)
                cv2.imwrite(save_file2,save_mask_2)
                
                # 分类阈值
                # mask1=mask1>0.5
                # mask_1=mask_1>0.5
                
                #iou
            #     iou_1 = db_eval_iou(mask1, mask_1)
            #     #iou_2 = db_eval_iou(mask2, mask_2)
            #     #F1
            #     f1_1=db_eval_F1(mask1,mask_1)
            #    # f1_2=db_eval_F1(mask2,mask_2)
                
            #     #MCC
            #     MCC_1=db_eval_MCC(mask1,mask_1)
            #     #MCC_2=
                
            #     iou.append(iou_1)
            #     F1.append(f1_1)
            #     MCC.append(MCC_1)
                

                # if iou_1+f1_1 >= better_mask1:
                #     #cv2.imwrite(save_file1,mask_1*255)
                #     better_mask1=iou_1+f1_1
                #     if len(iou)==0:
                #         iou.append(iou_1)
                #         F1.append(f1_1)
                #     else:
                #         iou[-1]=iou_1
                #         F1[-1]=f1_1

                # #cv2.imwrite(save_file2,mask_2*255)
                # iou.append(iou_2)
                # F1.append(f1_2)
    #         print(f"{video} iou: {np.mean(iou)},F1: {np.mean(F1)},MCC: {np.mean(MCC)} ", )
    #         iou_all+=iou
    #         F1_all+=F1
    #         MCC_all+=MCC
        print(video)
    # print('iou_mean:   ',np.mean(iou_all))
    # print('F1_mean:   ',np.mean(F1_all))
    # print('MCC_mean:   ',np.mean(MCC_all))
    
def randomTest():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tr = transforms.ToTensor()
    image_transforms = transforms.Compose([tr, normalize])
    
    encoder = EncoderNet()
    decoder = DecoderNet()  
    encoder.train(False)
    decoder.train(False)
    im1 = np.random.random((512, 512, 3)) #(width, height, channel)
    im2 = np.random.random((512, 512, 3))
    mid = np.random.random((512, 512, 3))
    
    im1 = image_transforms(im1).to(torch.float32)
    im2 = image_transforms(im2).to(torch.float32)
    mid = image_transforms(mid).to(torch.float32)
    
    im1 = im1.unsqueeze(0)
    im2 = im2.unsqueeze(0)
    mid = mid.unsqueeze(0)
    
    h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2 = encoder(im1, im2, mid)
    mask_1, c_1, mask_2, c_2 = decoder(h5_1, h4_1, h3_1, h2_1, h5_2, h4_2, h3_2, h2_2)
     
if __name__ == '__main__':

    #os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    img_size = (512, 512)
    use_flip = False

    model_path = r'/users/u202220081200014/video-inpainting/video_forgery/HCPN_MLP/results/HCPN_MLP_dsexpand_lr_3'
    epoch = 10

    #test_set = r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/sub_dataset/val/videos'
    video_dir = r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VTD/VTD_splicing/videos'
    #davis_flow_dir = '/Your_PathDAVIS/DAVIS/davis-flow'

    mask_dir = r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VTD/VTD_splicing/masks'
    mask_result_dir= r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VTD/VTD_splicing/SCFN_pred'

    test()
    #randomTest()
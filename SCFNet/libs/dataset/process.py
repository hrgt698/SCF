import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
def get_contour_mask():
    mask_path=r'E:\VideoDataset\Video_with_mask\all\video-splice\add\train\masks'
    target_path=r'E:\VideoDataset\Video_with_mask\all\video-splice\train\masks_ctr'
    
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for video in os.listdir(mask_path):
        masks=os.listdir(os.path.join(mask_path,video))
        
        if not os.path.exists(os.path.join(target_path,video)):
            os.mkdir(os.path.join(target_path,video))
        
        for mask in masks:
            save_path=os.path.join(target_path,video,mask)
            if os.path.exists(save_path):
                continue
            else:
                # 读取图像
                image = cv2.imread(os.path.join(mask_path,video,mask), 0)  # 读取为灰度图像
                
                _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contour_mask = np.zeros_like(image)
                cv2.drawContours(contour_mask, contours, -1, (255,255,255), 2)

                cv2.imwrite(save_path,contour_mask)
        print(video)
  
#get_contour_mask()

def move_path():
    path=r'E:\VideoDataset\Video_with_mask\all\video-splice\add\val\videos'
    
    for video in os.listdir(path):
        shutil.move(os.path.join(r'E:\VideoDataset\Video_with_mask\all\VCMS\masks',video),r'E:\VideoDataset\Video_with_mask\all\video-splice\add\val\masks')
        
#move_path()
# 

def plot():
    train_loss=[]
    train_iou=[]
    
    val_loss=[]
    val_iou=[]
    
    with open('/users/u202220081200014/video-inpainting/video_forgery/HCPN_VOS/log_3rgb.txt','r') as file:
        for line in file:
            if 'Epoch:' in line:
                data=line.strip().split('\t')
                loss_index = next((i for i, s in enumerate(data) if 'Loss:' in s), None)
                iou_index=next((i for i, s in enumerate(data) if 'IOU:' in s), None)

                loss=float(data[loss_index].split(' ')[-1])
                iou=float(data[iou_index].split(' ')[-1])
                
                if '/3393]' in line:
                    train_loss.append(loss)
                    train_iou.append(iou)
                elif '/364]' in line:
                    val_loss.append(loss)
                    val_iou.append(iou)  
                
        plt.plot(train_loss,label='train loss')
        plt.show()
        plt.plot(train_iou,label='train iou')
        plt.plot(val_loss,label='val loss')
        plt.plot(val_iou,label='val iou')
        
plot()
    
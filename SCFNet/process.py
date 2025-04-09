import cv2
import subprocess
import os
import random
import glob
import numpy as np
import shutil
random.seed(5)
from skimage.transform import resize as imresize
from libs.utils.eval import db_eval_iou,db_eval_F1,db_eval_MCC
# 将一个文件夹中的所有帧编码成视频
# 帧序列的命名必须是000.png, 001.png, 002.png......
def encode_video(folder_path, output_dir):
    # 获取文件夹名字作为视频名字
    video_name = os.path.basename(folder_path) + ".mp4"
    # 设置输出路径
    output_path = os.path.join(output_dir, video_name)
    
    QP = random.choice(['15', '23', '30'])
    
    input_files = sorted(os.listdir(folder_path))
    
    frame_name=input_files[0].split('.',1)[0]
    # subprocess.call([
    # "ffmpeg", "-framerate", "30", "-i", input_sequence, "-c:v", "libx264", "-crf", "30",
    # "-pix_fmt", "yuv420p", output_path
    # ])
   
    ffmpeg_cmd = [
        'ffmpeg',                       # FFmpeg命令
        '-framerate', "30",   # 输入图像序列的帧率
        '-start_number', str(frame_name), 
        '-i',os.path.join(folder_path,"%05d.png") ,  # 输入图像序列的文件名格式
        '-c:v', 'libx264',              # 使用H.264视频编码器
        '-crf', QP,           # 视频压缩质量
        '-pix_fmt', 'yuv420p',          # 设置像素格式
        output_path                   # 输出视频文件名
    ]

    subprocess.run(ffmpeg_cmd)
    # #调用ffmpeg进行编码
    # subprocess.call(
    #     ["ffmpeg", "-framerate", "30", "-i", os.path.join(folder_path, "*.png"), "-c:v", "libx264", "-crf", str(QP),
    #      "-pix_fmt", "yuv420p", output_path])
    return output_path,int(frame_name)

def extract_frames(video_path, output_dir,num):
    """
    Extracts all frames from a video and saves them as png files.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Path to the output directory.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the video basename
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Read and save all frames
    i = num
    while True:
        # Read the next frame
        ret, frame = cap.read()

        # Stop if there are no more frames
        if not ret:
            break

        # Save the frame as a png file
        frame_path = os.path.join(output_dir, f"{i:05d}.png")
        cv2.imwrite(frame_path, frame)
        
        i += 1

    # Release the video file
    cap.release()
  
    
def make_even(number):
    if number % 2 !=0:
        return number-1
    return number

#adjust resolution
def adjust():
    path='/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSplicingDataset2.0/a_train_reencode/videos'
    mask_path='/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSplicingDataset2.0/train/videos'
    for video in os.listdir(path):
        folder_path=os.path.join(path,video)
        for frame in os.listdir(folder_path):
            image=cv2.imread(os.path.join(folder_path,frame))
            mask=cv2.imread(os.path.join(folder_path.replace('a_train_reencode','train').replace('videos','masks'),frame))
            height,width=(image.shape[:2])
            new_height=make_even(height)
            new_width=make_even(width)
            if new_height!= height or new_width!=width:
                crop_image=image[:new_height,:new_width]
                crop_mask=mask[:new_height,:new_width]
                image_save=os.path.join(folder_path,frame)
                mask_save=os.path.join(folder_path.replace('a_train_reencode','train').replace('videos','masks'),frame)
                cv2.imwrite(image_save,crop_image)
                cv2.imwrite(mask_save,crop_mask)
            else:
                break
            # if height>width:
            #     image_save=os.path.join(folder_path,frame)
            #     mask_save=os.path.join(folder_path.replace('videos','masks'),frame)
            #     image_adj=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            #     mask_adj=cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            #     cv2.imwrite(image_save,image_adj)
            #     cv2.imwrite(mask_save,mask_adj)
#adjust()           
                
    
def reencode_videos():
    video_path=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSplicingDataset2.0/a_train_reencode/videos'
    output=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSplicingDataset2.0/a_train_reencode/reencoded'
    random.seed(5)
    videos=os.listdir(video_path)
    random.shuffle(videos)
    for video in videos[:]:
        folder_path=os.path.join(video_path,video)
        temp=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSplicingDataset2.0/'
        output_path=os.path.join(output,video)
        temp_video,num=encode_video(folder_path, temp)
        extract_frames(temp_video, output_path,num)
        os.remove(temp_video)
         
#reencode_videos()

def image_rename():
    all_path=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/splicing-Avino2017/complete_object/complete_object/masks'
    target_path=r'D:\video-inpainting-dataset\to_inpaint\E2FGVI'

    for video in os.listdir(all_path):
        path=os.path.join(all_path,video)
        num=0  
        for frame in os.listdir(path):
            # l=str(num).zfill(5)
            # r='png'
            # num+=1
            new=frame.split('_')[-1]
            os.rename(os.path.join(path,frame),os.path.join(path,new))  
#image_rename()
#adjust(r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/train/videos')       
#reencode_videos()

def sta():
    with open('/users/u202220081200014/video-inpainting/video_forgery/HCPN_VOS/log_reencode_val.txt','r') as file:
        iou_all=[]
        f1_all=[]
        for line in file:
            data=line.strip().split('\t')
           
            if 'iou' in line:
                index=next((i for i, s in enumerate(data) if 'iou: ' in s), None)
                iou=data[index].rsplit(' ',1)[-1]
                if iou != 'nan':
                    iou_all.append(float(iou))
            else :
                index=next((i for i, s in enumerate(data) if 'F1: ' in s), None)
                f1=data[index].rsplit(' ',1)[-1]
                if f1 != 'nan':
                    f1_all.append(float(f1))
        print(np.mean(iou_all))
        print(np.mean(f1_all))
#sta()

def move_compressed():
    path=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/sub_dataset/train/videos'
    uncom=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/train/videos'
    uncom_tar=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/sub_dataset/train/uncom_videos'
    for video in os.listdir(path):
        shutil.move(os.path.join(uncom,video),uncom_tar)
        shutil.move(os.path.join(path,video),uncom)

def move_empty():
    path=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/val/videos'
    uncom=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/train/videos'
    uncom_tar=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/video-splice/sub_dataset/train/uncom_videos'
    for video in os.listdir(path):
        frames=os.listdir(os.path.join(path,video))
        if len(frames)==0:
            #shutil.rmtree(os.path.join(path,video))
            print(video)
#move_empty()

def threshold_():
    pred_path=r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/HEVC/SCFNet_VSD2.0_pred' #VTD/VTD_splicing
    print('HVTD')
    gt_path='/groups/imageNSFC/home/u202220081200014/share/videoDataset/HEVC/masks'#r'/groups/imageNSFC/home/u202220081200014/share/videoDataset/VideoSham/add/groundtruth'
    thresholds=np.arange(0,1,0.1)
    IOU=np.zeros(len(thresholds))
    F1=np.zeros(len(thresholds))
    MCC=np.zeros(len(thresholds))
    num=0
    for video in os.listdir(pred_path):
        for frame in os.listdir(os.path.join(pred_path,video)):
            num+=1
            pred=os.path.join(pred_path,video,frame)
            gt=os.path.join(gt_path,video,frame)#.replace('forged','mask'))
            
            mask1 = cv2.imread(pred, 0).astype(float)
            mask2 = cv2.imread(gt, 0).astype(float)
            
          
            if mask1.shape!= mask2.shape:
                mask1 = imresize(mask1,mask2.shape)
            
            mask1/=255.0
            mask2/=255.0 
            mask2[mask2 > 0] = 1.0
            
            for i,threshold in enumerate(thresholds):
                # mask_1=mask1>threshold
                # mask_2=mask2>threshold
                
                iou=db_eval_iou(mask2,mask1,threshold)
                f1=db_eval_F1(mask2,mask1,threshold)
                mcc=db_eval_MCC(mask2,mask1,threshold)
                
                IOU[i]+=iou
                F1[i]+=f1
                MCC[i]+=mcc
        
    IOU/=num
    F1/=num
    MCC/=num
    
    print(IOU)
    print(F1)
    print(MCC)

threshold_()
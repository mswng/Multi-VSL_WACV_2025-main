import os
import numpy as np
import torch
from torch.utils.data import  Dataset
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
import cv2
import torchvision
from dataset.utils import crop_hand
import json
from PIL import Image
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import glob
import time
from decord import VideoReader
import threading
from utils.video_augmentation import *



class ThreeViewsData(Dataset):
    def __init__(self, base_url,split,dataset_cfg,**kwargs):
        
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
       
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.pose_transform  = Compose(DeleteFlowKeypoints(list(range(112, 113))),
                                        DeleteFlowKeypoints(list(range(11, 92))),
                                        DeleteFlowKeypoints(list(range(0, 5))),
                                        ToFloatTensor())
        self.transform = self.build_transform(split)
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomHorizontalFlip(), 
                                RandomRotate(p=0.3),
                                RandomShear(0.3,0.3,p = 0.3),
                                Salt( p = 0.3),
                                GaussianBlur( sigma=1,p = 0.3),
                                ColorJitter(0.5, 0.5, 0.5,p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames,width,height
    def read_one_view(self,name,selected_index,width,height):
       
        clip = []
       
        missing_wrists_left = []
        missing_wrists_right = []


        # path = '/mnt/disk3/anhnct/Hand-Sign-Recognition/data/VN_SIGN/videos.zip'+'@'+f'videos/{name}'
        # video_byte = ZipReader.read(path)
        # frames = _load_frame_nums_to_4darray(video_byte,selected_index) #T,H,W,3

       
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'Yolo_dataset/Blur_video/{name}'   
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        for frame_index,frame in zip(selected_index,frames):
            if self.data_cfg['crop_two_hand']:
                
                kp_path = os.path.join(self.base_url,'poses',name.replace(".mp4",""),
                                    name.replace(".mp4","") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['pose_threshold_02']) # 26,3
                    x = 320*keypoints[:,0]/width
                    y = 256*keypoints[:,1]/height
                   
                keypoints = np.stack((x, y), axis=0)
               
           
            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            
           
              
            assert crops is not None
            clip.append(crops)
            
        
        if  len(missing_wrists_left) > 0 or len(missing_wrists_right) > 0:
            print("Missing left ",len(missing_wrists_left),"Missing right ",len(missing_wrists_right),name)
            for clip_index in range(len(clip)):
                if clip_index in missing_wrists_left:
                # Find temporally closest not missing frame for left wrist
                    replacement_index = -1
                    distance = np.inf
                    for ci in range(len(clip)):
                        if ci not in missing_wrists_left:
                            dist = abs(ci - clip_index)
                            if dist < distance:
                                distance = dist
                                replacement_index = ci
                    if replacement_index != -1:
                        clip[clip_index][0] = clip[replacement_index][0]
                        
                       
                # Same for right crop
                if clip_index in missing_wrists_right:
                    # Find temporally closest not missing frame for right wrist
                    replacement_index = -1
                    distance = np.inf
                    for ci in range(len(clip)):
                        if ci not in missing_wrists_right:
                            dist = abs(ci - clip_index)
                            if dist < distance:
                                distance = dist
                                replacement_index = ci
                    if replacement_index != -1:
                        clip[clip_index][1] = clip[replacement_index][1]
                       
       

        clip = torch.stack(clip,dim = 0)
        
        return clip

    def read_videos(self,center,left,right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        # 
        vlen1,c_width,c_height = self.count_frames(os.path.join('Yolo_dataset/Blur_video',center))
        vlen2,l_width,l_height = self.count_frames(os.path.join('Yolo_dataset/Blur_video',left))
        vlen3,r_width,r_height = self.count_frames(os.path.join('Yolo_dataset/Blur_video',right))

        # Initialize output videos = 0
        center_video = torch.zeros((self.data_cfg['num_output_frames'], 3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
        left_video = torch.zeros((self.data_cfg['num_output_frames'], 3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
        right_video = torch.zeros((self.data_cfg['num_output_frames'], 3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))

        # print("Start================================")
        # print("Size of right:", right_video.size())
        # print("Size of center:", center_video.size())
        # print("Size of left:", left_video.size())
        # print("=================================")

        min_vlen = min(vlen1,min(vlen2,vlen3))
        max_vlen = max(vlen1,max(vlen2,vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center,left,right)
                selected_index  = pad_index(selected_index,pad).tolist()

            if np.random.random() > self.data_cfg['missing_rate']:
                center_video = self.read_one_view(center,selected_index,width=c_width,height=c_height)
            
            if np.random.random() > self.data_cfg['missing_rate']:
                left_video = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            if np.random.random() > self.data_cfg['missing_rate']:
                right_video = self.read_one_view(right,selected_index,width=r_width,height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center)
                selected_index  = pad_index(selected_index,pad).tolist()

            if np.random.random() > self.data_cfg['missing_rate']:
                center_video = self.read_one_view(center,selected_index,width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",left)
                selected_index  = pad_index(selected_index,pad).tolist()
            
            if np.random.random() > self.data_cfg['missing_rate']:
                left_video = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            selected_index, pad = get_selected_indexs(vlen3-2,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",right)
                selected_index  = pad_index(selected_index,pad).tolist()

            if np.random.random() > self.data_cfg['missing_rate']:
                right_video = self.read_one_view(right,selected_index,width=r_width,height=r_height)

       
        # print("Size of right:", right_video.size())
        # print("Size of center:", center_video.size())
        # print("Size of left:", left_video.size())
        # # In ra kiểu dữ liệu của tensor
        # print("Data type of center:", right_video.dtype)

        return center_video,left_video,right_video
        # return 1


    def __getitem__(self, idx):
        self.transform.randomize_parameters()
       
        center,left,right,label = self.train_labels.iloc[idx].values
       
        center_video,left_video,right_video = self.read_videos(center,left,right)
        
        return center_video,left_video,right_video,torch.tensor(label)
     
    
    def __len__(self):
        return len(self.train_labels)
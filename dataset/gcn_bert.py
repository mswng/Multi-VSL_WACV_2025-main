import json
import math
import os
from argparse import ArgumentParser
from utils.video_augmentation import *
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import pickle
import cv2
from dataset.videoLoader import load_batch_video,get_selected_indexs,pad_array,pad_index
from dataset.utils import crop_hand




class GCN_BERT(Dataset):
    def __init__(self, base_url,split,transform,dataset_cfg):
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
    
        print(split,len(self.train_labels))
        self.transform = transform
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

    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames,width,height
    
    def read_keypoints(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'   
            vlen,width,height = self.count_frames(path)

        else:
            video_file = os.path.join(self.base_url,f'mp4/{self.split}',f"{name}_color.mp4")
        
            # load video frames
            frames, _, _ = torchvision.io.read_video(video_file,
                                                    pts_unit='sec')
            vlen = frames.shape[0]
       
        selected_index, pad = get_selected_indexs(vlen-2,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()

        poses = []

       
       
        for frame_index in selected_index:
            if self.data_cfg['dataset_name'] == "VN_SIGN":
               
                kp_path = os.path.join(self.base_url,'wholebody',name.replace(".mp4",""),
                                    name.replace(".mp4","") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['wholebody_threshold_02']) # 26,3
                    x = keypoints[:,0]
                    y = keypoints[:,1]
                keypoints = np.stack((x, y), axis=0)
               
            else:
                video_file = os.path.join(self.base_url,f'mp4/{self.split}',f"{name}_color.mp4")
                frame = frames[frame_index]
                # clip.append(self.transform(frame.numpy()))
                kp_path = os.path.join(
                                # self.root_path.replace('mp4', 'kp'), self.job_path,   
                               video_file.replace('mp4', 'kp'), '{}_{:012d}_keypoints.json'.format(
                                video_file.split('/')[-1].replace('.mp4', ''), frame_index))
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['people'][0]['keypoints'])
                    x = keypoints[0::3]
                    y = keypoints[1::3]
                    keypoints = np.stack((x, y), axis=0)
            
            pose = self.pose_transform(keypoints.T) 
            # normalize
            pose[:,0]/=width
            pose[:,1]/=height
            
            poses.append(pose)
        
        poses = torch.tensor(np.stack(poses,axis=0)) # T,N,C
       
        return poses 


    def __getitem__(self, idx):
        data = self.train_labels.iloc[idx].values
        name,label = data[0],data[1]
        if self.data_name == 'AUTSL':
            text = self.labels.iloc[label].values[-1]
            # remove _
            text = text.replace("_"," ")
            prompt = f"a photo of {text}"
            return self.read_video(name),torch.tensor(label),prompt
        elif self.data_name == 'VN_SIGN':
            poses = self.read_keypoints(name)
            return poses,torch.tensor(label)
    
    def __len__(self):
        return len(self.train_labels)
    


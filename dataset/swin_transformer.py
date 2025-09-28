import os
import numpy as np
import torch
from torch.utils.data import  Dataset
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
from decord import VideoReader
from utils.video_augmentation import *

class SwinTransformer(Dataset):
    def __init__(self, base_url,split,dataset_cfg,train_labels = None,**kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
        
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                # Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']), -> no v
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
    def read_videos(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        clip = []
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            #path = f'data/Blur_video/{name}'   
            path = f'Yolo_dataset/Blur_video/{name}'   
        elif self.data_cfg['dataset_name'] == "AUTSL":
            path = f'{self.base_url}/{self.split}/{name}'   

        vr = VideoReader(path,width=320, height=256)
        vlen = len(vr)
        selected_index, pad = get_selected_indexs(vlen,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        for frame in frames:
            clip.append(self.transform(frame))
            
        clip = torch.stack(clip,dim = 0)
        return clip

    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name,label = data[0],data[1]

        clip = self.read_videos(name)
        
        return clip,torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)

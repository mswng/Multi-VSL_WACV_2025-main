import os
import numpy as np
import torch
from torch.utils.data import  Dataset
import pandas as pd
from dataset.videoLoader import get_selected_indexs,pad_index
from decord import VideoReader
from utils.video_augmentation import *
from dataset.utils import crop_hand
import json



class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

class VideoMAE(Dataset):
    def __init__(self, base_url,split,dataset_cfg):
        if dataset_cfg['dataset_name'] == "VN_SIGN":
            print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
            self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
    
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
        self.gen_mask = TubeMaskingGenerator((self.data_cfg['num_output_frames']//2,14,14),0.9)
        
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
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
    
    def read_videos(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
       
        path = f'{self.base_url}/videos/{name}'   
        vlen,width,height = self.count_frames(path)
       
        selected_index, pad = get_selected_indexs(vlen,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        for frame,frame_index in zip(frames,selected_index):
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
            clip.append(crops)
            
        clip = torch.stack(clip,dim = 0)
        return clip


    def __getitem__(self, idx):
        self.transform.randomize_parameters()
        data = self.train_labels.iloc[idx].values
        name,label = data[0],data[1]

        clip = self.read_videos(name)
        mask = torch.tensor(self.gen_mask())
        
        return clip,mask,torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)




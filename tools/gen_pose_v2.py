from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import time
import json
import os
import pandas as pd
from tqdm.auto import tqdm
"""
# LeftHand : [92,111]
# RightHand: [113,132] include 10
# Pose: [0,10] include 10

# After append neck and headtop at first:

# LeftHand : [94,113] include 113
# RightHand: [115,134] include 134
# Pose: [0,10] include 10
# Neck: 17 HeadTop: 18
"""

def read_and_write_video(input_video_path, output_video_path,keypoints = None):
    assert keypoints is not None
    """
    Đọc video từ đường dẫn đầu vào và ghi lại video vào đường dẫn đầu ra.
    
    Args:
    - input_video_path (str): Đường dẫn đến video đầu vào.
    - output_video_path (str): Đường dẫn đến video đầu ra.
    """
    # Mở video đầu vào
    cap = cv2.VideoCapture(input_video_path)
    
    # Lấy thông số của video đầu vào
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tạo video writer để ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    cnt = 0
    keypoints = np.array(keypoints).reshape(-1,26,3)
    # Đọc từng frame từ video đầu vào và ghi vào video đầu ra
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        wholebody = keypoints[cnt].tolist()
        # pose
        for kp in wholebody[0:26]:
            cv2.circle(frame, (int(kp[0]),int(kp[1])), 3, (0, 255, 0), -1)
        
        out.write(frame)
        cnt+=1
    # Giải phóng tài nguyên
    cap.release()
    out.release()

def gen_pose(base_url,file_name,pose_detector):
    video_url = os.path.join(base_url,file_name)
    pose_results = pose_detector(video_url)

    kp_folder = video_url.replace("Blur_video",'poses_1_1000').replace('.mp4',"")
    if not os.path.exists(kp_folder):
        os.makedirs(kp_folder,exist_ok=True)
        for idx,pose_result in enumerate(pose_results):
            pose = pose_result['predictions'][0][0]['keypoints']
            prob = pose_result['predictions'][0][0]['keypoint_scores']
            raw_pose = [[value[0],value[1],0] for idx,value in enumerate(pose)]
            pose_threshold_02 = [[value[0],value[1],0] if prob[idx] > 0.2 else [0,0,0] for idx,value in enumerate(pose)]
            dict_data = {
                "raw_pose": raw_pose,
                "pose_threshold_02": pose_threshold_02,
                "prob": prob
            }
            dest = os.path.join(kp_folder,file_name.replace(".mp4","") + '_{:06d}_'.format(idx) + 'keypoints.json')
            
            with open(dest, 'w') as f:
                json.dump(dict_data, f)
    else:
        print('exists')
       


if __name__ == "__main__":

    # CHIA THÀNH NHIỀU LUỒNG CHẠY SONG SONG
    
    full_data = pd.read_csv("data/label_1_1000/labels_1_1000.csv")[:17000]
    #full_data = pd.read_csv("data/label_1_1000/labels_1_1000.csv")[17000:34000]
    #full_data = pd.read_csv("data/label_1_1000/labels_1_1000.csv")[34000:51000]
    #full_data = pd.read_csv("data/label_1_1000/labels_1_1000.csv")[51000:68000]
    #full_data = pd.read_csv("data/label_1_1000/labels_1_1000.csv")[68000:]

    pose_detector = MMPoseInferencer( "rtmpose-m_8xb512-700e_body8-halpe26-256x192")

    print(full_data.shape)
    
    for idx, data in tqdm(full_data.iterrows(), total=full_data.shape[0]):
        gen_pose("Yolo_dataset/Blur_video",data['name'],pose_detector)
    
   
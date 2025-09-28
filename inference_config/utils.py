from collections import defaultdict
import numpy as np
import math
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import torch
from dataset.videoLoader import load_batch_video,get_selected_indexs,pad_array,pad_index

def impute_missing_keypoints(poses):
    """Replace missing keypoints (on the origin) by values from neighbouring frames."""
    # 1. Collect missing keypoints
    missing_keypoints = defaultdict(list)  # frame index -> keypoint indices that are missing
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Missing keypoint at (0, 0)
                missing_keypoints[i].append(kpi)
    # 2. Impute them
    for i in missing_keypoints.keys():
        missing = missing_keypoints[i]
        for kpi in missing:
            # Possible replacements
            candidates = poses[:, kpi] # n_frame,2    
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Replace
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
    # 3. We have imputed as many keypoints as possible with the closest non-missing temporal neighbours
    return poses


def normalize(poses):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    for i in range(poses.shape[0]):
        upper_neck = poses[i, 17]
        head_top = poses[i, 18]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return poses

def compute_pose_flow(poses):
        pose_flow = []
        prev = poses[0]
        for i in range(1, poses.shape[0]):
            next = poses[i]
            flow = calc_pose_flow(prev, next)
            prev = next
            pose_flow.append(flow)
        return np.stack(pose_flow,axis = 0)
    
def calc_pose_flow(prev, next):
        result = np.zeros_like(prev) # 135,2
        for kpi in range(prev.shape[0]):
            if np.count_nonzero(prev[kpi]) == 0 or np.count_nonzero(next[kpi]) == 0:
                result[kpi, 0] = 0.0
                result[kpi, 1] = 0.0
                continue

            ang = math.atan2(next[kpi, 1] - prev[kpi, 1], next[kpi, 0] - prev[kpi, 0])
            mag = np.linalg.norm(next[kpi] - prev[kpi])

            result[kpi, 0] = ang
            result[kpi, 1] = mag

        return result
    
def detect_keypoints(frames,selected_index,
                    wholebody_task_processor,wholebody_input_shape,wholebody_detector,
                    pose_task_processor,pose_input_shape,pose_detector
                    ):
    
    keypoints  = []
    with torch.no_grad():
        for idx in selected_index:
            frame = frames[idx]
            wholebody_ = {}
            pose_ = {}
            wholebody_model_inputs, _ = wholebody_task_processor.create_input(frame.numpy(), wholebody_input_shape)
            wholebody_result = wholebody_detector.test_step(wholebody_model_inputs)
            wholebody_model_inputs, _ = pose_task_processor.create_input(frame.numpy(), pose_input_shape)
            pose_result = pose_detector.test_step(wholebody_model_inputs)
            wholebody_['keypoints'] = wholebody_result[0].pred_instances['keypoints'].reshape(-1,2).tolist()
            wholebody_['keypoint_scores'] = wholebody_result[0].pred_instances['keypoint_scores'].reshape(-1).tolist()
            pose_['keypoints'] = pose_result[0].pred_instances['keypoints'].reshape(-1,2).tolist()
            pose_['keypoint_scores'] = pose_result[0].pred_instances['keypoint_scores'].reshape(-1).tolist()
            
            # .. UpperNeck ,HeadTop ...
            wholebody = wholebody_['keypoints'][:16] + pose_['keypoints'][17:19][::-1] +  wholebody_['keypoints'][16:]
            prob = wholebody_['keypoint_scores'][:16] + pose_['keypoint_scores'][17:19][::-1] +  wholebody_['keypoint_scores'][16:]
            wholebody_threshold_02 = [[value[0],value[1]] if prob[idx] > 0.3 else [0,0] for idx,value in enumerate(wholebody)]
            keypoints.append(wholebody_threshold_02)
    return np.array(keypoints)  

    
def vtn_inference_v2_collate_fn_(batch,frames,data_cfg,transform,temporal_stride,
                                wholebody_task_processor,wholebody_input_shape,wholebody_detector,
                                pose_task_processor,pose_input_shape,pose_detector
                                ):
    index_setting = data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
    clips = []
    poseflow_clips = []
    start_idx = [v[0] for v in batch]
    end_idx = [v[1] for v in batch]
    
    for s_idx,e_idx in zip(start_idx,end_idx):
        clip = []
        poseflow_clip = []
        missing_wrists_left, missing_wrists_right = [], []
        selected_index, pad = get_selected_indexs(e_idx - s_idx + 1,data_cfg['num_output_frames'],False,index_setting,temporal_stride=temporal_stride)
        selected_index = (np.array(selected_index)) + s_idx
        poses = detect_keypoints(frames,selected_index,wholebody_task_processor,wholebody_input_shape,wholebody_detector
                                    ,pose_task_processor,pose_input_shape,pose_detector)
        poses = impute_missing_keypoints(poses)
        pose_flows = compute_pose_flow(poses)
        pose_flows= np.concatenate([np.zeros((1,135, 2)),pose_flows],axis = 0)
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
            pose_flows = pad_array(pose_flows,pad)
            poses = pad_array(poses,pad)
        for pose_index,frame_index in enumerate(selected_index):
            # Let's say the first frame has a pose flow of 0 
            poseflow = None
            frame_index_poseflow = frame_index
            keypoints = poses[pose_index].T
            
            poseflow = pose_flows[pose_index]
            # Normalize the angle between -1 and 1 from -pi and pi
            poseflow[:, 0] /= math.pi
            # Magnitude is already normalized from the pre-processing done before calculating the flow
            
                

            
            frame = frames[frame_index]

            left_wrist_index = 9
            left_elbow_index = 7
            right_wrist_index = 10
            right_elbow_index = 8

            # Crop out both wrists and apply transform
            left_wrist = keypoints[0:2, left_wrist_index]
            left_elbow = keypoints[0:2, left_elbow_index]
        
            left_hand_center = left_wrist + data_cfg['WRIST_DELTA'] * (left_wrist - left_elbow)
            left_hand_center_x = left_hand_center[0]
            left_hand_center_y = left_hand_center[1]
            shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * data_cfg['SHOULDER_DIST_EPSILON']
            left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
            left_hand_xmax = min(frame.size(1), int(left_hand_center_x + shoulder_dist // 2))
            left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
            left_hand_ymax = min(frame.size(0), int(left_hand_center_y + shoulder_dist // 2))
            
            if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                # missing_wrists_left.append(len(clip) + 1) # Check again
                missing_wrists_left.append(len(clip) ) # I tried this and achived 93% on test
                
            else:
                left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
            left_hand_crop = transform(left_hand_crop.numpy())

            right_wrist = keypoints[0:2, right_wrist_index]
            right_elbow = keypoints[0:2, right_elbow_index]
            right_hand_center = right_wrist + data_cfg['WRIST_DELTA'] * (right_wrist - right_elbow)
            right_hand_center_x = right_hand_center[0]
            right_hand_center_y = right_hand_center[1]
            right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
            right_hand_xmax = min(frame.size(1), int(right_hand_center_x + shoulder_dist // 2))
            right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
            right_hand_ymax = min(frame.size(0), int(right_hand_center_y + shoulder_dist // 2))

            if not np.any(right_wrist) or not np.any(
                    right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                right_hand_crop = frame
                # missing_wrists_right.append(len(clip) + 1) # Check again
                missing_wrists_right.append(len(clip)) # I tried this and achived 93% on test
                
            else:
                right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
            right_hand_crop = transform(right_hand_crop.numpy())

            crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

            clip.append(crops)
            # get 0->10, 17->18,25->64
            # mmpose 
            # # LeftHand : [94,113] include 113
            # RightHand: [115,134] include 134
            # Pose: [0,10] include 10
            # 17,18
            pose_transform = Compose(
                                    DeleteFlowKeypoints(list(range(114, 115))), # 114
                                    DeleteFlowKeypoints(list(range(19, 94))),# 19 -> 93
                                    DeleteFlowKeypoints(list(range(11, 17))), # 11 -> 16
                                    ToFloatTensor())

            poseflow = pose_transform(poseflow).view(-1)
            poseflow_clip.append(poseflow)
        # Try to impute hand crops from frames where the elbow and wrist weren't missing as close as possible temporally
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

        clip = torch.stack(clip, dim=0)
        poseflow_clip = torch.stack(poseflow_clip, dim=0)
        clips.append(clip)
        poseflow_clips.append(poseflow_clip)
    clips = torch.stack(clips, dim=0)
    poseflow_clips = torch.stack(poseflow_clips,dim = 0)
    return {'video':clips,'poseflow':poseflow_clips}


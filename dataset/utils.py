import numpy as np
import torch
import cv2

def crop_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              transform,clip_len,missing_wrists_left,missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                missing_wrists_left.append(clip_len) # I tried this and achived 93% on test
                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    if not isinstance(left_hand_crop,np.ndarray):
        left_hand_crop = transform(left_hand_crop.numpy())
    else:
        left_hand_crop = transform(left_hand_crop)

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            missing_wrists_right.append(clip_len) # I tried this and achived 93% on test
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    if not isinstance(right_hand_crop,np.ndarray):
        right_hand_crop = transform(right_hand_crop.numpy())
    else:
        right_hand_crop = transform(right_hand_crop)

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
    # left_hand_crop = cv2.resize(left_hand_crop,(224,224))
    # right_hand_crop = cv2.resize(right_hand_crop,(224,224))
    # new_img = np.concatenate((right_hand_crop,left_hand_crop),axis = 1)
    # crops = transform(crops)

   

    return crops,missing_wrists_left,missing_wrists_right

def crop_optical_flow_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,resize,transform):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    if not isinstance(left_hand_crop,np.ndarray):
        left_hand_crop = left_hand_crop.numpy()
       
    left_hand_crop = resize(left_hand_crop)
    left_hand_crop = transform(left_hand_crop)

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    if not isinstance(right_hand_crop,np.ndarray):
        right_hand_crop = right_hand_crop.numpy()
    
    right_hand_crop = resize(right_hand_crop)
    right_hand_crop = transform(right_hand_crop)

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
   

    return crops




def crop_center(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              transform,clip_len,missing_center):

    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + 0.15 * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * 1.2
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))
    

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + 0.15 * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    xmin = min(min(right_hand_xmin,keypoints[0, 6]),min(left_hand_xmin,right_elbow[0]))
    xmax = max(max(left_hand_xmax,keypoints[0, 5]),max(right_hand_xmax,left_elbow[0]))
    
    ymin = min(max(0,min(left_wrist[1]-10,right_wrist[1]-10)),min(min(min(left_hand_ymin,right_hand_ymin),min(right_elbow[1],left_elbow[1])),min(keypoints[1, 6],keypoints[1, 5])))
    ymax = max(max(left_wrist[1]+10,right_wrist[1]+10),max(max(max(left_hand_ymax,right_hand_ymax),max(right_elbow[1],left_elbow[1])),max(keypoints[1, 6],keypoints[1, 5])))

    if int(ymax) - int(ymin) == 0 or int(xmax) - int(xmin) == 0:
        print("Missing center")
        missing_center.append(clip_len)
        frame = transform(frame)
        return frame,missing_center,True
    else:
        frame = transform(frame[int(ymin):int(ymax),int(xmin):int(xmax)])

        return frame,missing_center,False
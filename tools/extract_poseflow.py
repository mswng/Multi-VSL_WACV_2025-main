import argparse
import glob
import json
import math
import os
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

def read_pose(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = np.array(value['wholebody_threshold_02'])
        x = kps[:,0]
        y = kps[:,1]
        return np.stack((x, y), axis=1)

def read_neck_and_head_top(kp_file):
    with open(kp_file) as kf:
        value = json.loads(kf.read())
        kps = np.array(value['pose_threshold_02'])
        x = kps[:,0]
        y = kps[:,1]
        return np.stack((x, y), axis=1)[17:19]

def calc_pose_flow(prev, next):
    result = np.zeros_like(prev)
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
            candidates = poses[:, kpi]
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


def normalize(poses,neck_and_head_top):
    """Normalize each pose in the array to account for camera position. We normalize
    by dividing keypoints by a factor such that the length of the neck becomes 1."""
    new_poses = []
    for i in range(poses.shape[0]):
        upper_neck = neck_and_head_top[i][1]
        head_top = neck_and_head_top[i][0]
        neck_length = np.linalg.norm(upper_neck - head_top)
        poses[i] /= neck_length
        upper_neck  /= neck_length
        head_top /= neck_length
        new_pose = np.zeros((135,2))
        new_pose[:17] = poses[i][:17]
        new_pose[17] = upper_neck
        new_pose[18] = head_top
        new_pose[19:] = poses[i][17:]
        new_poses.append(new_pose)
        assert math.isclose(np.linalg.norm(upper_neck - head_top), 1)
    return np.stack(new_poses,axis = 0)


def main():
    input_dir_ = 'data/poses_1_1000'
    input_dirs = sorted(glob.glob(os.path.join(input_dir_, '*')))
    input_dir_index = 0
    total = len(input_dirs)

    for input_dir in tqdm(input_dirs, total=total):

        print(f'{input_dir_index}/{total}')
        input_dir_index += 1

        output_dir = input_dir.replace('poses_1_1000', 'poseflow_1_1000')
        if os.path.exists(output_dir):
            continue
        os.makedirs(output_dir, exist_ok=True)

        kp_files = sorted(glob.glob(os.path.join(input_dir, '*.json')))

        # 1. Collect all keypoint files and pre-process them
        poses = []
        neck_and_head_top = []
        for i in range(len(kp_files)):
            poses.append(read_pose(kp_files[i].replace('poses','wholebody')))
            neck_and_head_top.append(read_neck_and_head_top(kp_files[i]))
        if len(poses) == 0:
            print(input_dir)
        poses = np.stack(poses)
        neck_and_head_top = np.stack(neck_and_head_top)
        poses = impute_missing_keypoints(poses)
        neck_and_head_top = impute_missing_keypoints(neck_and_head_top)
        poses = normalize(poses,neck_and_head_top)

        # 2. Compute pose flow
        prev = poses[0]
        for i in range(1, poses.shape[0]):
            next = poses[i]
            flow = calc_pose_flow(prev, next)
            np.save(os.path.join(output_dir, 'flow_{:05d}'.format(i - 1)), flow)
            prev = next

if __name__ == '__main__':
    main()



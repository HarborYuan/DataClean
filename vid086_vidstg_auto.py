import copy
import os

import mmcv
import mmengine
import ffmpeg # ffmpeg_python==0.2.0
import numpy as np
from mmengine.visualization import Visualizer

PATH = 'data/video_grounding/VidSTG/val.json'
VIDEO_DIR = 'data/video_grounding/VidSTG'
OUT_DIR = 'data/vid/086_vidstg_auto'

SELECT = 'object'
SELECT_NAME = ['car', 'SUV', 'sedan', 'truck', 'train', 'airplane', 'bus']

VIDEO_SET = []


def judge(item):
    flag = item['type'] == SELECT
    flag = flag and any([" " + itm in item['caption'] for itm in SELECT_NAME])
    return flag

if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VideoGrounding",
        "data_source": "VidSTG",
        "type": "comprehension",
        "modality": {
            "in": ["video", "text"],
            "out": ["box"],
        },
        "version": "1.0",
        "data": [
        ]
    }
    data_list = out_json['data']
    json_file = mmengine.load(PATH)

    video_idx = 0
    cur_vid_out_path = os.path.join(OUT_DIR, "videos")
    assert not os.path.exists(cur_vid_out_path)
    mmengine.mkdir_or_exist(cur_vid_out_path)
    for item in json_file['videos']:
        video_path = os.path.join(VIDEO_DIR, 'video', item['video_path'])
        assert os.path.exists(video_path)
        caption = item['caption']

        if not judge(item):
            continue
        
        # if item['original_video_id'] not in VIDEO_SET:
        #     VIDEO_SET.append(item['original_video_id'])
        # else:
        #     continue

        # selected
        print(f"Caption: {caption}")
        new_item = copy.deepcopy(item)
        ext = video_path.split('.')[-1]
        mmengine.copyfile(video_path, os.path.join(cur_vid_out_path, '{:06d}.{}'.format(video_idx, ext)))
        new_item['video_path'] = '{:06d}.{}'.format(video_idx, ext)
        trajectory = json_file['trajectories'][item['original_video_id']][str(item['target_id'])]
        new_item['trajectory'] = trajectory
        data_list.append(new_item)
        video_idx += 1
        if video_idx >= 300:
            break

    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

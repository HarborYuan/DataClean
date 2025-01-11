import os

import mmcv
import mmengine
import ffmpeg # ffmpeg_python==0.2.0
import numpy as np
from mmengine.visualization import Visualizer

import copy

PATH = 'data/video_grounding/HC-STVG/val_v2_proc.json'
VIDEO_DIR = 'data/video_grounding/HC-STVG'
OUT_DIR = 'data/vid/083_hc2_dynamic'

SELECT = ['and']


def judge(text):
    return any([s in text for s in SELECT])

if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VideoGrounding",
        "data_source": "HC-STVG2",
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
    for item in json_file:
        video_path = os.path.join(VIDEO_DIR, 'video_parts', item['video_path'])
        assert os.path.exists(video_path)
        caption = item['caption']

        if not judge(caption):
            continue
        
        # selected
        print(caption)
        new_item = copy.deepcopy(item)
        ext = video_path.split('.')[-1]
        mmengine.copyfile(video_path, os.path.join(cur_vid_out_path, '{:06d}.{}'.format(video_idx, ext)))
        new_item['video_path'] = '{:06d}.{}'.format(video_idx, ext)
        data_list.append(new_item)
        video_idx += 1
        if video_idx >= 300:
            break

    print('done')
    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

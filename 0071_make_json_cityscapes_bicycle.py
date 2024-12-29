import os

import mmcv
import mmengine
import numpy as np
import pycocotools.mask as maskUtils

PATH = 'data/cityscapes-dvps/video_sequence/train'
OUT_DIR = 'data/seg/0071_cityscapes_bicycle'


CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
)

THING_CLASSES = (
    'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle'
)
STUFF_CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation',
    'terrain', 'sky'
)
NO_OBJ = 32
NO_OBJ_HB = 255
DIVISOR_PAN = 1000
NUM_THING = len(THING_CLASSES)
NUM_STUFF = len(STUFF_CLASSES)


FIND_PERSON = 0
FIND_CAR = 2

FIND_BICYCLE = 7

def to_coco(pan_map, divisor=0):
    # Haobo : This is to_coco situation #2
    # idx for stuff will be sem * div
    # Datasets: Cityscapes-DVPS
    pan_new = - np.ones_like(pan_map).astype(np.int32)

    thing_mapper = {CLASSES.index(itm): idx for idx, itm in enumerate(THING_CLASSES)}
    stuff_mapper = {CLASSES.index(itm): idx + NUM_THING for idx, itm in enumerate(STUFF_CLASSES)}
    mapper = {**thing_mapper, **stuff_mapper}
    for idx in np.unique(pan_map):
        if idx == NO_OBJ * DIVISOR_PAN:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        else:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = mapper[cls_id]
            inst_id = idx % DIVISOR_PAN
            if cls_id in stuff_mapper:
                assert inst_id == 0
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id
    assert -1. not in np.unique(pan_new)
    return pan_new

if __name__ == '__main__':
    file_list = list(sorted(mmengine.scandir(PATH, recursive=False, suffix='_leftImg8bit.png')))

    vid_frames = []
    last_vid = []
    last_vid_id = "xx"
    for file in file_list:
        vid_id, frame_id, _ = file.split('_', maxsplit=2)
        if vid_id != last_vid_id:
            last_vid = [file]
            vid_frames.append(last_vid)
        else:
            last_vid.append(file)
        last_vid_id = vid_id


    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VOS",
        "data_source": "Cityscapes",
        "type": "comprehension",
        "modality": {
            "in": ["video"],
            "out": ["video"],
        },
        "version": "1.0",
        "data": [
        ]
    }
    data_list = out_json['data']

    for vid in vid_frames:
        frame_0 = vid[0]
        img_path = os.path.join(PATH, frame_0)
        seg_path = os.path.join(PATH, frame_0.replace('_leftImg8bit.png', '_gtFine_instanceTrainIds.png'))

        seg_map = mmcv.imread(seg_path, flag='unchanged')
        seg_map = to_coco(seg_map, divisor=DIVISOR_PAN)

        selected_ids = []
        for pan_id in np.unique(seg_map):
            sem_id = pan_id // DIVISOR_PAN
            if sem_id == FIND_BICYCLE:
                selected_ids.append(pan_id)

                # choose 1
                break

        for selected_id in selected_ids:
            cur_vid_out_path = os.path.join(OUT_DIR, "images", "{:06d}".format(vid_id))
            assert not os.path.exists(cur_vid_out_path)
            mmengine.mkdir_or_exist(cur_vid_out_path)
            this_item = {
                "id": vid_id,
                "input": {
                    "video_folder": cur_vid_out_path
                },
                "output": {
                }
            }
            img_id = 0
            for frame in vid:
                img_path = os.path.join(PATH, frame)
                seg_path = os.path.join(PATH, frame.replace('_leftImg8bit.png', '_gtFine_instanceTrainIds.png'))
                seg_map = mmcv.imread(seg_path, flag='unchanged')
                seg_map = to_coco(seg_map, divisor=DIVISOR_PAN)

                mask = (seg_map == selected_id).astype(np.uint8)
                mask = maskUtils.encode(np.asfortranarray(mask))
                mask['counts'] = mask['counts'].decode()

                mmengine.copyfile(img_path, os.path.join(cur_vid_out_path, '{:06d}.jpg'.format(img_id)))

                this_item['output'][img_id] = {
                    'mask': mask,
                }

                img_id += 1
            data_list.append(this_item)

            vid_id += 1
            if vid_id >= 200:
                # stop at 300
                break
        if vid_id >= 200:
            # stop at 300
            break

    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

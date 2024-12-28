import os

import mmcv
import mmengine
import numpy as np
from mmengine import scandir
import pycocotools.mask as maskUtils

PATH = 'data/omg_data/VIPSeg/'
OUT_DIR = 'data/seg/002_vipseg_car'

INSTANCE_OFFSET_HB = 10000

NO_OBJ = 0
NO_OBJ_HB = 255
NO_OBJ_BUG = (200,)
DIVISOR_PAN = 100
NUM_THING = 58
NUM_STUFF = 66

from ext.class_names.VIPSeg import CLASSES_THING, CLASSES_STUFF, COCO_CLASSES, COCO_THINGS, COCO_STUFF, PALETTE


def to_coco(pan_map, divisor=INSTANCE_OFFSET_HB):
    pan_new = - np.ones_like(pan_map)
    vip2hb_thing = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_THING)}
    assert len(vip2hb_thing) == NUM_THING
    vip2hb_stuff = {itm['id'] + 1: idx for idx, itm in enumerate(CLASSES_STUFF)}
    assert len(vip2hb_stuff) == NUM_STUFF
    for idx in np.unique(pan_map):
        # 200 is a bug in vipseg dataset.
        # Please refer to https://github.com/VIPSeg-Dataset/VIPSeg-Dataset/issues/1
        if idx == NO_OBJ or idx in NO_OBJ_BUG:
            pan_new[pan_map == idx] = NO_OBJ_HB * divisor
        elif idx > 128:
            cls_id = idx // DIVISOR_PAN
            cls_new_id = vip2hb_thing[cls_id]
            inst_id = idx % DIVISOR_PAN
            pan_new[pan_map == idx] = cls_new_id * divisor + inst_id + 1
        else:
            cls_new_id = vip2hb_stuff[idx]
            cls_new_id += NUM_THING
            pan_new[pan_map == idx] = cls_new_id * divisor
    assert -1 not in np.unique(pan_new)
    return pan_new


cls_mapping = {
    "person": "person",
    "car": "car",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    'cattle': 'animal',
    'other_animal': 'animal',
}

if __name__ == '__main__':
    val_list = mmengine.list_from_file(os.path.join(PATH, 'val.txt'))

    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VOS",
        "data_source": "VIPSeg",
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

    for val_name in val_list:
        vid_folder = os.path.join(PATH, 'imgs', val_name)
        ann_folder = os.path.join(PATH, 'panomasks', val_name)

        _tmp_img_id = -1
        imgs_cur = sorted(list(map(
            lambda x: str(x), scandir(vid_folder, recursive=False, suffix='.jpg')
        )))
        pans_cur = sorted(list(map(
            lambda x: str(x), scandir(ann_folder, recursive=False, suffix='.png')
        )))

        # start: Find a thing in the first frame
        img_cur, pan_cur = imgs_cur[0], pans_cur[0]
        assert img_cur.split('.')[0] == pan_cur.split('.')[0]
        item_full = os.path.join(vid_folder, img_cur)
        inst_map = os.path.join(ann_folder, pan_cur)
        img_dict = {
            'img_path': item_full,
            'ann_path': inst_map,
        }
        assert os.path.exists(img_dict['img_path'])
        assert os.path.exists(img_dict['ann_path'])
        instances = []
        ann_map = mmcv.imread(img_dict['ann_path'], flag='unchanged').astype(np.uint32)
        img_dict['height'], img_dict['width'] = ann_map.shape
        pan_map = to_coco(ann_map, INSTANCE_OFFSET_HB)

        has_obj = False
        obj_ids = []
        for pan_seg_id in np.unique(pan_map):
            label = pan_seg_id // INSTANCE_OFFSET_HB
            if label == NO_OBJ_HB:
                continue
            label_name = COCO_CLASSES[label]
            if label_name not in cls_mapping:
                continue
            final_cls_name = cls_mapping[label_name]
            if not final_cls_name in ['car']:
                continue
            has_obj = True
            obj_ids.append(pan_seg_id)

        if not has_obj:
            continue
        assert len(obj_ids) > 0
        # end: Find a thing in the first frame

        # we will include this video
        for obj_id in obj_ids:
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

            for img_cur, pan_cur in zip(imgs_cur, pans_cur):
                assert img_cur.split('.')[0] == pan_cur.split('.')[0]
                _tmp_img_id += 1
                img_id = _tmp_img_id
                item_full = os.path.join(vid_folder, img_cur)
                inst_map = os.path.join(ann_folder, pan_cur)
                img_dict = {
                    'img_path': item_full,
                    'ann_path': inst_map,
                }
                assert os.path.exists(img_dict['img_path'])
                assert os.path.exists(img_dict['ann_path'])
                instances = []
                ann_map = mmcv.imread(img_dict['ann_path'], flag='unchanged').astype(np.uint32)
                img_dict['height'], img_dict['width'] = ann_map.shape
                pan_map = to_coco(ann_map, INSTANCE_OFFSET_HB)

                mask = (pan_map == obj_id).astype(np.uint8)
                mask = maskUtils.encode(np.asfortranarray(mask))
                mask['counts'] = mask['counts'].decode()

                mmengine.copyfile(img_dict['img_path'], os.path.join(cur_vid_out_path, '{:06d}.jpg'.format(img_id)))

                this_item['output'][img_id] = {
                    'mask': mask,
                }
            data_list.append(this_item)

            vid_id += 1
            _tmp_img_id = -1
            if vid_id >= 200:
                # stop at 300
                break
        if vid_id >= 200:
            # stop at 300
            break

    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

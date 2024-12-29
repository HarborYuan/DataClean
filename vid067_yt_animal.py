import os

import mmengine
import numpy as np
import pycocotools.mask as maskUtils

from ext.datasets.read_json import ReadJson

PATH = 'data/omg_data/youtube_vis_2019/'
OUT_DIR = 'data/vid/067_yt_animal/'


classes_2019_version = ('person', 'giant_panda', 'lizard', 'parrot',
                            'skateboard', 'sedan', 'ape', 'dog', 'snake',
                            'monkey', 'hand', 'rabbit', 'duck', 'cat',
                            'cow', 'fish', 'train', 'horse', 'turtle',
                            'bear', 'motorbike', 'giraffe', 'leopard',
                            'fox', 'deer', 'owl', 'surfboard', 'airplane',
                            'truck', 'zebra', 'tiger', 'elephant',
                            'snowboard', 'boat', 'shark', 'mouse', 'frog',
                            'eagle', 'earless_seal', 'tennis_racket')

LABEL = ['ape', 'dog', 'snake', 'rabbit', 'duck', 'cat', 'bear', 'zebra', 'tiger', 'elephant', 'shark', 'mouse', 'frog']
LABEL = [classes_2019_version.index(itm) for itm in LABEL]


if __name__ == '__main__':
    ann_json = os.path.join(PATH, 'annotations', 'youtube_vis_2019_train.json')
    img_path = os.path.join(PATH, 'train', 'JPEGImages')

    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VOS",
        "data_source": "YouTubeVOS",
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

    dataset = ReadJson(ann_json, img_path=img_path).load_data_list()

    for video_info in dataset:
        instance_first_frame = video_info['images'][0]['instances']
        obj_ids = []
        shape=None
        for instance_info in instance_first_frame:
            label = instance_info['bbox_label']
            assert label < len(classes_2019_version)
            instance_id = instance_info['instance_id']
            shape = instance_info['mask']['size']
            if label in LABEL:
                obj_ids.append(instance_id)

                # # limit 1
                # break

        if len(obj_ids) == 0:
            continue

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

            img_id = 0
            for img_info in video_info['images']:
                img_path = img_info['img_path']
                instances = img_info['instances']
                the_instance = None
                for instance_info in instances:
                    if instance_info['instance_id'] == obj_id:
                        the_instance = instance_info
                if the_instance is None:
                    mask = np.zeros(shape, dtype=np.uint8)
                    mask = maskUtils.encode(np.asfortranarray(mask))
                    mask['counts'] = mask['counts'].decode()
                else:
                    mask = the_instance['mask']
                    mask = maskUtils.frPyObjects(mask, mask['size'][0], mask['size'][1])
                    mask['counts'] = mask['counts'].decode()

                mmengine.copyfile(img_path, os.path.join(cur_vid_out_path, '{:06d}.jpg'.format(img_id)))

                this_item['output'][img_id] = {
                    'mask': mask,
                }

                img_id += 1

            data_list.append(this_item)

            vid_id += 1
            if vid_id >= 500:
                break
        if vid_id >= 500:
            break


    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

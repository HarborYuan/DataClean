import os

import mmengine
import numpy as np

import pycocotools.mask as maskUtils

PATH = 'data/video_datas/revos/meta_expressions_valid_.json'
MASK_PATH = 'data/video_datas/revos/mask_dict.json'
OUT_DIR = 'data/vid/076_revos_auto'

IMG_PREFIX = 'data/video_datas/revos/'

SELECT = ['airplane', ' car', ' bus', ' truck', ' train']
START = []

if __name__ == '__main__':
    json_file = mmengine.load(PATH)['videos']
    mask_json_file = mmengine.load(MASK_PATH)

    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "RefVOS",
        "data_source": "ReVOS",
        "type": "comprehension",
        "modality": {
            "in": ["video", "text"],
            "out": ["video"],
        },
        "version": "1.0",
        "data": [
        ]
    }
    data_list = out_json['data']

    metas = []
    anno_count = 0  # serve as anno_id
    vid2metaid = {}
    for vid_name in json_file:
        vid_express_data = json_file[vid_name]
        vid_frames = sorted(vid_express_data['frames'])
        vid_len = len(vid_frames)
        exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
        for exp_id in exp_id_list:
            exp_dict = vid_express_data['expressions'][exp_id]
            meta = {}
            meta['video'] = vid_name
            meta['exp'] = exp_dict['exp']  # str
            meta['mask_anno_id'] = exp_dict['anno_id']

            if 'obj_id' in exp_dict.keys():
                meta['obj_id'] = exp_dict['obj_id']
            else:
                meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
            meta['anno_id'] = [str(anno_count), ]
            anno_count += 1
            meta['frames'] = vid_frames
            meta['exp_id'] = exp_id

            meta['length'] = vid_len
            metas.append(meta)
            if vid_name not in vid2metaid.keys():
                vid2metaid[vid_name] = []
            vid2metaid[vid_name].append(len(metas) - 1)

    for vid_name in vid2metaid.keys():
        for meta_id in vid2metaid[vid_name]:
            meta = metas[meta_id]
            exp = meta['exp']
            video_name = meta['video']
            anno_ids = meta['mask_anno_id']
            if len(anno_ids) == 0:
                print("SKIP as not anno_ids")
                continue
            obj_masks = []
            shape = None
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = mask_json_file[anno_id]
                frames_masks_ = []
                for _mask in frames_masks:
                    if _mask is not None:
                        shape = _mask['size']
                        break
                for frame_idx in range(len(frames_masks)):
                    _mask = frames_masks[frame_idx]
                    if _mask is not None:
                        frames_masks_.append(maskUtils.decode(_mask))
                    else:
                        frames_masks_.append(np.zeros(shape, dtype=np.uint8))
                obj_masks.append(frames_masks_)

            masks = []
            for i_frame in range(len(obj_masks[0])):
                image_size = obj_masks[0][0].shape
                _mask = np.zeros(image_size, dtype=np.uint8)
                for i_anno in range(len(obj_masks)):
                    if obj_masks[i_anno][i_frame] is None:
                        continue
                    m = obj_masks[i_anno][i_frame].astype(np.uint8)
                    _mask = _mask | m
                _mask = maskUtils.encode(np.asfortranarray(_mask))
                _mask['counts'] = _mask['counts'].decode()
                masks.append(_mask)

            frames = meta['frames']
            images = []
            for frame_id in frames:
                images.append(os.path.join(meta['video'], frame_id + '.jpg'))

            flag = False
            for select in SELECT:
                if select in exp:
                    flag = True
                    break

            for start in START:
                if exp.startswith(start):
                    flag = True
                    break
            if not flag:
                continue

            print(exp)

            cur_vid_out_path = os.path.join(OUT_DIR, "images", "{:06d}".format(vid_id))
            assert not os.path.exists(cur_vid_out_path)
            mmengine.mkdir_or_exist(cur_vid_out_path)
            this_item = {
                "id": vid_id,
                "input": {
                    "video_folder": cur_vid_out_path,
                    "prompt": exp,
                },
                "output": {
                }
            }
            img_id = 0
            for idx, img_path in enumerate(images):
                mmengine.copyfile(os.path.join(IMG_PREFIX, img_path), os.path.join(cur_vid_out_path, '{:06d}.jpg'.format(img_id)))

                this_item['output'][img_id] = {
                    'mask': masks[idx],
                }
                img_id += 1

            data_list.append(this_item)


            vid_id += 1
            if vid_id >= 500:
                # stop at 300
                break
        if vid_id >= 500:
            # stop at 300
            break

    mmengine.dump(out_json, os.path.join(OUT_DIR, 'annotations', 'annotation.json'))

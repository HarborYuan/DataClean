import os

import mmengine
import numpy as np

import pycocotools.mask as maskUtils

PATH = 'data/video_datas/davis17/meta_expressions/valid/meta_expressions.json'
MASK_PATH = 'data/video_datas/davis17/valid/mask_dict.pkl'
OUT_DIR = 'data/vid/072_refDAVIS_human'

IMG_PREFIX = 'data/video_datas/davis17/valid/JPEGImages'

SELECT = []
START = ["a man", 'a women', 'a boy', 'a girl', 'a person']

if __name__ == '__main__':
    json_file = mmengine.load(PATH)['videos']
    mask_json_file = mmengine.load(MASK_PATH)

    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "RefVOS",
        "data_source": "DAVIS",
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
            meta['mask_anno_id'] = [str(anno_count), ]

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
            mask_id = meta['mask_anno_id']
            masks = mask_json_file[mask_id[0]]
            shape = -1
            if masks[0] is None:
                continue
            for mask_idx in range(len(masks)):
                if masks[mask_idx] is None:
                    mask = np.zeros(shape, dtype=np.uint8)
                    mask = maskUtils.encode(np.asfortranarray(mask))
                    mask['counts'] = mask['counts'].decode()
                    masks[mask_idx] = mask
                else:
                    masks[mask_idx]['counts'] = masks[mask_idx]['counts'].decode()
                    shape = masks[mask_idx]['size']

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

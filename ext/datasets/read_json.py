import copy
import os.path as osp
from typing import Tuple, List

from .cocoapi import COCO
from mmengine import fileio


class ReadJson:

    def __init__(self, ann_file, img_path):
        self.ann_file = ann_file
        self.data_prefix = {
            'img_path': img_path,
        }

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        with fileio.get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the classes
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        # used in `filter_data`
        self.img_ids_with_ann = set()

        img_ids = self.coco.get_img_ids()
        total_ann_ids = []
        # if ``video_id`` is not in the annotation file, we will assign a big
        # unique video_id for this video.
        single_video_id = 100000
        videos = {}
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            if 'video_id' not in raw_img_info:
                single_video_id = single_video_id + 1
                video_id = single_video_id
            else:
                video_id = raw_img_info['video_id']

            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id,
                    'images': [],
                    'video_length': 0
                }

            videos[video_id]['video_length'] += 1
            ann_ids = self.coco.get_ann_ids(
                img_ids=[img_id], cat_ids=self.cat_ids)
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))

            if len(parsed_data_info['instances']) > 0:
                self.img_ids_with_ann.add(parsed_data_info['img_id'])

            videos[video_id]['images'].append(parsed_data_info)

        data_list = [v for v in videos.values()]

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        data_info = {}

        data_info.update(img_info)
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'],
                                img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            if ann.get('instance_id', None):
                instance['instance_id'] = ann['instance_id']
            else:
                # image dataset usually has no `instance_id`.
                # Therefore, we set it to `i`.
                instance['instance_id'] = i
            instances.append(instance)
        data_info['instances'] = instances
        return data_info
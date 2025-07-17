import os
import json
import copy
import shutil

import tqdm

TEMPLATE = {
    'set_type': "openset",
    'task': 'bench',
    'data_source': 'dummy',
    'modality': {
        "in": [
            "video",
        ],
        "out": [
            "video"
        ]
    },
    'type': 'video comprehension',
    'domain': "Speech",
    'general_capability': ["Reaonsing Ablaity"],
    'version': "1.0",
    'count': 0,
}



METALIST = {
    '061_vipseg_car': {
        'task': 'IWAutoVOS',
        'task_fullname': 'In the Wild Automobile Video Object Segmentation',
        'task_id': '#V-C-5-1',
        'data_source': 'VIPSeg',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '062_vipseg_human': {
        'task': 'IWHumanVOS',
        'task_fullname': 'In the Wild Human Video Object Segmentation',
        'task_id': '#V-C-5-2',
        'data_source': 'VIPSeg',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '063_vipseg_animal': {
        'task': 'IWAnimalVOS',
        'task_fullname': 'In the Wild Animal Video Object Segmentation',
        'task_id': '#V-C-5-3',
        'data_source': 'VIPSeg',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '064_vipseg_furniture': {
        'task': 'IWFurnitureVOS',
        'task_fullname': 'In the Wild Furniture Video Object Segmentation',
        'task_id': '#V-C-5-4',
        'data_source': 'VIPSeg',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    # ok
    '065_yt_auto': {
        'task': 'AutoVOS',
        'task_fullname': 'Automobile Video Object Segmentation',
        'task_id': '#V-C-6-1',
        'data_source': 'YouTube-VOS',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '066_yt_human': {
        'task': 'HumanVOS',
        'task_fullname': 'Human Video Object Segmentation',
        'task_id': '#V-C-6-2',
        'data_source': 'YouTube-VOS',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '067_yt_animal': {
        'task': 'AnimalVOS',
        'task_fullname': 'Animal Video Object Segmentation',
        'task_id': '#V-C-6-3',
        'data_source': 'YouTube-VOS',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '068_yt_sports': {
        'task': 'SportsVOS',
        'task_fullname': 'Sports Video Object Segmentation',
        'task_id': '#V-C-6-4',
        'data_source': 'YouTube-VOS',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Culture",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '069_city_auto': {
        'task': 'AutoStreetVOS',
        'task_fullname': 'Automobile Street-Scene Video Object Segmentation',
        'task_id': '#V-C-7-1',
        'data_source': 'Cityscapes',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '070_city_human': {
        'task': 'HumanStreetVOS',
        'task_fullname': 'Human Street-Scene Video Object Segmentation',
        'task_id': '#V-C-7-2',
        'data_source': 'Cityscapes',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '071_city_bicycle': {
        'task': 'BicycleStreetVOS',
        'task_fullname': 'Bicycle Street-Scene Video Object Segmentation',
        'task_id': '#V-C-7-3',
        'data_source': 'Cityscapes',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '072_refDAVIS_human': {
        'task': 'HumanRVOS',
        'task_fullname': 'Human Referring Video Object Segmentation',
        'task_id': '#V-C-8-1',
        'data_source': 'Ref-DAVIS 2017',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '073_refDAVIS_animal': {
        'task': 'AnimalRVOS',
        'task_fullname': 'Animal Referring Video Object Segmentation',
        'task_id': '#V-C-8-2',
        'data_source': 'Ref-DAVIS 2017',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '074_revos_person': {
        'task': 'HumanReVOS',
        'task_fullname': 'Human Reasoning Video Object Segmentation',
        'task_id': '#V-C-9-1',
        'data_source': 'ReVOS',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    '075_revos_animal': {
        'task': 'AnimalReVOS',
        'task_fullname': 'Animal Reasoning Video Object Segmentation',
        'task_id': '#V-C-9-2',
        'data_source': 'ReVOS',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    '076_revos_auto': {
        'task': 'AutoReVOS',
        'task_fullname': 'Automobile Reasoning Video Object Segmentation',
        'task_id': '#V-C-9-3',
        'data_source': 'ReVOS',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    "077_refsav_human": {
        'task': 'HumanCReVOS',
        'task_fullname': 'Human Complex-Scene Reasoning Video Object Segmentation',
        'task_id': '#V-C-11-1',
        'data_source': 'SA-V',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Humanity",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    "078_refsav_animal": {
        'task': 'AnimalCReVOS',
        'task_fullname': 'Animal Complex-Scene Reasoning Video Object Segmentation',
        'task_id': '#V-C-11-2',
        'data_source': 'SA-V',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    "079_refsav_auto": {
        'task': 'AutoCReVOS',
        'task_fullname': 'Automobile Complex-Scene Reasoning Video Object Segmentation',
        'task_id': '#V-C-11-3',
        'data_source': 'SA-V',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    "080_refsav_human_part": {
        'task': 'HumanPartCReVOS',
        'task_fullname': 'Human Part Complex-Scene Reasoning Video Object Segmentation',
        'task_id': '#V-C-11-4',
        'data_source': 'SA-V',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
    "081_refsav_equipment": {
        'task': 'EquipmentCReVOS',
        'task_fullname': 'Equipment Complex-Scene Reasoning Video Object Segmentation',
        'task_id': '#V-C-11-5',
        'data_source': 'SA-V',
        'modality': {
            "in": [
                "text",
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Reasoning Ability", "Interactive Capability"],
        'version': "1.0",
    },
}



CLOSE_WITH_ANSWER = 1


import random
open_ratio = 0.4

def deterministic_shuffle(n):
    lst = list(range(n))
    random.seed(3407)
    random.shuffle(lst)
    return lst


OPEN_SAVE = 'release/open/'
CLOSE_SAVE = 'release/close/'
SOURCE_FOLDER = 'benchmark_data/'

if __name__ == '__main__':
    for folder in METALIST:
        folder = os.path.join(SOURCE_FOLDER, folder)
        assert os.path.exists(folder), f"Path {folder} does not exist"
    
    for folder in tqdm.tqdm(METALIST):
        folder = os.path.join(SOURCE_FOLDER, folder)
        assert os.path.exists(folder), f"Path {folder} does not exist"
        assert os.path.exists(os.path.join(folder, 'annotations/annotation.json')), f"Annotation file does not exist in {folder}"
        original_json = json.load(open(os.path.join(folder, 'annotations/annotation.json'), 'r'))

        tot_data = original_json['data']
        shuffled_indices = deterministic_shuffle(len(tot_data))
        tot_data = [tot_data[idx] for idx in shuffled_indices]

        open_data_list = tot_data[:int(len(tot_data) * open_ratio)]
        closed_data_list = tot_data[int(len(tot_data) * open_ratio):]

        # open data
        open_data_json = copy.deepcopy(TEMPLATE)
        open_data_json['set_type'] = "openset"
        open_data_json.update(METALIST[folder.split('/')[-1]])
        open_data_json['data'] = open_data_list
        open_data_json['count'] = len(open_data_list)

        package_name = open_data_json['task_fullname'].replace(' ', '').replace('-', '')
        open_save_path = os.path.join(OPEN_SAVE, 'video', 'comprehension', package_name)
        open_save_video_path = os.path.join(open_save_path, 'images')
        os.makedirs(open_save_video_path, exist_ok=True)
        for item in open_data_json['data']:
            video_id = item['id']
            ori_data_path = os.path.join(folder, 'images', "{:06d}".format(video_id))
            assert os.path.exists(ori_data_path), f"Path {ori_data_path} does not exist"
            assert ori_data_path[len(SOURCE_FOLDER):] in item['input']['video_folder'], f"Path {ori_data_path} not in {item['input']['video_path']}"
            item['input']['video_folder'] = item['input']['video_folder'].split(f'/{folder}/')[-1]

            # copy folder
            new_data_path = os.path.join(open_save_video_path, "{:06d}".format(video_id))
            shutil.copytree(ori_data_path, new_data_path, dirs_exist_ok=True)


        open_json_path = os.path.join(open_save_path, 'annotation.json')
        with open(open_json_path, 'w') as f:
            json.dump(open_data_json, f, indent=4)
        print(f"Open data json saved to {open_json_path}")

        # closed data
        closed_data_json = copy.deepcopy(TEMPLATE)
        closed_data_json['set_type'] = "closeset"
        closed_data_json.update(METALIST[folder.split('/')[-1]])
        closed_data_json['data'] = closed_data_list
        closed_data_json['count'] = len(closed_data_list)

        package_name = closed_data_json['task_fullname'].replace(' ', '').replace('-', '')
        closed_save_path = os.path.join(CLOSE_SAVE, 'video', 'comprehension', package_name)
        closed_save_video_path = os.path.join(closed_save_path, 'images')
        os.makedirs(closed_save_video_path, exist_ok=True)
        for item in closed_data_json['data']:
            video_id = item['id']
            ori_data_path = os.path.join(folder, 'images', "{:06d}".format(video_id))
            assert os.path.exists(ori_data_path), f"Path {ori_data_path} does not exist"
            assert ori_data_path[len(SOURCE_FOLDER):] in item['input']['video_folder'], f"Path {ori_data_path} not in {item['input']['video_path']}"
            item['input']['video_folder'] = item['input']['video_folder'].split(f'/{folder}/')[-1]

            # for close data, we remove output
            if CLOSE_WITH_ANSWER:
                pass
            else:
                if 'prompt' in item['input']:
                    del item['output']
                else:
                    # if without prompt, we keep the first output
                    item['output'] = {
                        '0': item['output']['0'],
                    }
                item['output'] = {}

            # copy folder
            new_data_path = os.path.join(closed_save_video_path, "{:06d}".format(video_id))
            shutil.copytree(ori_data_path, new_data_path, dirs_exist_ok=True)
        closed_json_path = os.path.join(closed_save_path, 'annotation.json')
        with open(closed_json_path, 'w') as f:
            json.dump(closed_data_json, f, indent=4)
        print(f"Closed data json saved to {closed_json_path}")

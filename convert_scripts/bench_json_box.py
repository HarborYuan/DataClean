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
    '082_hc2_static': {
        'task': 'StaticActionDet',
        'task_fullname': 'Spatial Temporal Static Action Detection',
        'task_id': '#V-C-10-1',
        'data_source': 'HC-STVG2',
        'modality': {
            "in": [
                "video",
                "text",
            ],
            "out": [
                "box"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '083_hc2_dynamic': {
        'task': 'DynamicActionDet',
        'task_fullname': 'Spatial Temporal Dynamic Action Detection',
        'task_id': '#V-C-10-2',
        'data_source': 'HC-STVG2',
        'modality': {
            "in": [
                "video",
                "text",
            ],
            "out": [
                "box"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '084_vidstg_person': {
        'task': 'HumanVG',
        'task_fullname': 'Human Visual Grounding',
        'task_id': '#V-C-12-1',
        'data_source': 'VidSTG',
        'modality': {
            "in": [
                "video",
                "text",
            ],
            "out": [
                "box"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '085_vidstg_animal': {
        'task': 'AnimalVG',
        'task_fullname': 'Animal Visual Grounding',
        'task_id': '#V-C-12-2',
        'data_source': 'VidSTG',
        'modality': {
            "in": [
                "video",
                "text",
            ],
            "out": [
                "box"
            ]
        },
        'domain': "Biology",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    },
    '086_vidstg_auto': {
        'task': 'AutoVG',
        'task_fullname': 'Vehicle Visual Grounding',
        'task_id': '#V-C-12-3',
        'data_source': 'VidSTG',
        'modality': {
            "in": [
                "video",
                "text",
            ],
            "out": [
                "box"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition", "Interactive Capability"],
        'version': "1.0",
    }
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
        open_save_video_path = os.path.join(open_save_path, 'videos')
        os.makedirs(open_save_video_path, exist_ok=True)
        for item in open_data_json['data']:
            video_path = item['video_path']
            ori_data_path = os.path.join(folder, 'videos', video_path)
            assert os.path.exists(ori_data_path), f"Path {ori_data_path} does not exist"

            # copy folder
            new_data_path = os.path.join(open_save_video_path, video_path)
            shutil.copyfile(ori_data_path, new_data_path)


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

        closed_save_path = os.path.join(CLOSE_SAVE, 'video', 'comprehension', package_name)
        closed_save_video_path = os.path.join(closed_save_path, 'videos')
        os.makedirs(closed_save_video_path, exist_ok=True)
        for item in closed_data_json['data']:
            video_path = item['video_path']
            ori_data_path = os.path.join(folder, 'videos', video_path)
            assert os.path.exists(ori_data_path), f"Path {ori_data_path} does not exist"

            if CLOSE_WITH_ANSWER:
                pass
            else:
                del item['trajectory']

            # copy folder
            new_data_path = os.path.join(closed_save_video_path, video_path)
            shutil.copyfile(ori_data_path, new_data_path)
        closed_json_path = os.path.join(closed_save_path, 'annotation.json')
        with open(closed_json_path, 'w') as f:
            json.dump(closed_data_json, f, indent=4)
        print(f"Closed data json saved to {closed_json_path}")

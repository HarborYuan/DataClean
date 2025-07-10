import os
import json
import copy
import shutil

import tqdm

import csv

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
    '087_sintel': {
        'task': 'SynVDE',
        'task_fullname': 'Synthetic Video Depth Estimation',
        'task_id': '#V-C-13-1',
        'data_source': 'Sintel',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition"],
        'version': "1.0",
    },
    '088_scannet': {
        'task': 'StaticVDE',
        'task_fullname': 'Static Video Depth Estimation',
        'task_id': '#V-C-13-2',
        'data_source': 'ScanNet',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition"],
        'version': "1.0",
    },
    '089_bonn': {
        'task': 'DynamicVDE',
        'task_fullname': 'Dynamic Video Depth Estimation',
        'task_id': '#V-C-13-3',
        'data_source': 'Bonn',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition"],
        'version': "1.0",
    },
    '090_kitti': {
        'task': 'StreetVDE',
        'task_fullname': 'Street Video Depth Estimation',
        'task_id': '#V-C-13-4',
        'data_source': 'KITTI',
        'modality': {
            "in": [
                "video",
            ],
            "out": [
                "video"
            ]
        },
        'domain': "General",
        'general_capability': ["Content Recognition"],
        'version': "1.0",
    }
}


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
        # read csv
        original_list = csv.reader(open(os.path.join(folder, 'split.csv'), newline=''))
        original_list = list(original_list)[1:]

        tot_data = original_list

        shuffled_indices = deterministic_shuffle(len(tot_data))
        tot_data = [tot_data[idx] for idx in shuffled_indices]

        open_data_list = tot_data[:int(len(tot_data) * open_ratio)]
        closed_data_list = tot_data[int(len(tot_data) * open_ratio):]

        # open data
        open_data_json = copy.deepcopy(TEMPLATE)
        open_data_json['set_type'] = "openset"
        open_data_json.update(METALIST[folder.split('/')[-1]])

        open_data_json['count'] = len(open_data_list)

        package_name = open_data_json['task_fullname'].replace(' ', '').replace('-', '')
        open_save_path = os.path.join(OPEN_SAVE, 'video', 'comprehension', package_name)
        open_save_video_path = os.path.join(open_save_path, 'video')
        open_save_depth_path = os.path.join(open_save_path, 'depth')
        os.makedirs(open_save_video_path, exist_ok=True)
        os.makedirs(open_save_depth_path, exist_ok=True)
        

        data_list = []
        for item in open_data_list:
            input_file, output_file = item
            input_file = os.path.basename(input_file)
            output_file = os.path.basename(output_file)

            source_input_path = os.path.join(folder, 'video', input_file)
            source_output_path = os.path.join(folder, 'video', output_file)
            assert os.path.exists(source_input_path), f"Path {source_input_path} does not exist"
            assert os.path.exists(source_output_path), f"Path {source_output_path} does not exist"

            # copy video
            new_input_path = os.path.join(open_save_video_path, input_file)
            new_output_path = os.path.join(open_save_depth_path, output_file)
            shutil.copy(source_input_path, new_input_path)
            shutil.copy(source_output_path, new_output_path)
            data_list.append({
                'input': input_file,
                'output': output_file,
                'id': input_file.replace('_rgb_left.mp4', ''),
            })

        open_data_json['data'] = data_list


        open_json_path = os.path.join(open_save_path, 'annotation.json')
        with open(open_json_path, 'w') as f:
            json.dump(open_data_json, f, indent=4)
        print(f"Open data json saved to {open_json_path}")

        # closed data
        closed_data_json = copy.deepcopy(TEMPLATE)
        closed_data_json['set_type'] = "closeset"
        closed_data_json.update(METALIST[folder.split('/')[-1]])
        closed_data_json['count'] = len(closed_data_list)

        package_name = closed_data_json['task_fullname'].replace(' ', '').replace('-', '')
        closed_save_path = os.path.join(CLOSE_SAVE, 'video', 'comprehension', package_name)
        closed_save_video_path = os.path.join(closed_save_path, 'video')
        closed_save_depth_path = os.path.join(closed_save_path, 'depth')
        os.makedirs(closed_save_video_path, exist_ok=True)
        os.makedirs(closed_save_depth_path, exist_ok=True)
        data_list = []

        for item in closed_data_list:
            input_file, output_file = item
            input_file = os.path.basename(input_file)
            output_file = os.path.basename(output_file)

            source_input_path = os.path.join(folder, 'video', input_file)
            source_output_path = os.path.join(folder, 'video', output_file)
            assert os.path.exists(source_input_path), f"Path {source_input_path} does not exist"
            assert os.path.exists(source_output_path), f"Path {source_output_path} does not exist"

            # copy video
            new_input_path = os.path.join(closed_save_video_path, input_file)
            new_output_path = os.path.join(closed_save_depth_path, output_file)
            shutil.copy(source_input_path, new_input_path)
            shutil.copy(source_output_path, new_output_path)
            data_list.append({
                'input': input_file,
                'output': output_file,
                'id': input_file.replace('_rgb_left.mp4', ''),
            })
        closed_data_json['data'] = data_list
        closed_json_path = os.path.join(closed_save_path, 'annotation.json')
        with open(closed_json_path, 'w') as f:
            json.dump(closed_data_json, f, indent=4)
        print(f"Closed data json saved to {closed_json_path}")


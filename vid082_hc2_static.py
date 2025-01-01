import os

import mmcv
import mmengine
import ffmpeg # ffmpeg_python==0.2.0
import numpy as np
from mmengine.visualization import Visualizer

PATH = 'data/video_grounding/HC-STVG/val_v2.json'
VIDEO_PART = 'data/video_grounding/HC-STVG/video_parts.json'
VIDEO_DIR = 'data/video_grounding/HC-STVG'
OUT_DIR = 'data/seg/tmpdebug'

SELECT = ['person', 'man', 'woman', 'girl', 'boy']


def visualize(pred_box, image, output_path):
    visualizer = Visualizer()
    visualizer.set_image(image)
    visualizer.draw_bboxes(pred_box)
    visual_result = visualizer.get_image()

    mmcv.imwrite(visual_result, output_path)


FPS = 5

if __name__ == '__main__':
    video_parts_json = mmengine.load(VIDEO_PART)
    video_part_reverse = {}
    for folder in video_parts_json:
        for file in video_parts_json[folder]:
            assert file not in video_part_reverse
            video_part_reverse[file] = folder

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
    for item in json_file:
        assert item in video_part_reverse

        if video_part_reverse[item] not in ['0', '1', '2']:
            continue

        video_path = os.path.join(VIDEO_DIR, 'video_parts', video_part_reverse[item], item)


        vid_info = json_file[item]
        video_num_images = vid_info["img_num"]
        video_fps = video_num_images / 20
        sampling_rate = FPS / video_fps
        assert sampling_rate <= 1  # downsampling at fps

        start_frame = 0
        end_frame = video_num_images - 1
        frame_ids = [start_frame]
        for frame_id in range(start_frame, end_frame):
            if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                frame_ids.append(frame_id)

        tube_start_frame = vid_info['st_frame']
        tube_end_frame = tube_start_frame + len(vid_info['bbox'])

        ss = 0
        t = 20
        w = vid_info["img_size"][1]
        h = vid_info["img_size"][0]
        cmd = ffmpeg.input(video_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) == len(frame_ids)

        mmengine.mkdir_or_exist(os.path.join(OUT_DIR, "{:06d}".format(video_idx)))
        img_folder = os.path.join(OUT_DIR, "{:06d}".format(video_idx), 'imgs')
        mmengine.mkdir_or_exist(img_folder)
        ann_folder = os.path.join(OUT_DIR, "{:06d}".format(video_idx), 'visualize')
        mmengine.mkdir_or_exist(ann_folder)
        for idx, frame_id in enumerate(frame_ids):
            if tube_end_frame > frame_id >= tube_start_frame:
                img = images_list[idx][...,::-1]
                x, y, _h, _w = vid_info['bbox'][frame_id - tube_start_frame]
                bbox = np.array([x, y, x + _h, y + _w])
            else:
                img = images_list[idx][...,::-1]
                bbox = None

            mmcv.imwrite(img, os.path.join(img_folder, '{:06d}.jpg'.format(idx)))

            if bbox is not None:
                visualize(bbox, img, os.path.join(ann_folder, '{:06d}.jpg'.format(idx)))
            else:
                mmcv.imwrite(img, os.path.join(ann_folder, '{:06d}.jpg'.format(idx)))

        print(vid_info['English'])
        print(vid_info['Chinese'])
        video_idx += 1


        if video_idx >= 10:
            break
    print('done')

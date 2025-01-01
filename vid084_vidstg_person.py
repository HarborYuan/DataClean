import os

import mmcv
import mmengine
import ffmpeg # ffmpeg_python==0.2.0
import numpy as np
from mmengine.visualization import Visualizer

PATH = 'data/video_grounding/VidSTG/val.json'
VIDEO_DIR = 'data/video_grounding/VidSTG'
OUT_DIR = 'data/seg/tmpdebug_vidstg'

SELECT = ['person', 'man', 'woman', 'girl', 'boy']


def visualize(pred_box, image, output_path):
    visualizer = Visualizer()
    visualizer.set_image(image)
    visualizer.draw_bboxes(pred_box)
    visual_result = visualizer.get_image()

    mmcv.imwrite(visual_result, output_path)


FPS = 5

if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        mmengine.mkdir_or_exist(OUT_DIR)

    vid_id = 0
    img_id = 0

    out_json = {
        "task": "VideoGrounding",
        "data_source": "VidSTG",
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
    for vid_info in json_file['videos']:
        video_num_images = vid_info["frame_count"]
        video_fps = vid_info["fps"]
        sampling_rate = FPS / video_fps
        assert sampling_rate <= 1  # downsampling at fps

        start_frame = vid_info["start_frame"]
        end_frame = vid_info["end_frame"]
        frame_ids = [start_frame]
        for frame_id in range(start_frame, end_frame):
            if int(frame_ids[-1] * sampling_rate) < int(frame_id * sampling_rate):
                frame_ids.append(frame_id)

        inter_frames = set(
            [
                frame_id
                for frame_id in frame_ids
                if vid_info["tube_start_frame"] <= frame_id < vid_info["tube_end_frame"]
            ]
        )  # frames in the annotated moment


        clip_start = vid_info["start_frame"]  # included
        clip_end = vid_info["end_frame"]  # excluded
        video_original_id = vid_info["original_video_id"]
        trajectory = json_file["trajectories"][video_original_id][
            str(vid_info["target_id"])
        ]

        vid_path = os.path.join(VIDEO_DIR, "video", vid_info["video_path"])
        video_fps = vid_info["fps"]
        ss = clip_start / video_fps
        t = (clip_end - clip_start) / video_fps
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        w = vid_info["width"]
        h = vid_info["height"]
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) == len(frame_ids)

        mmengine.mkdir_or_exist(os.path.join(OUT_DIR, "{:06d}".format(video_idx)))
        img_folder = os.path.join(OUT_DIR, "{:06d}".format(video_idx), 'imgs')
        mmengine.mkdir_or_exist(img_folder)
        ann_folder = os.path.join(OUT_DIR, "{:06d}".format(video_idx), 'visualize')
        mmengine.mkdir_or_exist(ann_folder)
        for idx, frame_id in enumerate(frame_ids):
            if frame_id in inter_frames:
                img = images_list[idx][...,::-1]
                x, y, _h, _w = trajectory[str(frame_id)]['bbox']
                bbox = np.array([x, y, x + _h, y + _w])
            else:
                img = images_list[idx][...,::-1]
                bbox = None

            mmcv.imwrite(img, os.path.join(img_folder, '{:06d}.jpg'.format(idx)))

            if bbox is not None:
                visualize(bbox, img, os.path.join(ann_folder, '{:06d}.jpg'.format(idx)))
            else:
                mmcv.imwrite(img, os.path.join(ann_folder, '{:06d}.jpg'.format(idx)))

        print(vid_info['caption'])
        video_idx += 1
        if video_idx >= 10:
            break
    print('done')

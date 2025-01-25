import argparse
import glob
import json
import os
import torch

import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import pycocotools.mask as maskUtils


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument(
        "--video-list-folder",
        default="data/benchmark/061_vipseg_car",
        metavar="FILE",
        help="path to the video list folder",
    )
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-8B")
    args = parser.parse_args()
    return args


RVOS_TEMPLATE = '<image>\nPlease segment {}.'

if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    ann_json = os.path.join(cfg.video_list_folder, "annotations/annotation.json")
    ann_json = json.load(open(ann_json))
    video_names = []
    for item in ann_json['data']:
        digit_id = item['id']
        video_name = f"{digit_id:06d}"
        video_names.append(video_name)

    print(f"running RVOS prediction on {len(video_names)} videos:\n")
    iou_list = []
    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        base_video_path = os.path.join(cfg.video_list_folder, "images")
        filenames = sorted(glob.glob(os.path.join(base_video_path, video_name, "*.jpg")))
        assert len(filenames), f"no images found for {video_name}"

        prompt = ann_json['data'][n_video]['input']['prompt'].strip()
        print(f"prompt: {RVOS_TEMPLATE.format(prompt)}")
        gt_list = ann_json['data'][n_video]['output']
        vid_frames = []
        for filename in filenames:
            image = Image.open(filename)
            vid_frames.append(image)

        result = model.predict_forward(
            video=vid_frames,
            text=RVOS_TEMPLATE.format(prompt),
            tokenizer=tokenizer,
        )
        prediction = result['prediction']
        print(f"The output is:\n{prediction}")
        if '[SEG]' in prediction:
            _seg_idx = 0
            pred_masks = result['prediction_masks'][_seg_idx]
            intersection = 0.
            union = 0.
            for frame_idx in range(len(gt_list)):
                gt_mask = gt_list[str(frame_idx)]['mask']
                gt_mask = maskUtils.decode(gt_mask)

                out_mask = pred_masks[frame_idx]
                # special case
                out_mask = out_mask.astype(np.uint8)

                assert gt_mask.shape == out_mask.shape, f"gt_mask.shape: {gt_mask.shape}, out_mask.shape: {out_mask.shape}"
                assert gt_mask.dtype == np.uint8 and out_mask.dtype == np.uint8, "type error"
                intersection += np.logical_and(out_mask, gt_mask).sum()
                union += np.logical_or(out_mask, gt_mask).sum()

            iou = intersection / union
        else:
            print("Warning: no segmentation found.")
            iou = 0.

        print(f"iou: {iou:.3f}")
        iou_list.append(iou)
    iou_mean = np.mean(iou_list) * 100
    print(f"mean iou: {iou_mean:.1f}")

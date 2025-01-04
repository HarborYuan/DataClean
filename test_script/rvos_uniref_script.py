# Modified by Haobo
# Please put it into uniref repo


# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import json
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import torch
import tqdm

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.modeling.meta_arch.build import build_model
from detectron2.utils.logger import setup_logger

from demo.predictor import VisualizationDemo
from detectron2.projects.uniref import add_uniref_config

import detectron2.data.transforms as T

import pycocotools.mask as maskUtils


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_uniref_config(cfg)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="projects/UniRef/configs/video/joint_task_vos_rvos_swin-l_16gpu.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default="checkpoints/video-joint_swin-l.pth",
        metavar="FILE",
        help="path to the model file",
    )
    parser.add_argument(
        "--video-list-folder",
        default="data/vid/072_refDAVIS_human",
        metavar="FILE",
        help="path to the video list folder",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    model = build_model(cfg)
    model.eval()
    state_dict = torch.load(args.weight, map_location="cpu")["model"]
    model.load_state_dict(state_dict, strict=True)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    input_format = cfg.INPUT.FORMAT
    input_format in ["RGB", "BGR"], input_format

    ann_json = os.path.join(args.video_list_folder, "annotations/annotation.json")
    ann_json = json.load(open(ann_json))
    video_names = []
    for item in ann_json['data']:
        digit_id = item['id']
        video_name = f"{digit_id:06d}"
        video_names.append(video_name)
    print(f"running VOS prediction on {len(video_names)} videos:\n")
    iou_list = []

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        base_video_path = os.path.join(args.video_list_folder, "images")
        filenames = sorted(glob.glob(os.path.join(base_video_path, video_name, "*.jpg")))
        assert len(filenames), f"no images found for {video_name}"

        prompt = ann_json['data'][n_video]['input']['prompt']
        gt_list = ann_json['data'][n_video]['output']

        with torch.no_grad():
            images = []
            for filename in filenames:
                original_image = read_image(filename, format="BGR")
                if input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                images.append(image)

            batched_inputs = [{
                'task': 'rvos',
                'image': images,
                "dataset_name": "refytvos",
                "height": height,
                "width": width,
                "file_names": filenames,
                "expressions": prompt,
            }]
            video_segments = model(batched_inputs)
            assert len(video_segments) == len(filenames)
            assert len(video_segments) == len(gt_list)

            intersection = 0
            union = 0
            for frame_idx in range(len(video_segments)):
                gt_mask = gt_list[str(frame_idx)]['mask']
                gt_mask = maskUtils.decode(gt_mask)

                out_mask = video_segments[frame_idx]

                # general case
                # out_ins_id = 0 # we only have one instance
                # out_mask = out_mask[out_ins_id].astype(np.uint8)

                # special case
                out_mask = out_mask.astype(np.uint8)

                assert gt_mask.shape == out_mask.shape, f"gt_mask.shape: {gt_mask.shape}, out_mask.shape: {out_mask.shape}"
                assert gt_mask.dtype == np.uint8 and out_mask.dtype == np.uint8, "type error"
                intersection += np.logical_and(out_mask, gt_mask).sum()
                union += np.logical_or(out_mask, gt_mask).sum()

            iou = intersection / union
            print(f"iou: {iou:.3f}")
            iou_list.append(iou)
    iou_mean = np.mean(iou_list) * 100
    print(f"mean iou: {iou_mean:.1f}")

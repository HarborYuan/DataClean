

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, Any, List
import json
import torch
import tqdm


from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import pycocotools.mask as mask_util
import numpy as np


PREFIX = 'data'

PROMPT = {
    'VOS': '<image>\nPlease segment the major object in the video.',
    'RVOS': '<image>\nPlease segment {}.',
    'ActionDet': '<image>\nPlease detect {}.',
    'VDE': '<image>\nPlease generate the depth map of the video.',
}


@dataclass
class Instance:
    input: Dict[str, Any]
    output: Dict[str, Any]
    id: str


class BaseTask(ABC):
    def __init__(self, task_data: str, model):
        self.task_data = task_data
        self.model = model
        self.task_name = os.path.basename(task_data)
        
        
        self.data = self._parse_data(task_data)

    @abstractmethod
    def _parse_data(self, task_data: str) -> List[Instance]:
        pass

    @abstractmethod
    def evaluate(self, results:List[Instance]) -> Dict[str, float]:
        pass

    @abstractmethod
    def run_inference(self) -> List[Instance]:
        pass


class TaskVOS(BaseTask):

    def _load_video(self, video_path: str) -> List[Image.Image]:
        video_frames = []
        for frame_file in sorted(os.listdir(video_path)):
            if frame_file.endswith('.jpg') or frame_file.endswith('.png'):
                frame_path = os.path.join(video_path, frame_file)
                video_frames.append(Image.open(frame_path).convert('RGB'))
        return video_frames
    
    
    def _parse_data(self, task_data: str) -> List[Instance]:
        json_path = os.path.join(task_data, 'annotation.json')
        json_data = json.load(open(json_path, 'r'))

        results = []
        json_data_data = json_data['data']
        for json_item in json_data_data:
            input_dict = {}
            input_dict['video_folder'] = json_item['input']['video_folder']
            input_dict['video'] = self._load_video(os.path.join(task_data, input_dict['video_folder']))

            output_dict = {}
            output_dict['serilized_masks'] = json_item['output']
            output_dict['masks'] = []
            for mask_id, mask_data in output_dict['serilized_masks'].items():
                mask = mask_util.decode(mask_data['mask'])
                output_dict['masks'].append(mask)
            instance_id = json_item['id']
            results.append(Instance(input=input_dict, output=output_dict, id=instance_id))
        return results

        

    def evaluate(self, results:List[Instance]) -> Dict[str, float]:
        iou_list = []
        for instance in results:
            masks = instance.output['masks']
            prediction_masks = instance.output['prediction_masks']

            assert len(masks) == len(prediction_masks), "Number of masks and prediction masks do not match."
            
            intersection = 0.
            union = 0.
            for gt_mask, pred_mask in zip(masks, prediction_masks):
                intersection += (gt_mask.astype(bool) & pred_mask.astype(bool)).sum()
                union += (gt_mask | pred_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)
        iou_mean = np.mean(iou_list).item() * 100
        return {"IoU": iou_mean}

    def run_inference(self) -> List[Instance]:
        results = []
        for instance in tqdm.tqdm(self.data, desc=f"Running inference on {self.task_name}"):
            input_data = instance.input

            result = self.model.predict_forward(
                video=input_data['video'],
                text=PROMPT['VOS'],
            )

            # output postprocessing
            output_masks = result['prediction_masks']

            instance.output['prediction_masks'] = output_masks[0]
            results.append(instance)
        return results


class TaskRVOS(BaseTask):
    def _load_video(self, video_path: str) -> List[Image.Image]:
        video_frames = []
        for frame_file in sorted(os.listdir(video_path)):
            if frame_file.endswith('.jpg') or frame_file.endswith('.png'):
                frame_path = os.path.join(video_path, frame_file)
                video_frames.append(Image.open(frame_path).convert('RGB'))
        return video_frames
    
    
    def _parse_data(self, task_data: str) -> List[Instance]:
        json_path = os.path.join(task_data, 'annotation.json')
        json_data = json.load(open(json_path, 'r'))

        results = []
        json_data_data = json_data['data']
        for json_item in json_data_data:
            input_dict = {}
            input_dict['video_folder'] = json_item['input']['video_folder']
            input_dict['video'] = self._load_video(os.path.join(task_data, input_dict['video_folder']))
            input_dict['prompt'] = json_item['input']['prompt']

            output_dict = {}
            output_dict['serilized_masks'] = json_item['output']
            output_dict['masks'] = []
            for mask_id, mask_data in output_dict['serilized_masks'].items():
                mask = mask_util.decode(mask_data['mask'])
                output_dict['masks'].append(mask)
            instance_id = json_item['id']
            results.append(Instance(input=input_dict, output=output_dict, id=instance_id))
        return results

        

    def evaluate(self, results:List[Instance]) -> Dict[str, float]:
        iou_list = []
        for instance in results:
            masks = instance.output['masks']
            prediction_masks = instance.output['prediction_masks']

            assert len(masks) == len(prediction_masks), "Number of masks and prediction masks do not match."
            
            intersection = 0.
            union = 0.
            for gt_mask, pred_mask in zip(masks, prediction_masks):
                intersection += (gt_mask.astype(bool) & pred_mask.astype(bool)).sum()
                union += (gt_mask | pred_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            iou_list.append(iou)
        iou_mean = np.mean(iou_list).item() * 100
        return {"IoU": iou_mean}

    def run_inference(self) -> List[Instance]:
        results = []
        for instance in tqdm.tqdm(self.data, desc=f"Running inference on {self.task_name}"):
            input_data = instance.input

            result = self.model.predict_forward(
                video=input_data['video'],
                text=PROMPT['RVOS'].format(input_data['prompt']),
            )

            # output postprocessing
            output_masks = result['prediction_masks']

            instance.output['prediction_masks'] = output_masks[0]
            results.append(instance)
        return results
    


class ActionDet(BaseTask):
    def _load_video(self, video_path: str) -> List[Image.Image]:
        import cv2
        cap = cv2.VideoCapture(video_path)
        img_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_list.append(Image.fromarray(frame).convert('RGB'))

        return img_list

    
    def _parse_data(self, task_data: str) -> List[Instance]:
        if self.task_name in ['AnimalVG', 'AutoVG', 'HumanVG']:
            self.is_vg = True
        else:
            self.is_vg = False
        
        json_path = os.path.join(task_data, 'annotation.json')
        json_data = json.load(open(json_path, 'r'))

        results = []
        json_data_data = json_data['data']
        for json_item in json_data_data:
            video_path = os.path.join(self.task_data, 'videos', json_item['video_path'])
            image_list = self._load_video(video_path)
            assert len(image_list) > 0, f"Video {video_path} has no frames."
            if len(image_list) != json_item['frame_count']:
                print(f"Warning: Frame count mismatch for video {video_path}. Expected {json_item['frame_count']}, got {len(image_list)}.")
                while len(image_list) < json_item['frame_count']:
                    image_list.append(image_list[-1])
            input_dict = {}
            input_dict['video'] = image_list
            input_dict['prompt'] = json_item['caption']

            output_dict = {}
            if self.is_vg:
                output_dict['tube_start_frame'] = json_item['tube_start_frame']
                output_dict['tube_end_frame'] = json_item['tube_end_frame']
            else:
                output_dict['tube_start_frame'] = json_item['tube_start_frame'] - 1
                output_dict['tube_end_frame'] = json_item['tube_end_frame'] - 1
            
            trajectory = json_item['trajectory']

            if self.is_vg:
                trajectory = [trajectory[frame_id_str]['bbox'] for frame_id_str in trajectory if output_dict['tube_start_frame'] <= int(frame_id_str) < output_dict['tube_end_frame']]

            assert len(trajectory) == output_dict['tube_end_frame'] - output_dict['tube_start_frame']
            bboxes = []
            for _ in range(output_dict['tube_start_frame']):
                bboxes.append([0, 0, 0, 0])

            # trajectory is a list of [x, y, w, h] for each frame
            for item in trajectory:
                x, y, w, h = item
                bbox = [x, y, x + w, y + h]
                bboxes.append(bbox)
            
            for _ in range(output_dict['tube_end_frame'], len(image_list)):
                bboxes.append([0, 0, 0, 0])
            output_dict['bboxes'] = bboxes

            instance_id = json_item['original_video_id']
            results.append(Instance(input=input_dict, output=output_dict, id=instance_id))
        return results

    def evaluate(self, results:List[Instance]) -> Dict[str, float]:
        iou_list = []
        for instance in results:
            boxes = instance.output['bboxes']
            prediction_boxes = instance.output['prediction_boxes']
            assert len(boxes) == len(prediction_boxes), "Number of boxes and prediction boxes do not match."
            iou = 0.
            frame_union = 0
            for gt_box, pred_box in zip(boxes, prediction_boxes):
                gt_box = np.array(gt_box)
                pred_box = np.array(pred_box)

                if np.all(gt_box == 0) and np.all(pred_box == 0):
                    continue
                frame_union += 1
                if np.all(gt_box == 0) or np.all(pred_box == 0):
                    continue
                
                intersection = np.maximum(0, np.minimum(gt_box[2:], pred_box[2:]) - np.maximum(gt_box[:2], pred_box[:2]))
                intersection_area = intersection[0] * intersection[1]
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                union_area = gt_area + pred_area - intersection_area
                iou += intersection_area / union_area
            if frame_union > 0:
                iou /= frame_union
            iou_list.append(iou)
        iou_mean = np.mean(iou_list).item() * 100
        return {"vIoU": iou_mean}

    def run_inference(self) -> List[Instance]:
        results = []
        for instance in tqdm.tqdm(self.data, desc=f"Running inference on {self.task_name}"):
            input_data = instance.input

            result = self.model.predict_boxes(
                video=input_data['video'],
                text=PROMPT['ActionDet'].format(input_data['prompt']),
            )

            # output postprocessing
            output_masks = result['prediction_boxes']
            instance.output['prediction_boxes'] = output_masks[0]
            results.append(instance)
        return results

tasks = {
    'AnimalVOS': TaskVOS,
    'AutoVOS':TaskVOS,
    'HumanVOS':TaskVOS,
    'SportsVOS':TaskVOS,

    ## IW
    'IWAnimalVOS':TaskVOS,
    'IWAutoVOS':TaskVOS,
    'IWFurnitureVOS':TaskVOS,
    'IWHumanVOS':TaskVOS,

    ## Street
    'AutoStreetVOS':TaskVOS,
    'BicycleStreetVOS':TaskVOS,
    'HumanStreetVOS':TaskVOS,
    
    # RVOS
    'AnimalRVOS':TaskRVOS,
    'HumanRVOS':TaskRVOS,

    ## ReVOS,
    'AnimalReVOS':TaskRVOS,
    'AutoReVOS': TaskRVOS,
    'HumanReVOS': TaskRVOS,

    ## CReVOS
    'AnimalCReVOS': TaskRVOS,
    'AutoCReVOS'    : TaskRVOS,
    'HumanCReVOS': TaskRVOS,
    'HumanPartCReVOS': TaskRVOS,
    'EquipmentCReVOS': TaskRVOS,


    ## Action Det
    # V-C-10 HCSTVG2
    'StaticActionDet': ActionDet,
    'DynamicActionDet': ActionDet,
    # V-C-12 VidSTG
    'AnimalVG': ActionDet,
    'AutoVG': ActionDet,
    'HumanVG': ActionDet,
}



def predict_dummy_boxes(video, text):
    # Dummy function to simulate box prediction
    # In practice, this should call the model's prediction method
    num_frames = len(video)
    return {
        'prediction_boxes': [
            [[0,0, 100, 100]] * num_frames, # Example boxes, [0, 0, 0, 0] is empty box
        ]
    }



def main(root:str, model_path:str):
    metrics = {}
    
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
    model.preparing_for_generation(tokenizer=tokenizer)
    
    model.predict_boxes = predict_dummy_boxes
    
    for task_name in tasks:
        task_class = tasks[task_name]
        task_data_path = os.path.join(root, task_name)
        task_instance = task_class(task_data=task_data_path, model=model)

        results = task_instance.run_inference()
        evaluation_results = task_instance.evaluate(results)
        metrics[task_instance.task_name] = evaluation_results
    
    print(metrics)


if __name__ == "__main__":
    root = os.path.join(PREFIX, "General-Bench-Openset/video/comprehension")
    import argparse
    from eval_scripts.video_comprehension_tasks import main
    parser = argparse.ArgumentParser(description="Run video tasks evaluation.")
    parser.add_argument("--model_path", type=str, default='ByteDance/Sa2VA-4B', required=False, help="Model to use for evaluation")
    args = parser.parse_args()
    main(root, args.model_path)

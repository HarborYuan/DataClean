import os

import mmcv
import mmengine
from mmengine.visualization import Visualizer
from pycocotools import mask as maskUtils

JSON_PATH = 'data/vid/069_city_auto/annotations/annotation.json'
TASK_ID = 'vis069'
RATIO = .05


def visualize(pred_mask, image, output_path):
    visualizer = Visualizer()
    visualizer.set_image(image)
    visualizer.draw_binary_masks(pred_mask, colors='g')
    visual_result = visualizer.get_image()

    mmengine.mkdir_or_exist(os.path.dirname(output_path))
    mmcv.imwrite(visual_result, output_path)


if __name__ == '__main__':
    json_content = mmengine.load(JSON_PATH)['data']

    tot = 0

    for item in json_content:
        video_folder = item['input']['video_folder']
        vid_len = len(item['output'])
        vid_id = item['id']
        out_folder = os.path.join(f'./data/visualize/{TASK_ID}', "{:06d}".format(vid_id))

        flag = True
        for image_id in range(vid_len):
            img_path = os.path.join(video_folder, "{:06d}.jpg".format(image_id))
            img = mmcv.imread(img_path)
            mask = maskUtils.decode(item['output'][str(image_id)]['mask']).astype(bool)

            if image_id == 0 and float(mask.sum()) / float(mask.shape[0] * mask.shape[1]) < RATIO:
                flag = False
                break

            mmengine.mkdir_or_exist(out_folder)
            mmengine.mkdir_or_exist(os.path.join(out_folder, 'img'))
            mmengine.mkdir_or_exist(os.path.join(out_folder, 'mask'))
            mmengine.copyfile(img_path, os.path.join(out_folder, 'img', "{:06d}.jpg".format(image_id)))
            visualize(mask, img, os.path.join(out_folder, 'mask', "{:06d}.jpg".format(image_id)))

        if flag:
            tot += 1

        if tot >= 5:
            break

    print("Done")

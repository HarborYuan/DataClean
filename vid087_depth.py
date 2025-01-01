import os

import mmcv
import numpy as np
import torch

import matplotlib.cm as cm
from PIL import Image

DEPTH_PATH = 'data/video_depth/rgbd_bonn_dataset/rgbd_bonn_synchronous'


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array

    depth_png = np.asarray(Image.open(filename))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255

    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth

class ColorMapper:
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res


if __name__ == '__main__':
    depth_path = os.path.join(DEPTH_PATH, 'depth')
    file_list = os.listdir(depth_path)
    for file in file_list:
        img_path = os.path.join(depth_path, file)
        depth_map = depth_read(img_path)

        v_max = depth_map.max()
        depth_map[depth_map == -1] = v_max
        mmcv.imwrite(vis_sequence_depth(depth_map), './depth_vis.png')
        assert 0

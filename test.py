#!/usr/bin/python
# -*- encoding: utf-8 -*-
import glob
import multiprocessing
from re import T
import argparse
import scipy.ndimage

from logger import setup_logger
from model import BiSeNet

import torch
import torch.multiprocessing
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from functools import reduce
from torch.multiprocessing import set_start_method

def vis_parsing_maps(im, parsing_anno, info, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts

    part_colors = [[0, 0, 0],
                   [250, 0, 0],  # skin face
                   [240, 0, 0],  # l_brow
                   [230, 0, 0],  # r_brow
                   [210, 0, 0],  # l_eye
                   [200, 0, 0],  # r_eye
                   [190, 0, 0],  # eye_glasses
                   [180, 0, 0],  # l_ear
                   [170, 0, 0],  # r_ear
                   [160, 0, 0],  # ear_r
                   [150, 0, 0],  # nose
                   [140, 0, 0],  # mouth
                   [130, 0, 0],  # u_lip
                   [120, 0, 0],  # l_lip
                   [110, 0, 0],  # neck
                   [100, 0, 0],  # neck_l
                   [90, 0, 0],  # cloth
                   [80, 0, 0],  # hair
                   [70, 0, 0],  # hat
                   [60, 0, 0],
                   [50, 0, 0],
                   [40, 0, 0],
                   [30, 0, 0],
                   [20, 0, 0],
                   [10, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    # vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    img = Image.fromarray(vis_parsing_anno_color)
    img = np.array(img)
    cv2.imwrite(save_path.replace('matted', 'seg_mask'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
save_pth = osp.join('res/cp', '79999_iter.pth')
net.load_state_dict(torch.load(save_pth))
net.eval()


def job(image_path):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        info = {}
        #image = img.resize((512, 512), Image.BILINEAR)
        #image, info = detector.get(image_path)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing_maps(image, parsing, info, stride=1, save_im=True, save_path=image_path)


def evaluate(actor):
    os.system(f"mkdir -p {actor}/seg_mask")
    paths = sorted(glob.glob(f'{actor}/matted/*.png'))
    for path in tqdm(paths):
        job(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor', help='Actor', required=True)
    args = parser.parse_args()
    set_start_method('forkserver', force=True)
    evaluate(args.actor)

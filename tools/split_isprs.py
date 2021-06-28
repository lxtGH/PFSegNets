#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from natsort import natsorted
from shutil import copyfile
from PIL import Image

import argparse
parser = argparse.ArgumentParser(description='Splitting the Images')

parser.add_argument('--src', default='./dataset/posdam', type=str, help='path for the original dataset')
parser.add_argument('--tar', default='./dataset/posdam_patches', type=str, help='path for saving the new dataset')
parser.add_argument('--image_sub_folder', default='images', type=str, help='name of subfolder inside the training, validation and test folders')
parser.add_argument('--set', default="train,val,test", type=str, help='evaluation mode')
parser.add_argument('--patch_width', default=800, type=int, help='Width of the cropped image patch')
parser.add_argument('--patch_height', default=800, type=int, help='Height of the cropped image patch')
parser.add_argument('--overlap_area', default=200, type=int, help='Overlap area')


args = parser.parse_args()

src=args.src
tar=args.tar
modes1=args.set.split(',')
mode2=args.image_sub_folder
patch_H, patch_W = args.patch_width, args.patch_height # image patch width and height
overlap = args.overlap_area #overlap area
extras = []

for i in modes1:
    extras=['',  '_gray_mask']
for mode1 in modes1:
    src_path = src  + "/" + mode1 + "/" + mode2
    tar_path = tar  + "/" + mode1 + "/" + mode2

    os.makedirs(tar_path, exist_ok=True)
    files = os.listdir(src_path)
    files = [token.split('.')[0] for token in files if token.endswith('.tif')]
    files = natsorted(files)

    for file_ in files:
        for extra in extras:
            if extra == '_gray_mask':
                filename = file_ + extra + '.png'
            else:
                filename = file_ + extra + '.tif'
            full_filename = src_path + '/' + filename
            img = cv2.imread(full_filename)
            img_H, img_W, _ = img.shape
            X = np.zeros_like(img,dtype=float)
            h_X, w_X,_ = X.shape

            if extra == '_gray_mask':
                save_path = tar_path + '/masks'
                os.makedirs(save_path, exist_ok=True)
            else:
                save_path = tar_path + '/images'
                os.makedirs(save_path, exist_ok=True)

            if img_H > patch_H and img_W > patch_W:
                for x in range(0, img_W, patch_W-overlap):
                    for y in range(0, img_H, patch_H-overlap):
                        x_str = x
                        x_end = x + patch_W
                        if x_end > img_W:
                            diff_x = x_end - img_W
                            x_str-=diff_x
                            x_end = img_W
                        y_str = y
                        y_end = y + patch_H
                        if y_end > img_H:
                            diff_y = y_end - img_H
                            y_str-=diff_y
                            y_end = img_H
                        if extra == '_gray_mask':
                            patch = img[y_str:y_end, x_str:x_end]
                        else:
                            patch = img[y_str:y_end,x_str:x_end,:]
                        image = file_+'_'+str(y_str)+'_'+str(y_end)+'_'+str(x_str)+'_'+str(x_end)+extra+'.png'
                        print(image)
                        save_path_image = save_path + '/' + image
                        if extra == '_gray_mask':
                            patch = Image.fromarray(patch)
                            patch = patch.convert('L')
                            patch.save(save_path_image)
                        else:
                            cv2.imwrite(save_path_image, patch)
            else:
                copyfile(full_filename, save_path+'/'+filename)

import os
import os.path as osp
import argparse
from PIL import Image
import cv2
from natsort import natsorted
import numpy as np

######### BGR!!!!!! #######
mask_mapping = {
    (255, 255, 255): 0,             # impervious surfaces
    (255, 0, 0): 1,                 # buildings
    (255, 255, 0): 2,               # low vegetation
    (0, 255, 0): 3,                 # trees
    (0, 255, 255): 4,               # cars
    (0, 0, 255): 5                  # cluster
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', help='root dir of isprs dataset', required=True)
    parser.add_argument('--save_file_name', help='save dir of converted gray masks of iSAID dataset', required=True)
    parser.add_argument('--phase', choices=['train', 'val', 'test'], required=True)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    masks_path = osp.join(args.root_path, args.phase, 'masks')
    save_path = osp.join(args.root_path, args.phase, args.save_file_name)
    os.makedirs(save_path, exist_ok=True)
    mask_names = os.listdir(masks_path)
    mask_names = natsorted(mask_names)
    num_masks = len(mask_names)
    i = 0
    for token in mask_names:
        i = i+1
        if not token.endswith('.tif'):
            continue
        save_name = osp.join(save_path, token.replace('.tif', '.png'))
        token_path = osp.join(masks_path, token)
        mask_array = cv2.imread(token_path)
        mask_gray = np.zeros(mask_array.shape[:2])
        for k, v in mask_mapping.items():
            mask_gray[(mask_array == k).all(axis=2)] = v
        assert mask_gray.max() <= 6, mask_gray.max()
        labels = np.unique(mask_gray)

        mask_gray = Image.fromarray(mask_gray.astype('uint8')).convert('L')
        mask_gray.save(save_name)
        print(f'Converted {token} to gray mask, [{i}/{num_masks}], {labels}')


if __name__ == '__main__':
    main()
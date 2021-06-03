import sys
import os
from os.path import join
import glob
import cv2
import pdb
import matplotlib.pyplot as plt
import numpy as np
from detection_utils import MaskRCNNDetector
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Detect labels on PV images')
parser.add_argument("--recording_path",
                        required=True,
                        help="Path to base folder")
parser.add_argument("--valid_categories",
                        required=False,
                        default="tv,couch,dining table,chair",
                        help="Categories that are to be detected")

args = parser.parse_args()

source = 'PV'
dest = 'labels'

basedir = args.recording_path
pv_dir = join(basedir, f'{source}')
Path(join(basedir, f'{dest}')).mkdir(exist_ok=True)

valid_categories = args.valid_categories.split(',')

config_file = "configs/mask_rcnn_R_50_FPN_inference_acc_test.yaml"
detector = MaskRCNNDetector(config_file, valid_categories=valid_categories)

# Collect colors of valid categories from COCO palette.
colors = []
for vc in valid_categories:
	colors.append(detector.category_colors[detector.category_names.index(vc)])

image_files = glob.glob(join(pv_dir,'*.png'))
image_files = sorted(image_files)

for i, image_file in enumerate(tqdm(image_files)):
	output_file = image_file.replace(f'{source}',f'{dest}')
	img = cv2.imread(image_file)

	output_masks, output_category_names, output_scores = detector.run_on_opencv_image(img)

	# set mask to zero
	if output_masks is None:
		cv2.imwrite(output_file,np.zeros_like(img)*255)
		continue

	num_detections = len(output_masks)

	masks = np.zeros((len(valid_categories) + 1, img.shape[0], img.shape[1]))
	masks_colored = []

	# for each pixel, keep the label with maximum score
	for idx in range(0, num_detections):
		vc_idx = valid_categories.index(output_category_names[idx]) + 1
		masks[vc_idx] = masks[vc_idx] + (output_masks[idx].astype(np.float))*np.array(output_scores[idx])
	masks = np.argmax(masks, axis=0)
	
	# color the masks according to COCO palette
	for vc_idx in range(0, len(valid_categories)):
		idx = masks == vc_idx + 1
		masks_ = np.zeros((img.shape[0], img.shape[1],3))
		masks_[idx,0] = colors[vc_idx][0]; masks_[idx,1] = colors[vc_idx][1]; masks_[idx,2] = colors[vc_idx][2]
		masks_colored.append(masks_)
	masks_colored = np.array(masks_colored).sum(axis=0)

	cv2.imwrite(output_file,masks_colored)
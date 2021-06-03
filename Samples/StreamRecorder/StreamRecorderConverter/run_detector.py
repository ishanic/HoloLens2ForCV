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
# import pdb; pdb.set_trace()
valid_categories = args.valid_categories.split(',')

config_file = "configs/mask_rcnn_R_50_FPN_inference_acc_test.yaml"
detector = MaskRCNNDetector(config_file, valid_categories=valid_categories)
colors = []
for vc in valid_categories:
	colors.append(detector.category_colors[detector.category_names.index(vc)])

image_files = glob.glob(join(pv_dir,'*.png'))
image_files = sorted(image_files)

for i, image_file in enumerate(tqdm(image_files)):
	# print(image_file)
	output_file = image_file.replace(f'{source}',f'{dest}')
	img = cv2.imread(image_file)

	output = detector.run_on_opencv_image(img)

	if output[0] is None:
		cv2.imwrite(output_file,np.zeros_like(img)*255)
		continue

	num_detections = len(output[0])

	masks = np.zeros((len(valid_categories) + 1, img.shape[0], img.shape[1]))
	masks_colored = []
	for idx in range(0, num_detections):
		vc_idx = valid_categories.index(output[1][idx]) + 1
		masks[vc_idx] = masks[vc_idx] + (output[0][idx].astype(np.float))*np.array(output[2][idx])
	masks = np.argmax(masks, axis=0)
	
	for vc_idx in range(0, len(valid_categories)):
		idx = masks == vc_idx + 1
		masks_ = np.zeros((img.shape[0], img.shape[1],3))
		masks_[idx,0] = colors[vc_idx][0]; masks_[idx,1] = colors[vc_idx][1]; masks_[idx,2] = colors[vc_idx][2]
		masks_colored.append(masks_)
	masks_colored = np.array(masks_colored).sum(axis=0)
	cv2.imwrite(output_file,masks_colored)




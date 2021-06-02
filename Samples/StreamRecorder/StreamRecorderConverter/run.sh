#!/bin/sh
RECORDING_PATH=$1
pinhole_ext='/pinhole_projection'
PINHOLE_PATH=$RECORDING_PATH'/pinhole_projection'
echo $PINHOLE_PATH
python convert_images.py --recording_path $RECORDING_PATH
#python run_detector.py --recording_path $RECORDING_PATH --valid_categories "tv,couch,dining table,chair"
python save_pclouds.py --recording_path $RECORDING_PATH --align_mode PV
python save_pclouds.py --recording_path $RECORDING_PATH --align_mode labels
python gen_lists_for_tsdf_integration.py --pinhole_path $PINHOLE_PATH --align_mode PV
python gen_lists_for_tsdf_integration.py --pinhole_path $PINHOLE_PATH --align_mode labels
python tsdf-integration.py --pinhole_path $PINHOLE_PATH --align_mode PV
python tsdf-integration.py --pinhole_path $PINHOLE_PATH --align_mode labels


# Introduction 
The StreamRecorder app captures and saves to disk the following HoloLens streams: VLC cameras, Long throw depth, AHAT depth, PV (i.e. RGB), head tracking, hand tracking, eye gaze tracking.

# Getting started
After cloning and opening the StreamRecorder solution in Visual Studio, build (ARM64) and deploy.

Do not forget to enable Device Portal and Research Mode, as specified here: https://docs.microsoft.com/en-us/windows/mixed-reality/research-mode

# Use the app
The streams to be captured should be specified at compile time, by modifying appropriately the first lines of AppMain.cpp.
For example:
```
std::vector<ResearchModeSensorType> AppMain::kEnabledRMStreamTypes = { ResearchModeSensorType::DEPTH_LONG_THROW };
std::vector<StreamTypes> AppMain::kEnabledStreamTypes = { StreamTypes::PV };
```

After app deployment, you should see a menu with two buttons, Start and Stop. Push Start to start the capture and Stop when you are done.

# Recorded data
The files saved by the app can be accessed via Device Portal, under:
```
System->FileExplorer->LocalAppData->StreamRecorder->LocalState
```
The app creates one folder per capture.

However, it is possible (and recommended) to use the `StreamRecorderConverter/recorder_console.py` script for data download and automated processing.

To use the recorder console, you can run:
```
python StreamRecorderConverter/recorder_console.py --workspace_path <output_folder>  --dev_portal_username <user> --dev_portal_password <password>
```

then use the`download` command to download from HoloLens to the output folder and then use the `process` command.

# Python postprocessing
To postprocess the recorded data, you can use the python scripts inside the `StreamRecorderConverter` folder.

Requirements: python3 with numpy, opencv-python, open3d.

The app comes with a set of python scripts. 

- To run all the scripts sequentially, launch 
```
./run.sh <path_to_capture_folder>
```

- PV (RGB) frames are saved in raw format. To obtain RGB png images, you can run the `convert_images.py` script:
```
  python convert_images.py --recording_path <path_to_capture_folder>
```
- Install Detectron2. To obtain semantic instance segmentation of RGB images, you can run the `run_detector.py` script:
```
  python run_detector.py --recording_path <path_to_capture_folder> --valid_categories <comma separated list of objects of interest within quotes and without spaces>
```

- To see hand tracking and eye gaze tracking results projected on PV images, you can run:
```
  python project_hand_eye_to_pv.py --recording_path <path_to_capture_folder>
```

- To obtain (colored) point clouds from depth images and save them as ply files, you can run the `save_pclouds.py` script.
```
  python save_pclouds.py --pinhole_path <path_to_pinhole_folder_inside_capture_folder> --align_mode <PV or labels>
```

- To make sure the list of paths is set correctly, run gen_lists_for_tsdf_integration.py
```
  python gen_lists_for_tsdf_integration.py --pinhole_path <path_to_pinhole_folder_inside_capture_folder> --align_mode <PV or labels>
```

- Finally, run tsdf-integration.py
```
  python tsdf-integration.py --pinhole_path <path_to_pinhole_folder_inside_capture_folder> --align_mode <PV or labels>
```
The mesh and point cloud ply files are saved in the working directory



All the point clouds are computed in the world coordinate system, unless the `cam_space` parameter is used. If PV frames were captured, the script will try to color the point clouds accordingly.

- To try our sample showcasing Truncated Signed Distance Function (TSDF) integration with open3d, you can run:
```
  python tsdf-integration.py --pinhole_path <path_to_pinhole_projected_camera>
```

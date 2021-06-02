import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
import glob
import torch
import sys

class MaskRCNNDetector():
    def __init__(self, config_file, valid_categories=None):
        from detectron2.config import get_cfg
        from meshrcnn.config import get_meshrcnn_cfg_defaults
        from detectron2.data import MetadataCatalog
        from detectron2.data.detection_utils import read_image
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.utils.logger import setup_logger
        import torch
        from detectron2.modeling import build_model
        import detectron2.data.transforms as T
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.export import Caffe2Tracer
        import onnx

		#self.valid_categories = ['bench', 'chair', 'couch', 'dining table', 'laptop', 'tv'] #'person', 'potted plant'
        cfg = get_cfg()
        get_meshrcnn_cfg_defaults(cfg)
        cfg.merge_from_file(config_file)
        cfg_onnx = cfg.clone()
        cfg_onnx.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.3
        cfg.freeze()
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.predictor = DefaultPredictor(cfg)
        self.category_names = metadata.get("thing_classes", None)
        self.category_colors = metadata.get("thing_colors", None)
        if valid_categories is None:
            self.valid_categories = self.category_names
        else:
            self.valid_categories = valid_categories
        self.threshold = 0.5
        self.filter_categories = True
        self.filter_scores = True
        self.model = build_model(cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        dummy_input = {}
        # dummy_input['file_name'] = 'test.jpg'
        # dummy_input['height'] = 400
        # dummy_input['width'] = 400
        # dummy_input['image_id'] = 1
        dummy_input['image'] = torch.randn(3, 800, 800)
        first_batch = [dummy_input]
        print(cfg_onnx.MODEL.DEVICE)
        self.tracer = Caffe2Tracer(cfg_onnx, self.model, first_batch)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = 'BGR'

    def export_to_onnx(self, onnx_model_path):
        import torch.onnx
        onnx_model = self.tracer.export_onnx()
        onnx.save(onnx_model, onnx_model_path)

    def run_on_opencv_image(self, img):
        predictions = self.predictor(img)
        instances = predictions["instances"].to(torch.device("cpu"))
        if len(instances) == 0:
            return None, None, None
        masks = instances.pred_masks.cpu().numpy()            
        boxes = instances.pred_boxes.tensor.int().cpu().numpy()
        scores = instances.scores
        labels = instances.pred_classes.int().cpu().numpy()
        classes = instances.pred_classes.cpu()
        if self.category_names:
            _names = np.array([self.category_names[i] for i in classes])
        
        
        # score based validation
        valid_scores = scores > self.threshold
        # label based validation
        valid_categories = []
        for _name in _names:
            if _name in self.valid_categories:
                valid_categories.append(True)
            else:
                 valid_categories.append(False)              
        valid_ids = valid_scores.numpy() * np.array(valid_categories)
        
        #valid_ids = scores > self.threshold
        masks = masks[valid_ids,:]
        boxes = boxes[valid_ids,:]
        scores = scores[valid_ids]
        _names = _names[valid_ids]
        if len(scores) == 0:
            return None, None, None

        return masks, _names, scores
        
    def run_on_opencv_images(self, images_tensor):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                images_tensor = images_tensor[:, :, :, ::-1]
            inputs_list = []                
            for original_image in images_tensor:
                height, width = original_image.shape[:2]
                image = self.transform_gen.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs = {"image": image, "height": height, "width": width}
                inputs_list.append(inputs)
            predictions = self.model(inputs_list)
        masks_ms = []
        boxes_ms = []
        scores_ms = []
        names_ms = []                
        colors_ms = []
        for pred in predictions:
            instances = pred["instances"].to(torch.device("cpu"))
            if len(instances) == 0:
                continue
            
            masks = instances.pred_masks.cpu().numpy()
            boxes = instances.pred_boxes.tensor.int().cpu().numpy()
            scores = instances.scores
            labels = instances.pred_classes.int().cpu().numpy()
            classes = instances.pred_classes.cpu()
            if self.category_names:
                _names = np.array([self.category_names[i] for i in classes])
            if self.category_colors:
                _colors = np.array([self.category_colors[i] for i in classes])
            # score based validation
            valid_scores = scores > self.threshold
            # label based validation
            valid_categories = []
            for _name in _names:
                if _name in self.valid_categories:
                    valid_categories.append(True)
                else:
                    valid_categories.append(False)
            valid_positions = []
            for mask in masks:
                if mask[int(height/2), int(width/2)] == True:
                    valid_positions.append(True)
                else:
                    valid_positions.append(False)
            
            valid_ids = valid_scores.numpy() * np.array(valid_categories) * np.array(valid_positions)
            
            masks = masks[valid_ids,:]
            boxes = boxes[valid_ids,:]
            scores = scores[valid_ids]
            _names = _names[valid_ids]
            _colors = _colors[valid_ids]
            
            if len(scores) > 1:
                id = np.argmax(scores)
                masks = masks[id]
                boxes = boxes[id]
                scores = scores[id]
                _names = _names[id]
                _colors = _colors[id]
            elif len(scores) == 0:
                    continue

            masks_ms.append(masks)
            names_ms.append(_names)
            colors_ms.append(_colors)
            scores_ms.append(scores)

        return masks_ms, names_ms, scores_ms, colors_ms
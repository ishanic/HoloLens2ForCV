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
        from detectron2.data import MetadataCatalog
        from detectron2.data.detection_utils import read_image
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.utils.logger import setup_logger
        import torch
        from detectron2.modeling import build_model
        import detectron2.data.transforms as T
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.export import Caffe2Tracer
        
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
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
        dummy_input['image'] = torch.randn(3, 800, 800)
        first_batch = [dummy_input]
        self.tracer = Caffe2Tracer(cfg_onnx, self.model, first_batch)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = 'BGR'

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
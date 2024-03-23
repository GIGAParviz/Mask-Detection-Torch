
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn

def build_fasterrcnn_model(device):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 4
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model.to(device)



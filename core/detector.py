import torch
from config import DETECT_CLASSES, CONFIDENCE_THRESH

class Detector:
    def __init__(self):
        # load local YOLOv5 model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = torch.load("yolov8n.pt", map_location=self.device)
        self.model = self.model['model'].float().fuse().eval().to(self.device)

    def detect(self, frame):
        # BGR numpy -> tensor
        img = torch.from_numpy(frame).to(self.device)

        # HWC -> CHW
        img = img.permute(2, 0, 1).float()

        # normalize
        img /= 255.0

        # add batch
        img = img.unsqueeze(0)

        # inference
        with torch.no_grad():
            pred = self.model(img)[0]

        detections = []

        # NMS (YOLOv5 style)
        pred = non_max_suppression(pred, CONFIDENCE_THRESH, 0.45)[0]

        if pred is not None:
            for *xyxy, conf, cls in pred:
                cls = int(cls)
                conf = float(conf)

                if cls in DETECT_CLASSES:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf])

        return detections

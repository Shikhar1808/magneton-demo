import torch
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from config import DETECT_CLASSES, CONFIDENCE_THRESH


class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # lightweight detection model
        self.model = ssdlite320_mobilenet_v3_large(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def detect(self, frame):
        img = self.transform(frame).to(self.device)

        with torch.no_grad():
            outputs = self.model([img])[0]

        detections = []

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        for box, score, label in zip(boxes, scores, labels):
            conf = float(score)

            if conf < CONFIDENCE_THRESH:
                continue

            cls = int(label)

            if cls in DETECT_CLASSES:
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append([x1, y1, x2, y2, conf])

        return detections

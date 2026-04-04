# from ultralytics import YOLO
# from config import DETECT_CLASSES, CONFIDENCE_THRESH

# class Detector:
#     def __init__(self):
#         self.model = YOLO('yolov8n.pt')

#     def detect(self, frame):
#         results = self.model(frame, imgsz=416)[0]

#         detections = []
#         for box in results.boxes:
#             cls = int(box.cls[0])
#             conf = float(box.conf[0])

#             if cls in DETECT_CLASSES and conf > CONFIDENCE_THRESH:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append([x1, y1, x2, y2, conf])

#         return detections
    

#
from jetson_inference import detectNet
import jetson_utils

class Detector:
    def __init__(self):
        # lightweight + fast
        self.net = detectNet("ssd-mobilenet-v2", threshold=0.5)
        self.person_class_id = 1

    def detect(self, frame):
        # convert numpy → CUDA
        cuda_img = jetson_utils.cudaFromNumpy(frame)

        detections = self.net.Detect(cuda_img)

        results = []
        for d in detections:
            if int(d.ClassID) != self.person_class_id:
                continue

            x1 = int(d.Left)
            y1 = int(d.Top)
            x2 = int(d.Right)
            y2 = int(d.Bottom)
            conf = float(d.Confidence)

            results.append([x1, y1, x2, y2, conf])

        return results
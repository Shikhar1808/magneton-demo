# import supervision as sv
# import numpy as np

# class Tracker:
#     def __init__(self,cam_id):
#         self.tracker = sv.ByteTrack()
#         self.cam_id = cam_id
#         self.seen_ids = set()

#     def update(self, detections):
#         if len(detections) == 0:
#             return sv.Detections.empty(), []

#         xyxy = np.array([d[:4] for d in detections])
#         confidence = np.array([d[4] for d in detections])

#         tracked = self.tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=confidence))
        
#         new_ids = []
#         if tracked.tracker_id is not None:
#             for tid in tracked.tracker_id:
#                 if tid not in self.seen_ids:
#                     self.seen_ids.add(tid)
#                     new_ids.append(tid)

#         return tracked, new_ids


import numpy as np

class Tracker:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.next_id = 0
        self.objects = {}     # id → centroid
        self.seen_ids = set()

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections):
        new_ids = []
        tracked = []

        updated_objects = {}

        for det in detections:
            box = det[:4]
            centroid = self._get_centroid(box)

            assigned_id = None

            # match with existing objects
            for obj_id, prev_centroid in self.objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))

                if dist < 60:   # tune this
                    assigned_id = obj_id
                    break

            if assigned_id is not None:
                updated_objects[assigned_id] = centroid
                tracked.append((*box, assigned_id))
            else:
                obj_id = self.next_id
                self.next_id += 1

                updated_objects[obj_id] = centroid
                tracked.append((*box, obj_id))

                if obj_id not in self.seen_ids:
                    self.seen_ids.add(obj_id)
                    new_ids.append(obj_id)

        self.objects = updated_objects

        return tracked, new_ids
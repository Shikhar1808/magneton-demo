from core.detector import Detector
from core.tracker import Tracker
from core.recorder import SingleFileRecorder
from core.alert import AlertManager

from utils.drawing import *
from utils.helpers import get_label
from utils.logger import setup_logging

from config import *

import cv2
import time
import signal
import sys

log = setup_logging()

running = True

def stop(sig, frame):
    global running
    running = False


def main():
    global running

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    num_cams = len(CAMERA_SOURCES)

    detector = Detector()
    trackers = [Tracker(i) for i in range(num_cams)]
    alert_mgr = AlertManager()
    recorder = SingleFileRecorder(num_cams, FPS)

    caps = [cv2.VideoCapture(src) for src in CAMERA_SOURCES]

    for i, c in enumerate(caps):
        log.info("Camera %d opened: %s", i, c.isOpened())

    frame_time = 1.0 / FPS

    try:
        while running:
            start = time.time()

            live_tiles = []
            active_frames = {}
            idle_ids = []

            for i, cap in enumerate(caps):

                if not cap.isOpened():
                    idle_ids.append(i)
                    live_tiles.append(make_idle_tile(i))
                    continue

                ret, frame = cap.read()

                # loop video files
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                    if not ret:
                        idle_ids.append(i)
                        live_tiles.append(make_idle_tile(i))
                        continue

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                dets = detector.detect(frame)
                print("cam", i, "detections:", len(dets))

                tracks, new_ids = trackers[i].update(dets)

                annotated = draw_active_tile(frame.copy(), tracks, i)
                live_tiles.append(annotated)

                if len(tracks) > 0:
                    active_frames[i] = annotated
                else:
                    idle_ids.append(i)

                if new_ids:
                    alert_mgr.trigger(i, get_label(i), new_ids, annotated)

            recorder.write(active_frames, idle_ids)

            # FPS control
            elapsed = time.time() - start
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        log.info("Stopping recorder...")

        recorder.close()

        for cap in caps:
            cap.release()

        log.info("Recording finalized")


if __name__ == "__main__":
    main()

import os
import subprocess
from datetime import datetime
import numpy as np
from typing import Tuple, Dict, List

from config import FRAME_WIDTH, FRAME_HEIGHT, RECORD_COLS, RECORD_ROWS, RECORDING_DIR, CAMERA_LABELS
from utils.drawing import make_idle_tile, build_grid, stamp_rec_header
from utils.logger import setup_logging

log = setup_logging()


def _label(i: int) -> str:
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM {i}"


def _grid_dims(n: int) -> Tuple[int, int]:
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


class SingleFileRecorder:
    def __init__(self, num_cams: int, fps: int):
        self.fps = fps
        self.num_cams = num_cams

        if RECORD_COLS and RECORD_ROWS:
            self.cols, self.rows = RECORD_COLS, RECORD_ROWS
        else:
            self.cols, self.rows = _grid_dims(num_cams)

        self.canvas_w = FRAME_WIDTH * self.cols
        self.canvas_h = FRAME_HEIGHT * self.rows

        os.makedirs(RECORDING_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"{RECORDING_DIR}/surveillance_{ts}.mp4"

        log.info(
            "Recording file: %s | canvas: %dx%d (%dx%d tiles)",
            self.path,
            self.canvas_w,
            self.canvas_h,
            self.cols,
            self.rows,
        )

        # IMPORTANT: enforce FPS
        cmd = [
            "ffmpeg",
            "-y",

            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.canvas_w}x{self.canvas_h}",
            "-r", str(self.fps),      # input FPS
            "-i", "-",

            "-an",

            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",

            "-pix_fmt", "yuv420p",

            "-vsync", "1",
            "-r", str(self.fps),      # output FPS

            "-movflags", "+faststart",
            self.path,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        log.info("SingleFileRecorder started - FFmpeg PID %d", self._proc.pid)

    def write(self, all_frames: Dict[int, np.ndarray], idle_cam_ids: List[int]):

        if self._proc is None:
            return

        tiles = []
        active_labels = []

        for cam_id in range(self.num_cams):
            if cam_id in all_frames:
                tiles.append(all_frames[cam_id])
                active_labels.append(_label(cam_id))
            else:
                tiles.append(make_idle_tile(cam_id))

        while len(tiles) < self.cols * self.rows:
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            tiles.append(blank)

        canvas = build_grid(tiles, self.cols)
        canvas = stamp_rec_header(canvas, active_labels)

        try:
            self._proc.stdin.write(canvas.tobytes())
        except Exception:
            err = self._proc.stderr.read().decode()
            log.error("FFmpeg crashed: %s", err)
            self._proc = None

    def close(self):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait()
            except Exception:
                pass

            log.info("Recording saved: %s", self.path)

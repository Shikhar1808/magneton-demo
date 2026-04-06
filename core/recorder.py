import os
import subprocess
from datetime import datetime
import numpy as np
import cv2
from dotenv import load_dotenv
load_dotenv()
from typing import Tuple, Dict, List

from config import FRAME_WIDTH, FRAME_HEIGHT, RECORD_COLS, RECORD_ROWS, RECORDING_DIR, CAMERA_LABELS
from utils.drawing import make_idle_tile, build_grid, stamp_rec_header
from utils.logger import setup_logging

log = setup_logging()


def _get_rtsp_url() -> str:
    return os.environ.get("RTSP_PUSH_URL", "rtsp://localhost:8554/mystream").strip()


RTSP_SERVER_URL = _get_rtsp_url()


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
        self.rtsp_url = RTSP_SERVER_URL

        log.info(
            "RTSP push target: %s  |  file: %s  |  canvas: %dx%d (%dx%d tiles)",
            self.rtsp_url, self.path, self.canvas_w, self.canvas_h, self.cols, self.rows,
        )

        tee_targets = "|".join([
            f"[f=mp4:movflags=faststart+frag_keyframe]{self.path}",
            f"[f=rtsp:rtsp_transport=tcp]{self.rtsp_url}",
        ])

        cmd = [
            "ffmpeg", "-y",

            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-strict", "experimental",

            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.canvas_w}x{self.canvas_h}",
            "-r", str(fps),
            "-i", "-",

            "-an",

            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-g", str(fps),
            "-sc_threshold", "0",
            "-bf", "0",

            "-f", "tee",
            "-map", "0:v",
            tee_targets,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        log.info("SingleFileRecorder started — FFmpeg PID %d", self._proc.pid)

    def write(
        self,
        all_frames: Dict[int, np.ndarray],
        idle_cam_ids: List[int],
    ) -> None:

        # ffmpeg died
        if self._proc is None or self._proc.stdin is None:
            return

        tiles: List[np.ndarray] = []
        active_labels: List[str] = []

        for cam_id in range(self.num_cams):

            frame = all_frames.get(cam_id, None)

            # camera missing / disconnected
            if frame is None:
                tiles.append(make_idle_tile(cam_id))
                continue

            # resize safety
            try:
                if (
                    frame.shape[1] != FRAME_WIDTH or
                    frame.shape[0] != FRAME_HEIGHT
                ):
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except Exception:
                frame = make_idle_tile(cam_id)

            tiles.append(frame)
            active_labels.append(_label(cam_id))

        # fill empty grid
        total = self.cols * self.rows
        while len(tiles) < total:
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            blank[:] = (10, 10, 10)
            tiles.append(blank)

        canvas = build_grid(tiles, self.cols)
        canvas = stamp_rec_header(canvas, active_labels)

        # safe write (headless safe)
        try:
            self._proc.stdin.write(canvas.tobytes())
        except (BrokenPipeError, OSError):
            try:
                err = self._proc.stderr.read().decode(errors="ignore")
                log.error("FFmpeg crashed: %s", err)
            except Exception:
                pass
            self._proc = None

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()

                self._proc.wait(timeout=15)
                log.info("SingleFileRecorder closed | file: %s", self.path)

            except Exception as exc:
                log.warning("FFmpeg close warning: %s", exc)

            finally:
                self._proc = None

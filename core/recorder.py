import os
import subprocess
from datetime import datetime
import numpy as np
import cv2

from config import FRAME_WIDTH, FRAME_HEIGHT, RECORD_COLS, RECORD_ROWS, RECORDING_DIR, CAMERA_LABELS
from utils.drawing import make_idle_tile, build_grid, stamp_rec_header
from utils.logger import setup_logging

log = setup_logging()

RTSP_SERVER_URL = os.environ.get("RTSP_SERVER_URL", "rtsp://localhost:8554/live")

def _label(i: int) -> str:
    return CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"CAM {i}"

def _grid_dims(n: int) -> tuple[int, int]:
    """Return (cols, rows) for a grid that fits n cameras."""
    import math
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows

class SingleFileRecorder:
    """
    Writes ONE video file for the entire session with a FIXED canvas size,
    and simultaneously streams via RTSP.

    Canvas = RECORD_COLS × RECORD_ROWS tiles  (each FRAME_WIDTH × FRAME_HEIGHT).

    Every frame written to the file contains:
      • Active cameras  → annotated feed (bounding boxes, label, timestamp)
      • Idle cameras    → dark placeholder tile (camera name + "NO DETECTION" + timestamp)
      • Top-right HUD   → ● REC + wall-clock time + active camera list

    Because the canvas size never changes, only one FFmpeg process is ever
    needed for the whole session — no segments, no gaps.
    """

    def __init__(self, num_cams: int, fps: int):
        self.fps      = fps
        self.num_cams = num_cams

        # Derive canvas grid dimensions
        if RECORD_COLS and RECORD_ROWS:
            self.cols, self.rows = RECORD_COLS, RECORD_ROWS
        else:
            self.cols, self.rows = _grid_dims(num_cams)

        self.canvas_w = FRAME_WIDTH  * self.cols
        self.canvas_h = FRAME_HEIGHT * self.rows

        os.makedirs(RECORDING_DIR, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = f"{RECORDING_DIR}/surveillance_{ts}.mp4"
        self.rtsp_url = RTSP_SERVER_URL

        # tee muxer: encode ONCE, output to both MP4 file and RTSP stream
        tee_targets = "|".join([
            f"[f=mp4]{self.path}",
            f"[f=rtsp:rtsp_transport=tcp]{self.rtsp_url}",
        ])

        # cmd = [
        #     "ffmpeg", "-y",
        #     "-f", "rawvideo",
        #     "-vcodec", "rawvideo",
        #     "-pix_fmt", "bgr24",
        #     "-s", f"{self.canvas_w}x{self.canvas_h}",
        #     "-r", str(fps),
        #     "-i", "-",

        #     "-an",

        #     "-vcodec", "libx264",
        #     "-preset", "veryfast",
        #     "-tune", "zerolatency",
        #     "-crf", "23",
        #     "-pix_fmt", "yuv420p",
        #     "-g", str(fps * 2),

        #     # tee muxer writes to both outputs in one pass
        #     "-f", "tee",
        #     "-map", "0:v",
        #     tee_targets,
        # ]

        cmd = [
            "ffmpeg", "-y",

            # ── Input: tell FFmpeg to treat this as a live source ──────────────────
            "-fflags", "nobuffer",          # don't buffer input frames
            "-flags", "low_delay",          # enable low-delay mode globally
            "-strict", "experimental",

            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.canvas_w}x{self.canvas_h}",
            "-r", str(fps),
            "-i", "-",

            "-an",

            # ── Encoding: optimise for latency over quality ─────────────────────────
            "-vcodec", "libx264",
            "-preset", "ultrafast",         # was "veryfast" — biggest single latency win
            "-tune", "zerolatency",
            "-crf", "28",                   # slightly lower quality, much faster encode
            "-pix_fmt", "yuv420p",
            "-g", str(fps),                 # keyframe every 1s instead of 2s — faster seek/join
            "-sc_threshold", "0",           # disable scene-change keyframes (keeps GOP stable)
            "-bf", "0",                     # no B-frames — they add latency

            # ── Output buffering: flush aggressively ───────────────────────────────
            "-flush_packets", "1",          # flush every packet immediately
            "-fflags", "+flush_packets",

            "-f", "tee",
            "-map", "0:v",
            tee_targets,
        ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stderr=subprocess.PIPE)

        log.info(
            "SingleFileRecorder started | canvas: %dx%d (%dx%d tiles) | file: %s | RTSP: %s",
            self.canvas_w, self.canvas_h, self.cols, self.rows, self.path, self.rtsp_url
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def write(self,
              all_frames:    dict[int, np.ndarray],
              idle_cam_ids:  list[int]):               # cams with NO detections this frame
        """
        Build the full recording canvas and push it to FFmpeg.

        all_frames    : {cam_id: frame} for cameras that HAVE detections
        idle_cam_ids  : cam IDs with no detections (get placeholder tile)
        """
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

        # Pad remaining slots (if canvas has more slots than cameras)
        while len(tiles) < self.cols * self.rows:
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            blank[:] = (10, 10, 10)
            tiles.append(blank)

        canvas = build_grid(tiles, self.cols)
        canvas = stamp_rec_header(canvas, active_labels)

        try:
            self._proc.stdin.write(canvas.tobytes())
        except (BrokenPipeError, OSError) as e:
            log.error("FFmpeg pipe error (%s): %s", e, self._proc.stderr.read().decode(errors="replace"))
            self._proc = None

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=15)
                log.info("SingleFileRecorder closed | file: %s | RTSP: %s", self.path, self.rtsp_url)
            except Exception as exc:
                log.warning("FFmpeg close warning: %s", exc)
            self._proc = None
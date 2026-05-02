"""
Traffic Violation Detection System - Final Version
==================================================

Features:
- YOLOv8 vehicle detection
- Centroid-based tracking
- Signal jump detection
- Wrong-way detection
- Basic helmet violation placeholder
- ROI-based automatic traffic signal color detection
- Evidence clip with moving rectangle around violating vehicle
- Cropped violation image
- Risk score
- CSV report with UTF-8 support
"""

import cv2
import csv
import time
import math
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed. Running in demo mode.")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def bottom_center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


@dataclass
class TrackedVehicle:
    vehicle_id: int
    class_name: str
    bbox: tuple
    center: tuple
    trail: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=120))
    frames_seen: int = 0
    last_seen: int = 0
    violations: list = field(default_factory=list)
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=5))

    @property
    def velocity(self):
        if len(self.trail) < 2:
            return (0.0, 0.0)
        dx = self.trail[-1][0] - self.trail[-2][0]
        dy = self.trail[-1][1] - self.trail[-2][1]
        return (float(dx), float(dy))

    @property
    def avg_velocity(self):
        if not self.velocity_history:
            return (0.0, 0.0)
        vx = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
        vy = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)
        return (vx, vy)


@dataclass
class ViolationEvent:
    timestamp: str
    vehicle_id: int
    vehicle_type: str
    violation_type: str
    frame_number: int
    bbox: tuple
    risk_score: int = 0
    snapshot_path: str = ""
    evidence_clip_path: str = ""


class Config:
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.45

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    MAX_DISAPPEARED = 30
    MAX_DISTANCE = 120
    MIN_FRAMES_TO_TRACK = 3

    ALLOWED_DIRECTION_DEGREES = 0.0
    DIRECTION_TOLERANCE_DEGREES = 75.0
    MIN_SPEED_PX = 2.5

    SIGNAL_CYCLE_FRAMES = 150
    SIGNAL_MODE = "cycle"   # cycle or roi
    SIGNAL_ROI = None       # (x1, y1, x2, y2)
    TREAT_YELLOW_AS_STOP = True

    OUTPUT_DIR = "output"
    VIOLATIONS_CSV = "violations.csv"

    SHOW_TRAILS = True
    TRAIL_LENGTH = 20

    CROP_PADDING = 35
    EVIDENCE_CLIP_SECONDS = 4

    MIN_CROP_WIDTH = 45
    MIN_CROP_HEIGHT = 45
    EDGE_MARGIN = 2

    DUPLICATE_IOU_THRESHOLD = 0.75
    DUPLICATE_FRAME_WINDOW = 35

    RISK_SCORE = {
        "signal_jump": 5,
        "wrong_way": 8,
        "helmet_violation": 3,
    }


class VehicleDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(cfg.MODEL_PATH)
                log.info(f"Loaded YOLO model: {cfg.MODEL_PATH}")
            except Exception as e:
                log.warning(f"Could not load YOLO model ({e}). Using demo mode.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            return []

        results = self.model(
            frame,
            conf=self.cfg.CONFIDENCE_THRESHOLD,
            iou=self.cfg.IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in self.cfg.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=cls,
                    class_name=self.cfg.VEHICLE_CLASSES[cls],
                )
            )

        return detections


class CentroidTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.next_id = 1
        self.vehicles = {}
        self._disappeared = {}

    def update(self, detections: list[Detection], frame_no: int):
        for vid in list(self._disappeared):
            self._disappeared[vid] += 1

        if not detections:
            self._remove_old()
            return self.vehicles

        if not self.vehicles:
            for det in detections:
                self._register(det, frame_no)
            return self.vehicles

        input_centers = [det.center for det in detections]
        existing_ids = list(self.vehicles.keys())
        existing_centers = [self.vehicles[vid].center for vid in existing_ids]

        distances = []
        for i, old_center in enumerate(existing_centers):
            for j, new_center in enumerate(input_centers):
                dist = math.hypot(old_center[0] - new_center[0], old_center[1] - new_center[1])
                distances.append((dist, existing_ids[i], j))

        distances.sort(key=lambda x: x[0])

        used_vehicle_ids = set()
        used_detection_indexes = set()

        for dist, vid, det_idx in distances:
            if vid in used_vehicle_ids or det_idx in used_detection_indexes:
                continue

            if dist > self.cfg.MAX_DISTANCE:
                continue

            self._update_vehicle(vid, detections[det_idx], frame_no)
            used_vehicle_ids.add(vid)
            used_detection_indexes.add(det_idx)

        for idx, det in enumerate(detections):
            if idx not in used_detection_indexes:
                self._register(det, frame_no)

        self._remove_old()
        return self.vehicles

    def _register(self, det: Detection, frame_no: int):
        vid = self.next_id
        self.next_id += 1

        vehicle = TrackedVehicle(
            vehicle_id=vid,
            class_name=det.class_name,
            bbox=det.bbox,
            center=det.center,
        )

        vehicle.trail.append(det.center)
        vehicle.bbox_history.append((frame_no, det.bbox))
        vehicle.frames_seen = 1
        vehicle.last_seen = frame_no

        self.vehicles[vid] = vehicle
        self._disappeared[vid] = 0

    def _update_vehicle(self, vid: int, det: Detection, frame_no: int):
        vehicle = self.vehicles[vid]

        vehicle.bbox = det.bbox
        vehicle.center = det.center
        vehicle.trail.append(det.center)
        vehicle.bbox_history.append((frame_no, det.bbox))
        vehicle.velocity_history.append(vehicle.velocity)
        vehicle.frames_seen += 1
        vehicle.last_seen = frame_no

        self._disappeared[vid] = 0

    def _remove_old(self):
        for vid in list(self._disappeared):
            if self._disappeared[vid] > self.cfg.MAX_DISAPPEARED:
                self.vehicles.pop(vid, None)
                self._disappeared.pop(vid, None)


class TrafficSignal:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cycle = cfg.SIGNAL_CYCLE_FRAMES
        self.state = "green"
        self.debug_counts = {"red": 0, "yellow": 0, "green": 0}

    def update(self, frame_no: int, frame: np.ndarray = None):
        if self.cfg.SIGNAL_MODE == "roi" and frame is not None and self.cfg.SIGNAL_ROI:
            self.state = self._detect_signal_from_roi(frame)
            return

        phase = (frame_no // self.cycle) % 2
        self.state = "red" if phase == 1 else "green"

    def _detect_signal_from_roi(self, frame: np.ndarray) -> str:
        x1, y1, x2, y2 = self.cfg.SIGNAL_ROI
        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return self.state

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return self.state

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 80, 100]), np.array([180, 255, 255]))
        red_mask = red_mask1 | red_mask2

        yellow_mask = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([38, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 60, 80]), np.array([90, 255, 255]))

        red_pixels = int(cv2.countNonZero(red_mask))
        yellow_pixels = int(cv2.countNonZero(yellow_mask))
        green_pixels = int(cv2.countNonZero(green_mask))

        self.debug_counts = {"red": red_pixels, "yellow": yellow_pixels, "green": green_pixels}

        counts = {"red": red_pixels, "yellow": yellow_pixels, "green": green_pixels}
        best_color = max(counts, key=counts.get)
        best_count = counts[best_color]

        min_pixels = max(6, int(roi.shape[0] * roi.shape[1] * 0.0015))

        if best_count < min_pixels:
            return self.state

        return best_color

    @property
    def is_stop_signal(self):
        if self.state == "red":
            return True
        if self.state == "yellow" and self.cfg.TREAT_YELLOW_AS_STOP:
            return True
        return False

    @property
    def color_bgr(self):
        if self.state == "red":
            return (0, 0, 220)
        if self.state == "yellow":
            return (0, 220, 220)
        return (0, 200, 0)


class ViolationDetector:
    def __init__(self, cfg: Config, stop_line_y: int, frame_width: int):
        self.cfg = cfg
        self.stop_line_y = stop_line_y
        self.frame_width = frame_width
        self._triggered = defaultdict(set)

    def check(self, vehicle: TrackedVehicle, signal: TrafficSignal):
        if vehicle.frames_seen < self.cfg.MIN_FRAMES_TO_TRACK:
            return []

        violations = []
        vid = vehicle.vehicle_id

        if "signal_jump" not in self._triggered[vid]:
            if self._check_signal_jump(vehicle, signal):
                violations.append("signal_jump")
                self._triggered[vid].add("signal_jump")

        if "wrong_way" not in self._triggered[vid]:
            if self._check_wrong_way(vehicle):
                violations.append("wrong_way")
                self._triggered[vid].add("wrong_way")

        if "helmet_violation" not in self._triggered[vid]:
            if self._check_helmet(vehicle):
                violations.append("helmet_violation")
                self._triggered[vid].add("helmet_violation")

        return violations

    def _check_signal_jump(self, vehicle: TrackedVehicle, signal: TrafficSignal):
        if not signal.is_stop_signal or len(vehicle.trail) < 2:
            return False

        prev_y = vehicle.trail[-2][1]
        curr_y = vehicle.trail[-1][1]

        crossed = prev_y < self.stop_line_y <= curr_y
        moving_down = curr_y - prev_y > 2

        return crossed and moving_down

    def _check_wrong_way(self, vehicle: TrackedVehicle):
        vx, vy = vehicle.avg_velocity
        speed = math.hypot(vx, vy)

        if speed < self.cfg.MIN_SPEED_PX:
            return False

        angle = math.degrees(math.atan2(vy, vx))
        allowed = self.cfg.ALLOWED_DIRECTION_DEGREES
        diff = abs((angle - allowed + 180) % 360 - 180)

        return diff > (180 - self.cfg.DIRECTION_TOLERANCE_DEGREES)

    def _check_helmet(self, vehicle: TrackedVehicle):
        if vehicle.class_name != "motorcycle":
            return False

        x1, y1, x2, y2 = vehicle.bbox
        box_width = x2 - x1
        box_height = y2 - y1

        if vehicle.frames_seen < 5:
            return False

        if box_width < 70 or box_height < 65:
            return False

        aspect_ratio = box_width / max(box_height, 1)

        if aspect_ratio < 0.35 or aspect_ratio > 3.2:
            return False

        return True


class OutputManager:
    VIOLATION_LABELS = {
        "signal_jump": "Signal Jump",
        "wrong_way": "Wrong Way",
        "helmet_violation": "Helmet Violation",
    }

    VIOLATION_COLORS = {
        "signal_jump": (0, 0, 255),
        "wrong_way": (0, 165, 255),
        "helmet_violation": (255, 0, 200),
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.OUTPUT_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for violation_type in self.VIOLATION_LABELS:
            (self.out_dir / violation_type).mkdir(exist_ok=True)

        (self.out_dir / "evidence_clips").mkdir(exist_ok=True)

        self.csv_path = self.out_dir / cfg.VIOLATIONS_CSV
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.violation_count = defaultdict(int)
        self.vehicle_risk_scores = defaultdict(int)
        self.saved_events = set()
        self.recent_saved = []

        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "frame",
                "vehicle_id",
                "vehicle_type",
                "violation_type",
                "risk_score",
                "snapshot_path",
                "evidence_clip_path",
            ])

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _is_duplicate_saved(self, event: ViolationEvent):
        key = (event.vehicle_id, event.violation_type)

        if key in self.saved_events:
            return True

        cleaned = []
        for frame_no, violation_type, bbox in self.recent_saved:
            if event.frame_number - frame_no <= self.cfg.DUPLICATE_FRAME_WINDOW:
                cleaned.append((frame_no, violation_type, bbox))

                if violation_type == event.violation_type:
                    iou = self._bbox_iou(event.bbox, bbox)
                    if iou >= self.cfg.DUPLICATE_IOU_THRESHOLD:
                        return True

        self.recent_saved = cleaned
        return False

    def _is_good_vehicle_bbox(self, frame, bbox, vehicle_type):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < self.cfg.MIN_CROP_WIDTH or box_h < self.cfg.MIN_CROP_HEIGHT:
            return False

        edge = self.cfg.EDGE_MARGIN
        if x1 <= edge or y1 <= edge or x2 >= w - edge or y2 >= h - edge:
            return False

        aspect = box_w / max(box_h, 1)
        if aspect < 0.35 or aspect > 3.2:
            return False

        if vehicle_type == "motorcycle":
            if box_w < 60 or box_h < 55:
                return False

        return True

    def _crop_vehicle(self, frame, bbox, vehicle_type):
        if not self._is_good_vehicle_bbox(frame, bbox, vehicle_type):
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = self.cfg.CROP_PADDING

        x1 = max(0, int(x1) - pad)
        y1 = max(0, int(y1) - pad)
        x2 = min(w, int(x2) + pad)
        y2 = min(h, int(y2) + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    def _draw_evidence_box(self, frame, bbox, event):
        vis = frame.copy()
        h, w = vis.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        color = self.VIOLATION_COLORS.get(event.violation_type, (0, 0, 255))
        label = self.VIOLATION_LABELS.get(event.violation_type, event.violation_type)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        text = f"ID {event.vehicle_id} | {label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.65, 2)

        text_y1 = max(0, y1 - th - 10)
        text_y2 = y1
        cv2.rectangle(vis, (x1, text_y1), (min(w - 1, x1 + tw + 10), text_y2), color, -1)
        cv2.putText(vis, text, (x1 + 5, max(18, y1 - 6)), font, 0.65, (255, 255, 255), 2)

        return vis

    def save_evidence_clip(self, frames_with_bboxes, event, fps, size):
        if not frames_with_bboxes:
            return ""

        clip_name = f"{event.violation_type}_id{event.vehicle_id}_f{event.frame_number}.mp4"
        clip_path = self.out_dir / "evidence_clips" / clip_name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, size)

        if not writer.isOpened():
            log.warning(f"Could not create evidence clip: {clip_path}")
            return ""

        written_frames = 0
        for frame, bbox in frames_with_bboxes:
            vis = self._draw_evidence_box(frame, bbox, event)
            writer.write(vis)
            written_frames += 1

        writer.release()

        if written_frames == 0:
            try:
                clip_path.unlink(missing_ok=True)
            except Exception:
                pass
            return ""

        return str(clip_path)

    def save_violation(self, frame, event, evidence_frames=None, fps=25.0, size=None):
        if self._is_duplicate_saved(event):
            return False

        crop = self._crop_vehicle(frame, event.bbox, event.vehicle_type)

        if crop is None:
            log.warning(f"Skipped {event.violation_type} for ID={event.vehicle_id}: partial/invalid crop")
            return False

        key = (event.vehicle_id, event.violation_type)
        self.saved_events.add(key)
        self.recent_saved.append((event.frame_number, event.violation_type, event.bbox))

        fname = f"{event.violation_type}_id{event.vehicle_id}_f{event.frame_number}.jpg"
        image_path = self.out_dir / event.violation_type / fname

        if not cv2.imwrite(str(image_path), crop):
            log.warning(f"Could not save image: {image_path}")
            return False

        event.snapshot_path = str(image_path)

        risk_value = self.cfg.RISK_SCORE.get(event.violation_type, 0)
        self.vehicle_risk_scores[event.vehicle_id] += risk_value
        event.risk_score = self.vehicle_risk_scores[event.vehicle_id]

        if evidence_frames and size:
            event.evidence_clip_path = self.save_evidence_clip(evidence_frames, event, fps, size)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                event.timestamp,
                event.frame_number,
                event.vehicle_id,
                event.vehicle_type,
                self.VIOLATION_LABELS.get(event.violation_type, event.violation_type),
                event.risk_score,
                event.snapshot_path,
                event.evidence_clip_path,
            ])

        self.violation_count[event.violation_type] += 1
        log.info(
            f"[VIOLATION] {self.VIOLATION_LABELS[event.violation_type]} | "
            f"ID={event.vehicle_id} | {event.vehicle_type} | Risk={event.risk_score}"
        )

        return True

    def init_video_writer(self, path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(path, fourcc, fps, size)

        if not self.video_writer.isOpened():
            log.warning(f"Could not create output video: {path}")
            self.video_writer = None

    def write_frame(self, frame):
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()


class Annotator:
    CLASS_COLORS = {
        "car": (50, 205, 50),
        "motorcycle": (30, 144, 255),
        "bus": (255, 215, 0),
        "truck": (148, 0, 211),
    }

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cfg: Config, out: OutputManager):
        self.cfg = cfg
        self.out = out

    def draw_scene(self, frame, vehicles, signal, stop_line_y, frame_no, fps):
        vis = frame.copy()
        h, w = vis.shape[:2]

        cv2.line(vis, (0, stop_line_y), (w, stop_line_y), (255, 255, 255), 2)
        cv2.putText(vis, "STOP LINE", (10, stop_line_y - 8), self.FONT, 0.55, (255, 255, 255), 1)

        self._draw_signal(vis, signal)
        self._draw_signal_roi(vis)

        for vehicle in vehicles.values():
            self._draw_vehicle(vis, vehicle)

        self._draw_hud(vis, frame_no, fps, vehicles)
        return vis

    def _draw_signal(self, frame, signal):
        h, w = frame.shape[:2]
        cx, cy, r = w - 60, 60, 22

        cv2.circle(frame, (cx, cy), r + 3, (30, 30, 30), -1)
        cv2.circle(frame, (cx, cy), r, signal.color_bgr, -1)

        label = signal.state.upper()
        cv2.putText(frame, label, (cx - 30, cy + r + 20), self.FONT, 0.55, signal.color_bgr, 1)

    def _draw_signal_roi(self, frame):
        if self.cfg.SIGNAL_MODE != "roi" or not self.cfg.SIGNAL_ROI:
            return

        x1, y1, x2, y2 = map(int, self.cfg.SIGNAL_ROI)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "SIGNAL ROI", (x1, max(20, y1 - 8)), self.FONT, 0.5, (255, 255, 0), 1)

    def _draw_vehicle(self, frame, vehicle):
        active_violations = vehicle.violations

        if active_violations:
            color = self.out.VIOLATION_COLORS.get(active_violations[-1], (0, 0, 255))
        else:
            color = self.CLASS_COLORS.get(vehicle.class_name, (200, 200, 200))

        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{vehicle.vehicle_id} {vehicle.class_name}"
        risk = self.out.vehicle_risk_scores.get(vehicle.vehicle_id, 0)

        if risk > 0:
            label += f" R:{risk}"

        (tw, th), _ = cv2.getTextSize(label, self.FONT, 0.5, 1)
        y_label = max(0, y1 - th - 7)

        cv2.rectangle(frame, (x1, y_label), (x1 + tw + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), self.FONT, 0.5, (0, 0, 0), 1)

        if active_violations:
            badge = self.out.VIOLATION_LABELS.get(active_violations[-1], "Violation")
            cv2.putText(frame, f"! {badge}", (x1, min(frame.shape[0] - 5, y2 + 17)), self.FONT, 0.5, color, 1)

        if self.cfg.SHOW_TRAILS and len(vehicle.trail) > 1:
            points = list(vehicle.trail)[-self.cfg.TRAIL_LENGTH:]
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 1)

        vx, vy = vehicle.avg_velocity
        speed = math.hypot(vx, vy)

        if speed > 2:
            cx, cy = vehicle.center
            scale = min(speed * 3, 45)
            ex = int(cx + vx / speed * scale)
            ey = int(cy + vy / speed * scale)
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 1, tipLength=0.35)

    def _draw_hud(self, frame, frame_no, fps, vehicles):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, h - 92), (390, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, f"Frame: {frame_no}", (8, h - 72), self.FONT, 0.48, (230, 230, 230), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (8, h - 54), self.FONT, 0.48, (230, 230, 230), 1)
        cv2.putText(frame, f"Vehicles: {len(vehicles)}", (8, h - 36), self.FONT, 0.48, (230, 230, 230), 1)

        total = sum(self.out.violation_count.values())
        cv2.putText(frame, f"Violations: {total}", (8, h - 18), self.FONT, 0.48, (0, 120, 255), 1)


class TrafficViolationSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = VehicleDetector(cfg)
        self.tracker = CentroidTracker(cfg)
        self.out_mgr = OutputManager(cfg)
        self.signal = TrafficSignal(cfg)
        self.annotator = Annotator(cfg, self.out_mgr)
        self.stop_line_y = 0
        self.viol_detector = None

    def run(self, source, display=True, save_video=True):
        cap = self._open_source(source)

        if cap is None:
            return

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in is None or fps_in <= 1:
            fps_in = 25.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if w <= 0 or h <= 0:
            log.error("Invalid video size. Check your input video.")
            cap.release()
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.stop_line_y = int(h * 0.55)
        self.viol_detector = ViolationDetector(self.cfg, self.stop_line_y, w)

        if save_video:
            output_video = str(Path(self.cfg.OUTPUT_DIR) / "output_annotated.mp4")
            self.out_mgr.init_video_writer(output_video, fps_in, (w, h))

        frame_buffer = deque(maxlen=max(1, int(fps_in * self.cfg.EVIDENCE_CLIP_SECONDS)))

        log.info(f"Processing: {w}x{h} @ {fps_in:.1f} fps | stop_line_y={self.stop_line_y}")

        frame_no = 0
        fps_timer = time.time()
        fps_display = 0.0
        fps_counter = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_no += 1
            fps_counter += 1
            frame_buffer.append(frame.copy())

            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            self.signal.update(frame_no, frame)

            detections = self.detector.detect(frame)
            vehicles = self.tracker.update(detections, frame_no)

            for vid, vehicle in list(vehicles.items()):
                new_violations = self.viol_detector.check(vehicle, self.signal)

                for vtype in new_violations:
                    bbox_history = list(vehicle.bbox_history)
                    evidence_frames = []

                    for idx, old_frame in enumerate(list(frame_buffer)):
                        old_frame_no = frame_no - len(frame_buffer) + 1 + idx
                        nearest_bbox = vehicle.bbox
                        best_gap = float("inf")

                        for hist_frame_no, hist_bbox in bbox_history:
                            gap = abs(hist_frame_no - old_frame_no)
                            if gap < best_gap:
                                best_gap = gap
                                nearest_bbox = hist_bbox

                        evidence_frames.append((old_frame, nearest_bbox))

                    event = ViolationEvent(
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        vehicle_id=vid,
                        vehicle_type=vehicle.class_name,
                        violation_type=vtype,
                        frame_number=frame_no,
                        bbox=vehicle.bbox,
                    )

                    saved = self.out_mgr.save_violation(
                        frame=frame,
                        event=event,
                        evidence_frames=evidence_frames,
                        fps=fps_in,
                        size=(w, h),
                    )

                    if saved:
                        vehicle.violations.append(vtype)

            vis = self.annotator.draw_scene(frame, vehicles, self.signal, self.stop_line_y, frame_no, fps_display)

            if save_video:
                self.out_mgr.write_frame(vis)

            if display:
                cv2.imshow("Traffic Violation Detection", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    log.info("User quit.")
                    break

                if key == ord("p"):
                    cv2.waitKey(0)

            if frame_no % 100 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                log.info(f"Frame {frame_no}/{total_frames} ({pct:.1f}%) | Vehicles={len(vehicles)} | FPS={fps_display:.1f}")

        cap.release()
        self.out_mgr.release()

        if display:
            cv2.destroyAllWindows()

        self._print_summary(frame_no)

    def _open_source(self, source):
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            log.error(f"Cannot open source: {source}")
            return None

        return cap

    def _print_summary(self, total_frames):
        log.info("=" * 55)
        log.info("PROCESSING COMPLETE")
        log.info(f"Total frames processed : {total_frames}")

        for vtype, label in OutputManager.VIOLATION_LABELS.items():
            log.info(f"{label:20s}: {self.out_mgr.violation_count.get(vtype, 0)}")

        log.info(f"CSV report : {self.out_mgr.csv_path}")
        log.info(f"Output dir : {self.cfg.OUTPUT_DIR}/")
        log.info("=" * 55)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Violation Detection System")

    parser.add_argument("--source", default="demo", help="Video path, camera index, or demo")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    parser.add_argument(
        "--allowed-direction",
        type=float,
        default=0.0,
        help="Allowed traffic direction: 0=right, 90=down, 180=left, -90=up",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--no-save-video", action="store_true", help="Disable annotated output video saving")

    parser.add_argument("--signal-mode", choices=["cycle", "roi"], default="cycle", help="Signal mode")
    parser.add_argument("--signal-roi", default="", help="Signal ROI as x1,y1,x2,y2")
    parser.add_argument("--yellow-is-stop", action="store_true", help="Treat yellow signal as stop")

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    cfg.MODEL_PATH = args.model
    cfg.OUTPUT_DIR = args.output
    cfg.CONFIDENCE_THRESHOLD = args.conf
    cfg.ALLOWED_DIRECTION_DEGREES = args.allowed_direction

    cfg.SIGNAL_MODE = args.signal_mode
    cfg.TREAT_YELLOW_AS_STOP = args.yellow_is_stop

    if args.signal_roi:
        try:
            parts = [int(x.strip()) for x in args.signal_roi.split(",")]
            if len(parts) == 4:
                cfg.SIGNAL_ROI = tuple(parts)
            else:
                log.warning("Invalid --signal-roi. Use format: x1,y1,x2,y2")
        except Exception:
            log.warning("Invalid --signal-roi. Use format: x1,y1,x2,y2")

    system = TrafficViolationSystem(cfg)

    system.run(
        source=args.source,
        display=not args.no_display,
        save_video=not args.no_save_video,
    )


if __name__ == "__main__":
    main()

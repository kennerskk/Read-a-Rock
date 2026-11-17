# =====================================================
#  hand_detector.py — ระบบตรวจจับมือและนับนิ้ว (MediaPipe)
#  ตรวจจับท่าเป่ายิ้งฉุบ:
#    - 0 นิ้ว (กำมือ)   -> "ค้อน"
#    - 2 นิ้ว (ชี้+กลาง) -> "กรรไกร"
#    - 5 นิ้ว (แบมือ)    -> "กระดาษ"
#
#  รองรับ MediaPipe ทั้งสองยุคโดยอัตโนมัติ:
#    - Legacy API (mediapipe 0.10.x) : mp.solutions.hands
#    - Tasks API  (mediapipe 1.x)    : mediapipe.tasks.python.vision.HandLandmarker
#      MediaPipe 1.x ถอด mp.solutions ออกแล้ว โค้ดจึงตรวจหาเองว่ามี API ไหน
#      แล้วเลือกใช้ให้เหมาะ ส่วนที่เรียกใช้ภายนอก (find_hands / classify_gesture)
#      ทำงานเหมือนเดิมทุกประการ ไม่ต้องแก้ main_game.py
#
#  Tasks API ต้องใช้ไฟล์โมเดล hand_landmarker.task
#  โค้ดจะดาวน์โหลดให้อัตโนมัติครั้งแรกไปไว้ที่ models/hand_landmarker.task
# =====================================================
import os
import math
import time
from collections import deque

import cv2
import mediapipe as mp

# ---------- ตรวจว่า MediaPipe เวอร์ชันนี้มี API ไหนให้ใช้ ----------
_HAS_LEGACY = hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions", None), "hands")
_HAS_TASKS = False
if not _HAS_LEGACY:
    try:
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision
        _HAS_TASKS = True
    except Exception:
        _HAS_TASKS = False

MP_VERSION = getattr(mp, "__version__", "unknown")

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "hand_landmarker.task")

# โครงสร้างเส้นเชื่อม 21 จุดของมือ (คงที่ทุกเวอร์ชัน) ใช้วาดเองเมื่อไม่มี drawing_utils เดิม
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_hand_landmarker_model(path: str = DEFAULT_MODEL_PATH) -> str:
    """ดาวน์โหลดไฟล์โมเดลสำหรับ Tasks API ถ้ายังไม่มี (ทำครั้งเดียว ~7 MB)"""
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import urllib.request
    print(f"[HandDetector] ดาวน์โหลดโมเดลตรวจจับมือครั้งแรก -> {path}")
    print(f"               จาก {HAND_LANDMARKER_URL}")
    urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    print("[HandDetector] ดาวน์โหลดเสร็จแล้ว")
    return path


def _dist(p1, p2):
    """คำนวณระยะทางแบบ Euclidean 2D ระหว่างจุด 2 จุด (พิกัด normalized หรือ pixel)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class HandDetector:
    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.6, history_len=5,
                 model_path=DEFAULT_MODEL_PATH):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.backend = None
        self._last_ts = 0

        if _HAS_LEGACY:
            # ---------- MediaPipe 0.10.x ----------
            self.backend = "legacy"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_con,
                min_tracking_confidence=track_con
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_draw_styles = mp.solutions.drawing_styles

        elif _HAS_TASKS:
            # ---------- MediaPipe 1.x ----------
            self.backend = "tasks"
            resolved = ensure_hand_landmarker_model(model_path)
            options = _mp_vision.HandLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=resolved),
                running_mode=_mp_vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_con,
                min_hand_presence_confidence=detection_con,
                min_tracking_confidence=track_con,
            )
            self.landmarker = _mp_vision.HandLandmarker.create_from_options(options)

        else:
            raise RuntimeError(
                f"MediaPipe {MP_VERSION} ที่ติดตั้งอยู่ไม่มีทั้ง mp.solutions.hands และ "
                "mediapipe.tasks.python.vision ให้ใช้งาน\n"
                "แก้ได้โดย:  pip install \"mediapipe==0.10.21\""
            )

        # บัฟเฟอร์สำหรับ smoothing ท่าที่ตรวจจับได้ ป้องกันการกระพริบ
        self.gesture_history = deque(maxlen=history_len)

    # ------------------------------------------------------------------
    def _draw_manual(self, frame, pixel_lms):
        """วาดโครงมือเองด้วย OpenCV (ใช้กับ Tasks API ที่ไม่มี drawing_utils เดิม)"""
        for a, b in HAND_CONNECTIONS:
            if a < len(pixel_lms) and b < len(pixel_lms):
                cv2.line(frame, pixel_lms[a][:2], pixel_lms[b][:2], (0, 255, 128), 2)
        for (px, py, _z) in pixel_lms:
            cv2.circle(frame, (px, py), 3, (0, 200, 255), -1)

    def _pack_hand(self, norm_pts, handedness, w, h):
        """แปลง landmark normalized -> โครงสร้าง dict มาตรฐานของโปรเจกต์"""
        pixel_lms, norm_lms, xs, ys = [], [], [], []
        for lm in norm_pts:
            px, py = int(lm.x * w), int(lm.y * h)
            pixel_lms.append((px, py, lm.z))
            norm_lms.append((lm.x, lm.y, lm.z))
            xs.append(px)
            ys.append(py)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox = (x_min - 20, y_min - 20, (x_max - x_min) + 40, (y_max - y_min) + 40)
        return {
            "landmarks": pixel_lms,
            "norm_landmarks": norm_lms,
            "bbox": bbox,
            "handedness": handedness,
            "raw_landmarks": norm_pts,
        }

    def find_hands(self, frame, draw=True):
        """
        รับภาพ BGR จากกล้อง
        คืนค่า:
          - frame ที่วาด landmark แล้ว (ถ้า draw=True)
          - list ของข้อมูลมือที่ตรวจพบ แต่ละมือเป็น dict ประกอบด้วย:
            'landmarks': [(x, y, z), ... 21 จุด พิกัด pixel]
            'norm_landmarks': [(x, y, z), ... 21 จุด พิกัด 0.0-1.0]
            'bbox': (x_min, y_min, w, h)
            'handedness': 'Right' หรือ 'Left'
        """
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_data = []

        if self.backend == "legacy":
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    handedness = "Right"
                    if results.multi_handedness and hand_idx < len(results.multi_handedness):
                        handedness = results.multi_handedness[hand_idx].classification[0].label

                    info = self._pack_hand(hand_lms.landmark, handedness, w, h)
                    info["raw_landmarks"] = hand_lms

                    if draw:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_lms,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2)
                        )
                    hands_data.append(info)

        else:  # tasks API
            # timestamp ต้องเพิ่มขึ้นเสมอใน VIDEO mode
            ts = max(self._last_ts + 1, int(time.perf_counter() * 1000))
            self._last_ts = ts
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = self.landmarker.detect_for_video(mp_image, ts)

            for hand_idx, norm_pts in enumerate(result.hand_landmarks or []):
                handedness = "Right"
                try:
                    cats = result.handedness[hand_idx]
                    if cats:
                        handedness = cats[0].category_name
                except Exception:
                    pass

                info = self._pack_hand(norm_pts, handedness, w, h)
                if draw:
                    self._draw_manual(frame, info["landmarks"])
                hands_data.append(info)

        return frame, hands_data

    def classify_gesture(self, hand_info):
        """
        วิเคราะห์ท่าทางของมือ:
          - นิ้วโป้ง, นิ้วชี้, นิ้วกลาง, นิ้วนาง, นิ้วก้อย ว่าเปิด (1) หรือปิด (0)
          - นับจำนวนนิ้วที่ชู
          - แปลงเป็นท่าเป่ายิ้งฉุบ:
              ค้อน   (0 นิ้ว หรือ กำมือ)
              กรรไกร (2 นิ้ว ชี้+กลาง)
              กระดาษ (5 นิ้ว หรือ แบมือ 4-5 นิ้ว)
        """
        lms = hand_info["landmarks"]
        wrist = lms[0]

        # 4 นิ้วหลัก: ชี้ (8), กลาง (12), นาง (16), ก้อย (20)
        # PIP (ข้อกลาง): ชี้ (6), กลาง (10), นาง (14), ก้อย (18)
        # MCP (โคนนิ้ว): ชี้ (5), กลาง (9), นาง (13), ก้อย (17)
        fingers = []

        # 1. นิ้วชี้, กลาง, นาง, ก้อย
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        for tip_id, pip_id in zip(finger_tips, finger_pips):
            tip = lms[tip_id]
            pip = lms[pip_id]

            # เงื่อนไข 1: ปลายนิ้วอยู่สูงกว่าข้อกลาง (y น้อยกว่าใน OpenCV)
            # เงื่อนไข 2: ระยะห่างจากข้อมือไปยังปลายนิ้ว ยาวกว่าระยะไปยังข้อกลาง
            is_extended = (tip[1] < pip[1]) and (_dist(tip, wrist) > _dist(pip, wrist) * 1.05)
            fingers.append(1 if is_extended else 0)

        index_open, middle_open, ring_open, pinky_open = fingers

        # 2. นิ้วโป้ง: tip=4, ip=3, mcp=2
        # นิ้วโป้งกางออกด้านข้าง หรือหุบเข้าหาฝ่ามือ
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        pinky_mcp = lms[17]
        index_mcp = lms[5]

        # ถ้าระยะห่างของปลายโป้งจากโคนก้อย มากกว่าข้อต่อโป้ง แสดงว่ากางออก
        thumb_dist_pinky = _dist(thumb_tip, pinky_mcp)
        thumb_ip_dist_pinky = _dist(thumb_ip, pinky_mcp)
        thumb_dist_index = _dist(thumb_tip, index_mcp)

        thumb_open = 1 if (thumb_dist_pinky > thumb_ip_dist_pinky * 1.15 and thumb_dist_index > _dist(thumb_ip, index_mcp)) else 0

        # รายการนิ้วทั้งหมด 5 นิ้ว: [โป้ง, ชี้, กลาง, นาง, ก้อย]
        all_fingers = [thumb_open, index_open, middle_open, ring_open, pinky_open]
        total_open = sum(all_fingers)

        # จำแนกท่าเป่ายิ้งฉุบ
        gesture = None
        gesture_thai = None

        # --- ค้อน (0 นิ้ว / กำมือ) ---
        # นิ้วทั้ง 4 หลักพับลงหมด
        if index_open == 0 and middle_open == 0 and ring_open == 0 and pinky_open == 0:
            gesture = "rock"
            gesture_thai = "ค้อน"
            total_open = 0  # เพื่อแสดงผล 0 นิ้วตามกฎกติกาชัดเจน

        # --- กรรไกร (2 นิ้ว ชี้ + กลาง) ---
        # นิ้วชี้และนิ้วกลางชูขึ้น แต่นิ้วนางและนิ้วก้อยพับลง
        elif index_open == 1 and middle_open == 1 and ring_open == 0 and pinky_open == 0:
            gesture = "scissors"
            gesture_thai = "กรรไกร"
            total_open = 2  # แสดง 2 นิ้ว

        # --- กระดาษ (5 นิ้ว / แบมือ) ---
        # นิ้วทั้ง 4 หรือทั้ง 5 ชูออกทั้งหมด
        elif index_open == 1 and middle_open == 1 and ring_open == 1 and pinky_open == 1:
            gesture = "paper"
            gesture_thai = "กระดาษ"
            total_open = 5  # แสดง 5 นิ้ว

        else:
            gesture = "unknown"
            gesture_thai = "รอท่า..."

        # เพิ่มลงประวัติเพื่อดูความสม่ำเสมอ
        self.gesture_history.append(gesture_thai)

        # หาค่าที่ออกบ่อยสุดในประวัติล่าสุด (Smoothing)
        if gesture_thai != "รอท่า...":
            stable_gesture = gesture_thai
        else:
            # ถ้าเป็น unknown ชั่วคราว 1-2 เฟรม ให้ใช้ค่าเดิมก่อนได้
            counts = {g: self.gesture_history.count(g) for g in set(self.gesture_history) if g != "รอท่า..."}
            if counts:
                stable_gesture = max(counts, key=counts.get)
            else:
                stable_gesture = "รอท่า..."

        return {
            "fingers": all_fingers,
            "total_open": total_open,
            "gesture": gesture,
            "gesture_thai": gesture_thai,
            "stable_gesture": stable_gesture
        }

    def close(self):
        try:
            if self.backend == "legacy":
                self.hands.close()
            else:
                self.landmarker.close()
        except Exception:
            pass

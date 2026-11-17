# =====================================================
#  hand_detector.py — ระบบตรวจจับมือและนับนิ้ว (MediaPipe)
#  ตรวจจับท่าเป่ายิ้งฉุบ:
#    - 0 นิ้ว (กำมือ)   -> "ค้อน"
#    - 2 นิ้ว (ชี้+กลาง) -> "กรรไกร"
#    - 5 นิ้ว (แบมือ)    -> "กระดาษ"
# =====================================================
import math
from collections import deque
import cv2
import mediapipe as mp


def _dist(p1, p2):
    """คำนวณระยะทางแบบ Euclidean 2D ระหว่างจุด 2 จุด (พิกัด normalized หรือ pixel)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class HandDetector:
    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.6, history_len=5):
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # บัฟเฟอร์สำหรับ smoothing ท่าที่ตรวจจับได้ ป้องกันการกระพริบ
        self.gesture_history = deque(maxlen=history_len)

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
        results = self.hands.process(img_rgb)

        hands_data = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # ตรวจ handedness ถ้ามี
                handedness = "Right"
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[hand_idx].classification[0].label

                pixel_lms = []
                norm_lms = []
                x_list = []
                y_list = []

                for lm in hand_lms.landmark:
                    px, py = int(lm.x * w), int(lm.y * h)
                    pixel_lms.append((px, py, lm.z))
                    norm_lms.append((lm.x, lm.y, lm.z))
                    x_list.append(px)
                    y_list.append(py)

                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                bbox = (x_min - 20, y_min - 20, (x_max - x_min) + 40, (y_max - y_min) + 40)

                if draw:
                    # วาดเส้นเชื่อมต่อจุด landmark
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2)
                    )

                hands_data.append({
                    "landmarks": pixel_lms,
                    "norm_landmarks": norm_lms,
                    "bbox": bbox,
                    "handedness": handedness,
                    "raw_landmarks": hand_lms
                })

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
        self.hands.close()

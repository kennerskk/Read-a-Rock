# =====================================================
#  main_game.py — เกมเป่ายิ้งฉุบสู้กับ AI ผ่านกล้อง Webcam
#
#  การทำงานร่วมกันของ 2 โมเดล:
#  1. Vision Model (MediaPipe Hands):
#     - ตรวจจับจุดพิกัดมือ 21 จุด
#     - นับจำนวนนิ้ว: 0 นิ้ว = ค้อน | 2 นิ้ว = กรรไกร | 5 นิ้ว = กระดาษ
#  2. AI Brain Model (adaptive_arena.py & model_pipeline.py):
#     - เลือกบอส AI ได้ 8 ระดับ (รวมถึง Data Science ML Boss)
#     - AI เรียนรู้จับทางผู้เล่นตามสถิติและพฤติกรรมในแต่ละรอบ
#     - รองรับการ Export [S] โมเดลที่เล่นจนเก่งออกมาเป็นไฟล์ JSON
#     - รองรับการ Import [L] โมเดลที่เคยบันทึกไว้กลับมาสู้ต่อ
#     - บันทึก Telemetry การเล่นจริงลง data/gameplay_telemetry.csv อัตโนมัติ
# =====================================================
import os
import csv
import time
import sys
import argparse
from datetime import datetime
import cv2

# นำเข้าโมเดลและคลาสจาก adaptive_arena.py
from adaptive_arena import BOSSES, judge, MOVES, BEATS, COUNTER
from hand_detector import HandDetector
from ui_renderer import UIRenderer
from sound_effects import SoundManager
from model_io import save_boss_model, load_boss_model, list_saved_models, get_latest_saved_model


# นิยามสถานะเกม
STATE_IDLE = "IDLE"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RESULT = "RESULT"
STATE_MENU = "MENU"


class RockPaperScissorsGame:
    def __init__(self, camera_id=0, countdown=3.0):
        self.camera_id = camera_id
        self.cap = None

        # โมเดล 1: ตรวจจับมือและนิ้ว
        self.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.6)

        # โมเดล 2: AI บอส จาก adaptive_arena.py
        self.boss_index = 5  # ค่าเริ่มต้น: บอส 6 (Ensemble/Meta ทายแม่นสุด)
        self.boss = BOSSES[self.boss_index]()

        # ส่วนแสดงผล UI และระบบเสียง
        self.ui = UIRenderer()
        self.sound = SoundManager(enabled=True)

        # สถิติคะแนน
        self.opp_history = []  # ประวัติท่าที่ผู้เล่นเคยออก (ให้บอสเรียนรู้)
        self.round_no = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        # ตัวแปรสถานะเกม
        self.state = STATE_IDLE
        self.auto_mode = False       # โหมดเล่นต่อเนื่องอัตโนมัติ
        self.draw_landmarks = True   # เปิด/ปิดการวาดเส้นโครงสร้างมือ

        # ตัวจับเวลาและความเร็วรอบ
        self.countdown_start = 0
        self.speed_presets = [3.0, 2.0, 1.5, 1.0]
        self.countdown_duration = float(countdown)
        self.result_duration = max(0.8, round(self.countdown_duration * 0.7, 1))
        self.last_tick_sec = -1
        self.last_player_move = "-"
        self.last_boss_move = "-"
        self.last_result = "-"

        # จำท่าล่าสุดที่ตรวจจับได้แบบเสถียร
        self.last_valid_detected_move = "ค้อน"

        # ระบบ Toast แจ้งเตือนบน HUD
        self.toast_text = ""
        self.toast_color = (0, 255, 120)
        self.toast_expiry = 0.0

        # ระบบบันทึกข้อมูล Telemetry สดสำหรับการเทรนใน Data Science Pipeline
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.telemetry_path = os.path.join(self.data_dir, "gameplay_telemetry.csv")
        self.session_id = f"webcam_{int(time.time())}"
        self._init_telemetry_file()

        # ตัวนับสำหรับวนโหลดโมเดลที่บันทึกไว้
        self.saved_model_cycle_idx = 0

    def _init_telemetry_file(self):
        """ตรวจสอบและสร้างหัวตาราง CSV หากยังไม่มีไฟล์"""
        if not os.path.exists(self.telemetry_path) or os.path.getsize(self.telemetry_path) == 0:
            with open(self.telemetry_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "round_id", "player_move", "ai_move", "result", "reaction_time_ms"])

    def log_telemetry(self, player_move: str, ai_move: str, result: str):
        """บันทึกข้อมูลผลการเล่นแต่ละรอบลง Telemetry Dataset"""
        try:
            with open(self.telemetry_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.round_no,
                    player_move,
                    ai_move,
                    result,
                    1200  # ค่าเฉลี่ยเวลานับถอยหลัง
                ])
        except Exception as e:
            print(f"[Telemetry Error] {e}")

    def show_toast(self, text: str, color=(0, 255, 120), duration: float = 3.2):
        """แสดงข้อความแจ้งเตือน Toast Message บนหน้าจอ OpenCV"""
        self.toast_text = text
        self.toast_color = color
        self.toast_expiry = time.time() + duration

    def export_current_boss(self):
        """ส่งออก (Export) สมองบอสที่กำลังเล่นอยู่ลงไฟล์ JSON"""
        ai_winrate = (self.losses / self.round_no * 100) if self.round_no > 0 else 0.0
        meta = {
            "custom_name": f"บอสแก้ทางฉัน (ชนะ AI {ai_winrate:.0f}%)",
            "stats": {
                "total_rounds": self.round_no,
                "player_wins": self.wins,
                "ai_wins": self.losses,
                "draws": self.draws,
                "ai_winrate_pct": ai_winrate,
                "session_id": self.session_id
            }
        }
        saved_file = save_boss_model(self.boss, metadata=meta)
        fname = os.path.basename(saved_file)
        self.show_toast(f"บันทึกโมเดลสำเร็จ: {fname}", color=(0, 255, 120))
        self.sound.shoot()
        print(f"\n[Export Model] บันทึกโมเดลสมองบอสเรียบร้อยแล้ว: {saved_file}")

    def import_saved_boss(self):
        """นำเข้า (Import) โมเดลสมองบอสที่เคยบันทึกไว้ในโฟลเดอร์ models/"""
        models = list_saved_models("models")
        if not models:
            self.show_toast("ยังไม่มีไฟล์โมเดลใน models/ ให้กด [S] เซฟก่อน", color=(0, 160, 255))
            return

        # วนรอบโมเดลที่มีอยู่
        idx = self.saved_model_cycle_idx % len(models)
        self.saved_model_cycle_idx += 1
        selected = models[idx]

        try:
            loaded_boss = load_boss_model(selected["filepath"])
            self.boss = loaded_boss
            b_name = getattr(loaded_boss, "name", selected.get("filename", "Custom Boss"))
            self.show_toast(f"โหลดโมเดล: {b_name[:28]}", color=(0, 220, 255))
            self.sound.change_boss()
            print(f"\n[Import Model] โหลดโมเดลเรียบร้อย: {selected['filepath']}")
        except Exception as e:
            self.show_toast(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}", color=(50, 70, 255))
            print(f"[Import Error] {e}")

    def init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] ไม่สามารถเปิดกล้อง ID {self.camera_id} ได้")
            return False

        # ตั้งค่าขนาดความละเอียดกล้องเป็น 1280x720 (หากกล้องรองรับ)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True

    def switch_boss(self, index):
        if 0 <= index < len(BOSSES):
            self.boss_index = index
            self.boss = BOSSES[self.boss_index]()
            self.sound.change_boss()
            self.show_toast(f"เปลี่ยนบอส: {self.boss.name[:26]}", color=(0, 220, 255))
            print(f">>> เปลี่ยนเป็นบอส: {self.boss.name}")

    def cycle_speed(self):
        """สลับความเร็วเวลานับถอยหลังต่อรอบ (3s -> 2s -> 1.5s -> 1s)"""
        try:
            curr_idx = self.speed_presets.index(self.countdown_duration)
            next_idx = (curr_idx + 1) % len(self.speed_presets)
        except ValueError:
            next_idx = 0
        self.countdown_duration = self.speed_presets[next_idx]
        self.result_duration = max(0.8, round(self.countdown_duration * 0.7, 1))
        self.show_toast(f"⏱️ ความเร็ว: นับถอยหลัง {self.countdown_duration:.1f}s (แสดงผล {self.result_duration:.1f}s)", color=(255, 200, 0))
        print(f">>> ปรับความเร็วรอบ: นับถอยหลัง={self.countdown_duration:.1f}s | ผล={self.result_duration:.1f}s")

    def reset_score(self):
        self.opp_history = []
        self.round_no = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.last_player_move = "-"
        self.last_boss_move = "-"
        self.last_result = "-"
        self.boss = BOSSES[self.boss_index]()  # รีเซ็ตสมองบอสใหม่
        self.sound.shoot()
        self.show_toast("รีเซ็ตคะแนนและล้างประวัติการเรียนรู้แล้ว", color=(0, 220, 255))
        print(">>> รีเซ็ตคะแนนและล้างประวัติการเรียนรู้เรียบร้อย")

    def start_round(self):
        if self.state in [STATE_IDLE, STATE_RESULT]:
            self.state = STATE_COUNTDOWN
            self.countdown_start = time.time()
            self.last_tick_sec = -1

    def execute_showdown(self, detected_gesture):
        # 1. กำหนดท่าของผู้เล่น
        if detected_gesture in MOVES:
            player_move = detected_gesture
        elif self.last_valid_detected_move in MOVES:
            player_move = self.last_valid_detected_move
        else:
            player_move = "ค้อน"  # ค่าเริ่มต้นเผื่อไม่ทันตรวจจับ

        # 2. ให้บอส AI เลือกท่าโดยดูจากประวัติการออกท่าของผู้เล่น
        boss_move = self.boss.choose(self.opp_history)

        # 3. ตัดสินผลแพ้ชนะ (มุมมองผู้เล่น: win / lose / draw)
        result = judge(player_move, boss_move)

        # 4. ส่งผลการแข่งให้บอส AI เรียนรู้ (reward: บอสชนะ +1 / เสมอ 0 / บอสแพ้ -1)
        self.opp_history.append(player_move)
        reward = {"win": -1, "draw": 0, "lose": 1}[result]
        self.boss.learn(reward, self.opp_history)

        # 5. อัปเดตสถิติ
        self.round_no += 1
        if result == "win":
            self.wins += 1
            self.sound.win()
        elif result == "lose":
            self.losses += 1
            self.sound.lose()
        else:
            self.draws += 1
            self.sound.draw()

        self.last_player_move = player_move
        self.last_boss_move = boss_move
        self.last_result = result
        self.result_start = time.time()
        self.state = STATE_RESULT

        # 6. บันทึกผลลง Telemetry Dataset
        self.log_telemetry(player_move, boss_move, result)

        print(f"[ตา {self.round_no}] คุณ: {player_move} | บอส: {boss_move} -> ผล: {result}")

    def run(self):
        if not self.init_camera():
            return

        window_name = "Rock-Paper-Scissors AI Arena (MediaPipe + Adaptive AI)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 65)
        print("     เกมเป่ายิ้งฉุบ AI Arena พร้อมเริ่มแล้ว!")
        print("  - โมเดลที่ 1: MediaPipe ตรวจจับนิ้วมือ (0=ค้อน, 2=กรรไกร, 5=กระดาษ)")
        print("  - โมเดลที่ 2: AI บอสวิเคราะห์จับทาง (adaptive_arena.py & DS Pipeline)")
        print("=" * 65)
        print("ปุ่มควบคุม:")
        print("  [SPACE]      : เริ่มรอบเป่ายิ้งฉุบ")
        print("  [A]          : สลับโหมดเล่นต่อเนื่องอัตโนมัติ (AUTO)")
        print("  [T]          : สลับความเร็วนับถอยหลังต่อตา (3s / 2s / 1.5s / 1s)")
        print("  [1 - 8]      : สลับโมเดลสมองบอส AI ทันที (8=Data Science ML Boss)")
        print("  [S]          : ส่งออก (Export) สมองบอสปัจจุบันที่เรียนรู้สไตล์เราไว้")
        print("  [L]          : นำเข้า (Import) โมเดลที่บันทึกไว้ใน models/ มาประลอง")
        print("  [B]          : เปิดเมนูดูรายละเอียดบอส")
        print("  [R]          : รีเซ็ตคะแนนและประวัติ")
        print("  [M]          : ปิด/เปิดเสียงเอฟเฟกต์")
        print("  [H]          : ซ่อน/แสดงจุด Landmark มือ")
        print("  [Q หรือ ESC] : ออกจากเกม\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] ไม่สามารถอ่านเฟรมจากกล้องได้")
                break

            # กลับด้านภาพแนวนอนเหมือนส่องกระจกเงา
            frame = cv2.flip(frame, 1)

            # โมเดล 1: ค้นหาและระบุตำแหน่งมือ
            frame, hands = self.detector.find_hands(frame, draw=self.draw_landmarks)

            detected_gesture = "รอท่า..."
            total_fingers = 0

            if hands:
                hand_info = hands[0]
                gesture_data = self.detector.classify_gesture(hand_info)
                detected_gesture = gesture_data["stable_gesture"]
                total_fingers = gesture_data["total_open"]

                if detected_gesture in MOVES:
                    self.last_valid_detected_move = detected_gesture

            # คำนวณอัตราชนะ
            winrate = (self.wins / self.round_no * 100) if self.round_no > 0 else 0.0

            # --- จัดการ State Machine ของเกม ---
            current_time = time.time()

            if self.state == STATE_COUNTDOWN:
                elapsed = current_time - self.countdown_start
                duration = max(0.5, self.countdown_duration)
                progress = elapsed / duration

                if progress < 0.33:
                    count_text = "3"
                    cur_sec = 3
                elif progress < 0.66:
                    count_text = "2"
                    cur_sec = 2
                elif progress < 1.0:
                    count_text = "1"
                    cur_sec = 1
                else:
                    count_text = "เป่ายิ้งฉุบ!"
                    cur_sec = 0

                # เล่นเสียงนับถอยหลังในจังหวะวินาที
                if cur_sec != self.last_tick_sec:
                    self.last_tick_sec = cur_sec
                    if cur_sec > 0:
                        self.sound.tick()
                    else:
                        self.sound.shoot()

                self.ui.draw_countdown(frame, count_text)

                # สิ้นสุดการนับถอยหลัง -> ตัดสินผลทันที!
                grace_period = max(0.25, duration * 0.15)
                if elapsed >= duration + grace_period:
                    self.execute_showdown(detected_gesture)

            elif self.state == STATE_RESULT:
                self.ui.draw_result_banner(frame, self.last_result, self.last_player_move, self.last_boss_move)
                if current_time - self.result_start > self.result_duration:
                    if self.auto_mode:
                        self.start_round()
                    else:
                        self.state = STATE_IDLE

            elif self.state == STATE_IDLE:
                self.ui.draw_idle_prompt(frame)

            elif self.state == STATE_MENU:
                self.ui.draw_boss_selection_menu(frame, BOSSES, self.boss_index)

            # วาดข้อมูลผู้เล่นและบอส
            self.ui.draw_player_card(frame, detected_gesture, total_fingers)
            self.ui.draw_boss_card(frame, self.last_boss_move if self.state == STATE_RESULT else "-", getattr(self.boss, "desc", ""))

            # วาดแถบหัวกระดานและเมนูด้านล่าง
            self.ui.draw_header_scoreboard(
                frame,
                self.round_no,
                self.wins,
                self.losses,
                self.draws,
                winrate,
                self.boss.name,
                auto_mode=self.auto_mode,
                sound_on=self.sound.enabled
            )
            self.ui.draw_footer_controls(frame)

            # วาด Toast แจ้งเตือนเมื่อมีการ Save / Load โมเดล
            if current_time < self.toast_expiry:
                self.ui.draw_toast_notification(frame, self.toast_text, color=self.toast_color)

            # แสดงผลลัพธ์บนหน้าต่าง OpenCV
            cv2.imshow(window_name, frame)

            # ตรวจจับการกดปุ่มคีย์บอร์ด
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:  # Q หรือ ESC เพื่อออก
                break
            elif key == ord(' '):  # SPACE เริ่มรอบใหม่
                if self.state == STATE_MENU:
                    self.state = STATE_IDLE
                else:
                    self.start_round()
            elif key in [ord('a'), ord('A')]:  # A สลับโหมดอัตโนมัติ
                self.auto_mode = not self.auto_mode
                if self.auto_mode and self.state == STATE_IDLE:
                    self.start_round()
            elif key in [ord('t'), ord('T')]:  # T ปรับความเร็วรอบ (Pace)
                self.cycle_speed()
            elif key in [ord('b'), ord('B')]:  # B เปิด/ปิดเมนูบอส
                self.state = STATE_IDLE if self.state == STATE_MENU else STATE_MENU
            elif ord('1') <= key <= ord('8'):  # 1-8 เปลี่ยนบอส
                idx = key - ord('1')
                self.switch_boss(idx)
                if self.state == STATE_MENU:
                    self.state = STATE_IDLE
            elif key in [ord('s'), ord('S')]:  # S ส่งออกโมเดลปัจจุบัน
                self.export_current_boss()
            elif key in [ord('l'), ord('L')]:  # L นำเข้าโมเดลที่เคยเซฟ
                self.import_saved_boss()
                if self.state == STATE_MENU:
                    self.state = STATE_IDLE
            elif key in [ord('r'), ord('R')]:  # R รีเซ็ตคะแนน
                self.reset_score()
            elif key in [ord('m'), ord('M')]:  # M ปิด/เปิดเสียง
                self.sound.enabled = not self.sound.enabled
            elif key in [ord('h'), ord('H')]:  # H ซ่อน/แสดงจุดมือ
                self.draw_landmarks = not self.draw_landmarks

        # คืนทรัพยากรเมื่อเลิกเล่น
        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nขอบคุณที่ร่วมสนุกกับเกมเป่ายิ้งฉุบ AI Arena ครับ! 👋")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-Paper-Scissors AI Arena (Webcam + MediaPipe + Adaptive AI)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--countdown", type=float, default=3.0, help="Countdown duration in seconds (default: 3.0, try 1.5 or 1.0 for fast mode)")
    args = parser.parse_args()

    game = RockPaperScissorsGame(camera_id=args.camera, countdown=args.countdown)
    game.run()

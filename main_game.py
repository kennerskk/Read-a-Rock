# =====================================================
#  main_game.py — เกมเป่ายิ้งฉุบสู้กับ AI ผ่านกล้อง Webcam
#
#  การทำงานร่วมกันของ 2 โมเดล:
#  1. Vision Model (MediaPipe Hands): ตรวจจับจุดมือ 21 จุด แล้วนับนิ้ว
#     0 นิ้ว = ค้อน | 2 นิ้ว = กรรไกร | 5 นิ้ว = กระดาษ
#  2. AI Brain Model (adaptive_arena.py): บอส 9 แบบ เรียนรู้จับทางผู้เล่น
#
#  ลำดับในแต่ละตา (ทั้งโหมดกด SPACE และโหมดออโต้):
#    รอท่า  -> ระบบ "ไม่" นับถอยหลังจนกว่าจะเห็นท่าชัด (ค้อน/กรรไกร/กระดาษ)
#              เวลาตั้งแต่พร้อมจนเห็นท่า = reaction_time_ms (บันทึกลง telemetry)
#    นับถอยหลัง 3-2-1 -> เปลี่ยนท่าได้จนกว่าจะนับจบ
#    ตัดสินผล -> บอสเลือกท่าจาก "ประวัติก่อนหน้า" เท่านั้น (ไม่เห็นท่าตานี้)
#
#  บันทึก Telemetry การเล่นจริงลง data/gameplay_telemetry.csv อัตโนมัติ
# =====================================================
import os
import csv
import time
import argparse
from datetime import datetime
import cv2

from adaptive_arena import (GAME_BOSSES, BEST_COMBOS, WARM_COMBO, WARM_DECAY, ComboEnsembleBoss, judge, MOVES)
from hand_detector import HandDetector
from ui_renderer import UIRenderer, short_boss_name, GREEN, ACCENT, AMBER, RED
from sound_effects import SoundManager
from model_io import save_boss_model, load_boss_model, list_saved_models


# สถานะเกม
STATE_IDLE = "IDLE"              # รอผู้เล่นกด SPACE / เปิดออโต้
STATE_WAIT = "WAIT_GESTURE"      # รอให้เห็นท่า (ยังไม่นับถอยหลัง)
STATE_COUNTDOWN = "COUNTDOWN"    # นับ 3-2-1
STATE_RESULT = "RESULT"          # แสดงผล

LOCK_SEC = 0.15                  # ท่าต้องนิ่งอย่างน้อยเท่านี้ถึงจะนับว่า "ออกท่าแล้ว"
LOCK_MIN_FRAMES = 3
DROP_FRAMES = 2                  # ไม่เห็นท่าติดกันกี่เฟรม จึงถือว่า "ลดมือ/ปล่อยท่าเดิม" แล้ว
REPEAT_HOLD_SEC = 0.8            # ค้างท่าเดิมจากตาที่แล้วนานเท่านี้ = ตั้งใจออกท่าเดิมซ้ำ
SHOOT_HOLD_SEC = 0.6             # เวลาโชว์บอสออกท่า (emoji) ก่อนขึ้นผล
DISPLAY_MIN_WIDTH = 1280         # ขยายภาพให้กว้างอย่างน้อยเท่านี้เพื่อให้ HUD ไม่ทับกัน


class RockPaperScissorsGame:
    def __init__(self, camera_id=0, countdown=3.0, player_name="Player", model_path=None):
        self.camera_id = camera_id
        self.player_name = str(player_name).strip() if player_name else "Player"
        self.cap = None

        # โมเดล 1: ตรวจจับมือ
        self.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.6)

        # โมเดล 2: บอส AI
        self.boss_index = 5             # บอส 6 (Ensemble) เป็นค่าเริ่มต้น
        self.combo_size = 2             # ขนาดของ Lite Ensemble (บอส 9)
        self.initial_model_toast = None
        self.boss = self.make_boss(self.boss_index)
        if model_path and os.path.exists(model_path):
            try:
                self.boss = load_boss_model(model_path)
                b_name = getattr(self.boss, "name", os.path.basename(model_path))
                self.initial_model_toast = f"โหลดโมเดล: {short_boss_name(b_name)[:30]}"
                print(f"โหลดโมเดลคู่ปรับสำเร็จ: {b_name} ({model_path})")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล {model_path} ได้: {e}")

        self.ui = UIRenderer()
        self.sound = SoundManager(enabled=True)

        # สถิติ
        self.opp_history = []
        self.recent_results = []
        self.round_no = self.wins = self.losses = self.draws = 0

        # สถานะเกม
        self.state = STATE_IDLE
        self.auto_mode = False
        self.draw_landmarks = True
        self.menu_open = False
        self.help_open = False

        # เวลา / ความเร็ว
        self.speed_presets = [3.0, 2.0, 1.5, 1.0]
        self.countdown_duration = float(countdown)
        self.result_duration = max(1.2, round(self.countdown_duration * 0.7, 1))
        self.countdown_start = 0.0
        self.result_start = 0.0
        self.last_tick = None

        # ตัวแปรของตาปัจจุบัน
        self.last_raw_move = None       # ท่าที่กล้องเห็นในเฟรมล่าสุด
        self._reset_round_vars()
        self.last_player_move = self.last_boss_move = self.last_result = "-"
        self.last_reaction_ms = None

        # Toast
        self.toast_text, self.toast_color, self.toast_expiry = "", GREEN, 0.0

        # Telemetry
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.telemetry_path = os.path.join(self.data_dir, "gameplay_telemetry.csv")
        self.session_id = f"webcam_{int(time.time())}"
        self._init_telemetry_file()
        self.saved_model_cycle_idx = 0

    # ------------------------------------------------------------------
    #  บอส
    # ------------------------------------------------------------------
    def make_boss(self, index):
        cls = GAME_BOSSES[index]
        if cls is ComboEnsembleBoss:
            if self.combo_size == "warm":
                # Prior เทรนจากข้อมูลผู้เล่นจริงทุกคนใน telemetry ตอนสร้างบอส
                return ComboEnsembleBoss(WARM_COMBO, WARM_DECAY)
            return ComboEnsembleBoss(BEST_COMBOS[self.combo_size])
        return cls()

    def switch_boss(self, index):
        if not 0 <= index < len(GAME_BOSSES):
            return
        if GAME_BOSSES[index] is ComboEnsembleBoss and isinstance(self.boss, ComboEnsembleBoss) \
                and self.boss_index == index:
            sizes = sorted(BEST_COMBOS) + ["warm"]          # 2 -> 3 -> 4 -> warm start -> 2
            self.combo_size = sizes[(sizes.index(self.combo_size) + 1) % len(sizes)]
        self.boss_index = index
        self.boss = self.make_boss(index)
        self.sound.change_boss()
        self.show_toast(f"เปลี่ยนบอส: {short_boss_name(self.boss.name)[:40]}", color=ACCENT)
        print(f">>> เปลี่ยนเป็นบอส: {self.boss.name}")

    # ------------------------------------------------------------------
    #  Telemetry
    # ------------------------------------------------------------------
    def _init_telemetry_file(self):
        if not os.path.exists(self.telemetry_path) or os.path.getsize(self.telemetry_path) == 0:
            with open(self.telemetry_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp", "round_id", "player_name", "player_move",
                                        "ai_move", "result", "reaction_time_ms"])

    def log_telemetry(self, player_move, ai_move, result, reaction_ms):
        """reaction_time_ms = เวลาจาก 'พร้อมรับท่า' จนระบบเห็นท่าสุดท้ายของผู้เล่นนิ่ง (วัดจริง)"""
        try:
            with open(self.telemetry_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([datetime.now().isoformat(), self.round_no, self.player_name,
                                        player_move, ai_move, result,
                                        "" if reaction_ms is None else int(round(reaction_ms))])
        except Exception as e:
            print(f"[Telemetry Error] {e}")

    # ------------------------------------------------------------------
    #  เครื่องมือ UI
    # ------------------------------------------------------------------
    def show_toast(self, text, color=GREEN, duration=3.0):
        self.toast_text, self.toast_color, self.toast_expiry = text, color, time.time() + duration

    def export_current_boss(self):
        ai_wr = (self.losses / self.round_no * 100) if self.round_no else 0.0
        meta = {
            "custom_name": f"บอสแก้ทาง {self.player_name} (ชนะ AI {ai_wr:.0f}%)",
            "target_player": self.player_name,
            "stats": {"total_rounds": self.round_no, "player_wins": self.wins, "ai_wins": self.losses,
                      "draws": self.draws, "ai_winrate_pct": ai_wr, "session_id": self.session_id},
        }
        saved = save_boss_model(self.boss, metadata=meta)
        self.show_toast(f"บันทึกโมเดลแล้ว: {os.path.basename(saved)}")
        self.sound.shoot()
        print(f"[Export Model] {saved}")

    def import_saved_boss(self):
        models = list_saved_models("models")
        if not models:
            self.show_toast("ยังไม่มีโมเดลใน models/ — กด S เพื่อบันทึกก่อน", color=AMBER)
            return
        selected = models[self.saved_model_cycle_idx % len(models)]
        self.saved_model_cycle_idx += 1
        try:
            self.boss = load_boss_model(selected["filepath"])
            name = getattr(self.boss, "name", selected.get("filename", "Custom Boss"))
            self.show_toast(f"โหลดโมเดล: {short_boss_name(name)[:34]}  (กด L ซ้ำเพื่อเปลี่ยน)", color=ACCENT)
            self.sound.change_boss()
            print(f"[Import Model] {selected['filepath']}")
        except Exception as e:
            self.show_toast(f"โหลดโมเดลไม่สำเร็จ: {e}", color=RED)

    def cycle_speed(self):
        try:
            nxt = (self.speed_presets.index(self.countdown_duration) + 1) % len(self.speed_presets)
        except ValueError:
            nxt = 0
        self.countdown_duration = self.speed_presets[nxt]
        self.result_duration = max(1.2, round(self.countdown_duration * 0.7, 1))
        self.show_toast(f"นับถอยหลัง {self.countdown_duration:.1f} วินาที", color=AMBER)

    def reset_score(self):
        self.opp_history, self.recent_results = [], []
        self.round_no = self.wins = self.losses = self.draws = 0
        self.last_player_move = self.last_boss_move = self.last_result = "-"
        self.boss = self.make_boss(self.boss_index)
        self.cancel_round()
        self.sound.shoot()
        self.show_toast("รีเซ็ตคะแนนและความจำบอสแล้ว", color=ACCENT)

    def init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[ERROR] ไม่สามารถเปิดกล้อง ID {self.camera_id} ได้")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True

    # ------------------------------------------------------------------
    #  ลำดับของหนึ่งตา
    # ------------------------------------------------------------------
    def _reset_round_vars(self):
        self.ready_time = None         # เวลาเริ่มจับ reaction time
        self.carry_move = None         # ท่าที่ค้างมาจากตาที่แล้ว (ยังไม่นับเป็นการออกท่าใหม่)
        self.drop_frames = 0
        self.run_move, self.run_start, self.run_frames = None, 0.0, 0
        self.current_move = None       # ท่าที่ล็อกล่าสุด
        self.reaction_ms = None
        self.repeat_progress = 0.0
        self.thrown = False            # บอสออกท่าแล้วหรือยัง (ตอนนับถึง "ออก!")
        self.pending_boss_move = None
        self.reveal_time = 0.0

    def begin_round(self):
        """เริ่มตาใหม่: รอให้เห็นท่าก่อน ยังไม่นับถอยหลัง (ไม่ต้องลดมือ)"""
        if self.state in (STATE_IDLE, STATE_RESULT):
            self._reset_round_vars()
            self.ready_time = time.time()
            self.carry_move = self.last_raw_move
            self.state = STATE_WAIT

    def cancel_round(self):
        if self.state in (STATE_WAIT, STATE_COUNTDOWN):
            self.state = STATE_IDLE
        self._reset_round_vars()

    def _track_run(self, raw_move, now):
        """นับว่าท่าเดียวกันอยู่ติดกันนานพอหรือยัง คืนค่าท่าที่ 'นิ่ง' แล้ว หรือ None"""
        if raw_move is None:
            self.run_move, self.run_frames = None, 0
            return None
        if raw_move != self.run_move:
            self.run_move, self.run_start, self.run_frames = raw_move, now, 1
        else:
            self.run_frames += 1
        if self.run_frames >= LOCK_MIN_FRAMES and now - self.run_start >= LOCK_SEC:
            return self.run_move
        return None

    def _lock(self, move, reaction_ms, now):
        self.current_move = move
        self.reaction_ms = reaction_ms
        self.state = STATE_COUNTDOWN
        self.countdown_start = now
        self.last_tick = None

    def update_wait(self, raw_move, now):
        """
        ไม่ต้องลดมือระหว่างตา:
          - เปลี่ยนเป็นท่าใหม่ (ต่างจากท่าที่ค้างอยู่) -> ล็อกทันที, reaction = จากเริ่มตาถึงตอนท่าใหม่ปรากฏ
          - ค้างท่าเดิมไว้ REPEAT_HOLD_SEC          -> ออกท่าเดิมซ้ำ (reaction ไม่บันทึก เพราะไม่ได้ขยับ)
          - ลดมือ/ไม่เห็นท่า แล้วทำท่าใหม่          -> ล็อกเมื่อนิ่ง, reaction = ตามจริง
        """
        if raw_move is None:
            self.drop_frames += 1
            if self.drop_frames >= DROP_FRAMES:
                self.carry_move = None
        else:
            self.drop_frames = 0
        stable = self._track_run(raw_move, now)
        self.repeat_progress = 0.0
        if not stable:
            return
        if stable != self.carry_move:
            self._lock(stable, max(0.0, self.run_start - self.ready_time) * 1000.0, now)
        else:
            held = now - max(self.run_start, self.ready_time)
            self.repeat_progress = held / REPEAT_HOLD_SEC
            if held >= REPEAT_HOLD_SEC:
                self._lock(stable, None, now)

    def update_countdown(self, raw_move, now):
        duration = max(0.5, self.countdown_duration)
        elapsed = now - self.countdown_start
        if not self.thrown:
            # เปลี่ยนท่าระหว่างนับได้ -> reaction time นับถึงท่าสุดท้าย
            stable = self._track_run(raw_move, now)
            if stable and stable != self.current_move:
                self.current_move = stable
                self.reaction_ms = max(0.0, self.run_start - self.ready_time) * 1000.0
            if elapsed >= duration:
                # "ออก!" -> ล็อกท่าผู้เล่น แล้วบอสออกท่า (บอสเห็นแค่ประวัติก่อนหน้า)
                self.thrown = True
                self.pending_boss_move = self.boss.choose(self.opp_history)
                self.reveal_time = now

        number = max(0, 3 - int(elapsed / duration * 3)) if elapsed < duration else 0
        if number != self.last_tick:
            self.last_tick = number
            (self.sound.tick if number > 0 else self.sound.shoot)()
        if self.thrown and now - self.reveal_time >= SHOOT_HOLD_SEC:
            self.execute_showdown(self.current_move, self.pending_boss_move)
        return number, min(1.0, elapsed / duration)

    def execute_showdown(self, player_move, boss_move):
        result = judge(player_move, boss_move)
        self.opp_history.append(player_move)
        self.boss.learn({"win": -1, "draw": 0, "lose": 1}[result], self.opp_history)

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
        self.recent_results.append(result)

        self.last_player_move, self.last_boss_move, self.last_result = player_move, boss_move, result
        self.last_reaction_ms = self.reaction_ms
        self.result_start = time.time()
        self.state = STATE_RESULT
        self.log_telemetry(player_move, boss_move, result, self.reaction_ms)
        rt = f"{self.reaction_ms:.0f} ms" if self.reaction_ms is not None else "-"
        print(f"[ตา {self.round_no}] คุณ: {player_move} | บอส: {boss_move} -> {result} | reaction {rt}")

    # ------------------------------------------------------------------
    #  ลูปหลัก
    # ------------------------------------------------------------------
    def run(self):
        if not self.init_camera():
            return
        window = "Read-a-Rock: AI Arena"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("\nเกมพร้อมแล้ว — กด H ในหน้าต่างเกมเพื่อดูปุ่มทั้งหมด, Q เพื่อออก\n")
        if self.initial_model_toast:
            self.show_toast(self.initial_model_toast, color=ACCENT, duration=4.0)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[ERROR] อ่านภาพจากกล้องไม่ได้")
                break
            frame = cv2.flip(frame, 1)
            frame, hands = self.detector.find_hands(frame, draw=self.draw_landmarks)

            hand_visible = bool(hands)
            raw_move, shown_move, fingers = None, None, 0
            if hands:
                g = self.detector.classify_gesture(hands[0])
                raw_move = g["gesture_thai"] if g["gesture_thai"] in MOVES else None
                shown_move = g["stable_gesture"] if g["stable_gesture"] in MOVES else None
                fingers = g["total_open"]

            self.last_raw_move = raw_move
            if frame.shape[1] < DISPLAY_MIN_WIDTH:
                s = DISPLAY_MIN_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            now = time.time()
            overlay_open = self.menu_open or self.help_open

            # ---------- state machine ----------
            if not overlay_open:
                if self.state == STATE_WAIT:
                    self.update_wait(raw_move, now)
                    if self.state == STATE_WAIT:
                        lock = 0.0
                        if self.run_move and self.run_move != self.carry_move:
                            lock = (now - self.run_start) / LOCK_SEC
                        waited = (now - self.ready_time) if self.ready_time else 0.0
                        self.ui.draw_wait_gesture(frame, self.carry_move, waited, lock, self.repeat_progress)
                if self.state == STATE_COUNTDOWN:
                    number, prog = self.update_countdown(raw_move, now)
                    if self.state == STATE_COUNTDOWN:
                        self.ui.draw_countdown(frame, number, prog, self.current_move, self.reaction_ms)
                if self.state == STATE_RESULT:
                    if now - self.result_start > self.result_duration:
                        if self.auto_mode:
                            self.begin_round()
                        else:
                            self.state = STATE_IDLE
                    else:
                        self.ui.draw_result_banner(frame, self.last_result, self.last_player_move,
                                                   self.last_boss_move, self.last_reaction_ms, self.auto_mode)
                if self.state == STATE_IDLE:
                    self.ui.draw_idle_prompt(frame)

            # ---------- HUD ----------
            locked = self.state == STATE_COUNTDOWN and shown_move == self.current_move
            self.ui.draw_player_card(frame, self.current_move if self.state == STATE_COUNTDOWN and not shown_move
                                     else shown_move, fingers, hand_visible, self.player_name, locked)
            if self.state == STATE_COUNTDOWN and self.thrown:
                b_move, b_mode, b_t = self.pending_boss_move, "reveal", now - self.reveal_time
            elif self.state == STATE_COUNTDOWN:
                b_move, b_mode, b_t = None, "pump", (now - self.countdown_start) / max(0.5, self.countdown_duration)
            elif self.state == STATE_RESULT:
                b_move, b_mode, b_t = self.last_boss_move, "reveal", now - self.reveal_time
            else:
                b_move, b_mode, b_t = None, "idle", 0.0
            self.ui.draw_boss_card(frame, b_move, getattr(self.boss, "desc", ""), b_mode, b_t)
            self.ui.draw_top_bar(frame, self.round_no, self.wins, self.losses, self.draws, self.boss.name,
                                 self.auto_mode, self.sound.enabled, self.countdown_duration, self.recent_results)
            self.ui.draw_footer_controls(frame)
            if self.menu_open:
                self.ui.draw_boss_selection_menu(frame, GAME_BOSSES, self.boss_index)
            if self.help_open:
                self.ui.draw_help_overlay(frame)
            if now < self.toast_expiry:
                self.ui.draw_toast_notification(frame, self.toast_text, color=self.toast_color)

            cv2.imshow(window, frame)
            if not self.handle_key(cv2.waitKey(1) & 0xFF):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.detector.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nขอบคุณที่เล่นครับ!")

    def _set_overlay(self, menu=None, help_=None):
        was_open = self.menu_open or self.help_open
        if menu is not None:
            self.menu_open = menu
        if help_ is not None:
            self.help_open = help_
        now_open = self.menu_open or self.help_open
        if now_open and not was_open:
            self.cancel_round()                 # หยุดตาที่ค้างอยู่ระหว่างเปิดเมนู
        elif was_open and not now_open and self.auto_mode:
            self.begin_round()

    def handle_key(self, key):
        """คืนค่า False เมื่อต้องการออกจากเกม"""
        if key == 255:
            return True
        k = chr(key).lower() if key < 128 else ""
        if key == 27 or k == "q":
            return False
        if k == " ":
            if self.menu_open or self.help_open:
                self._set_overlay(menu=False, help_=False)
            else:
                self.begin_round()
        elif k == "a":
            self.auto_mode = not self.auto_mode
            self.show_toast("ออโต้: เริ่มตาใหม่เองเมื่อเห็นท่า" if self.auto_mode else "ปิดออโต้",
                            color=GREEN if self.auto_mode else AMBER)
            if self.auto_mode:
                self.begin_round()
        elif k == "t":
            self.cycle_speed()
        elif k == "b":
            self._set_overlay(menu=not self.menu_open, help_=False)
        elif k == "h":
            self._set_overlay(help_=not self.help_open, menu=False)
        elif "1" <= k <= "9":
            self.switch_boss(int(k) - 1)
            self._set_overlay(menu=False)
        elif k == "s":
            self.export_current_boss()
        elif k == "l":
            self.import_saved_boss()
            self._set_overlay(menu=False)
        elif k == "r":
            self.reset_score()
        elif k == "m":
            self.sound.enabled = not self.sound.enabled
        elif k == "g":
            self.draw_landmarks = not self.draw_landmarks
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-Paper-Scissors AI Arena (Webcam + MediaPipe + Adaptive AI)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--countdown", type=float, default=3.0, help="Countdown seconds (default 3.0)")
    parser.add_argument("--player", type=str, default=None, help="Player name (e.g. --player Ken)")
    parser.add_argument("--model", type=str, default=None, help="Path to model JSON (e.g. models/ds_champion_ken.json)")
    parser.add_argument("--auto", action="store_true", help="เริ่มในโหมดออโต้")
    args = parser.parse_args()

    player_name = args.player
    if not player_name:
        try:
            player_name = input("ชื่อผู้เล่น (Enter = 'Player'): ").strip() or "Player"
        except (KeyboardInterrupt, EOFError):
            player_name = "Player"

    model_path = args.model
    if not model_path and player_name:
        candidate = f"models/ds_champion_{player_name.strip().lower().replace(' ', '_')}.json"
        if os.path.exists(candidate):
            try:
                ans = input(f"พบโมเดลเฉพาะบุคคล '{candidate}' — โหลดมาเป็นคู่ต่อสู้ไหม? (Y/n): ").strip().lower()
                if ans in ("", "y", "yes"):
                    model_path = candidate
            except (KeyboardInterrupt, EOFError):
                pass

    game = RockPaperScissorsGame(camera_id=args.camera, countdown=args.countdown,
                                 player_name=player_name, model_path=model_path)
    if args.auto:
        game.auto_mode = True
        game.begin_round()
    game.run()

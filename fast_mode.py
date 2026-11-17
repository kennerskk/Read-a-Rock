# =====================================================
#  fast_mode.py — โหมดประลองด่วน (ไม่ต้องเปิดกล้อง + ปรับเวลาได้)
#
#  คุณสมบัติ:
#  - ไม่ต้องใช้กล้อง Webcam (เลือกท่าด้วยแป้นพิมพ์ 1=ค้อน, 2=กรรไกร, 3=กระดาษ)
#  - ปรับความเร็วเวลาต่อตาได้อิสระ (0 วิ = รู้ผลทันที, 0.5 วิ, 1.0 วิ, 2.0 วิ, 3.0 วิ)
#  - บันทึกข้อมูล Telemetry ลง data/players/<ชื่อผู้เล่น>.csv อัตโนมัติทุกตา (แยกไฟล์ต่อคน)
#  - รองรับการบันทึกสมองบอส [S] และโหลดโมเดลกลับมาสู้ [L]
#  - เลือกสลับบอส AI ได้ครบ 9 ตัว (9 = Lite Ensemble) หรือโหลด Custom Model
#  - reaction_time_ms = เวลาตั้งแต่ขึ้นคำถามจนกด Enter ท่า (วัดจริง)
# =====================================================
import os
import sys
import time
import csv
import argparse
from datetime import datetime
from collections import Counter
from typing import Optional

from adaptive_arena import GAME_BOSSES as BOSSES, DEFAULT_GAME_BOSS, judge, MOVES, BEATS, COUNTER
from sound_effects import SoundManager
from model_io import save_boss_model, load_boss_model, list_saved_models
from telemetry import append_round, player_file, canonical_name, ask_player_name, validate_name
from ui_renderer import short_boss_name


class FastArena:
    def __init__(self, pace_sec: float = 0.5, boss_idx: int = DEFAULT_GAME_BOSS, sound_on: bool = True, player_name: str = "Player", model_path: Optional[str] = None):
        self.pace_sec = max(0.0, float(pace_sec))
        self.boss_idx = boss_idx
        self.player_name = str(player_name).strip() if player_name else "Player"
        self.sound = SoundManager(enabled=sound_on)

        if model_path and os.path.exists(model_path):
            try:
                self.boss = load_boss_model(model_path)
                b_name = getattr(self.boss, "name", os.path.basename(model_path))
                print(f"✅ โหลดโมเดลคู่ปรับสำเร็จ: {b_name} ({model_path})")
            except Exception as e:
                print(f"⚠️ ไม่สามารถโหลดโมเดล {model_path} ได้: {e}")
                self.boss = BOSSES[self.boss_idx]()
        else:
            self.boss = BOSSES[self.boss_idx]()

        self.opp_history = []
        self.round_no = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        self.player_name = canonical_name(self.player_name)
        self.telemetry_path = player_file(self.player_name)

    def log_telemetry(self, player_move: str, ai_move: str, result: str, reaction_ms: Optional[float] = None):
        """บันทึกข้อมูลเพื่อนำไปใช้เทรนใน Data Science Pipeline ต่อได้ทันที"""
        try:
            append_round(self.player_name, [datetime.now().isoformat(), self.round_no, self.player_name,
                                            player_move, ai_move, result,
                                            "" if reaction_ms is None else int(round(reaction_ms))])
        except Exception as e:
            print(f"[Telemetry Error] {e}")

    def switch_boss(self, index: int):
        if 0 <= index < len(BOSSES):
            self.boss_idx = index
            self.boss = BOSSES[self.boss_idx]()
            self.sound.change_boss()
            print(f"\n🔄 สลับไปเจอบอส: {self.boss.name}")

    def load_custom_model(self):
        models = list_saved_models()
        if not models:
            print("\n⚠️ ยังไม่มีโมเดลที่บันทึกไว้ในโฟลเดอร์ models/")
            return

        print("\n📂 รายการโมเดลที่พบใน models/:")
        for i, m in enumerate(models, start=1):
            name = m.get("custom_name") or m.get("boss_name") or m.get("filename")
            print(f"  [{i}] {name}  (ไฟล์: {m['filename']})")

        choice = input("เลือกหมายเลขโมเดลที่ต้องการโหลด (หรือกด Enter เพื่อยกเลิก): ").strip()
        if not choice.isdigit():
            return

        idx = int(choice) - 1
        if 0 <= idx < len(models):
            try:
                loaded = load_boss_model(models[idx]["filepath"])
                self.boss = loaded
                self.sound.change_boss()
                print(f"✅ โหลดโมเดลสำเร็จ: {models[idx]['filename']}")
            except Exception as e:
                print(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")

    def save_current_model(self):
        ai_wr = (self.losses / self.round_no * 100) if self.round_no > 0 else 0.0
        custom_name = input("ตั้งชื่อโมเดล (กด Enter เพื่อใช้ชื่ออัตโนมัติ): ").strip()
        meta = {
            "custom_name": custom_name if custom_name else f"บอสแก้ทาง {self.player_name} (Winrate {ai_wr:.0f}%)",
            "target_player": self.player_name,
            "stats": {
                "total_rounds": self.round_no,
                "player_wins": self.wins,
                "ai_wins": self.losses,
                "draws": self.draws,
                "ai_winrate_pct": ai_wr
            }
        }
        path = save_boss_model(self.boss, metadata=meta)
        print(f"💾 บันทึกโมเดลเรียบร้อยที่: {path}")

    def change_player_name(self):
        print(f"\n👤 ผู้เล่นปัจจุบัน: {self.player_name}")
        new_name = input("ใส่ชื่อผู้เล่นใหม่ (กด Enter เพื่อยกเลิก): ").strip()
        if new_name and validate_name(new_name):
            print(f"⚠️ {validate_name(new_name)} ยังไม่เปลี่ยนชื่อ")
            return
        if new_name:
            if self.round_no:
                ai_wr = self.losses / self.round_no * 100
                print(f"📋 สรุป {self.player_name}: {self.round_no} ตา | ชนะ {self.wins} | บอสชนะ {self.losses} ({ai_wr:.0f}%)")
            self.player_name = canonical_name(new_name)
            self.telemetry_path = player_file(self.player_name)
            # คนใหม่ = บอสต้องเริ่มเรียนรู้ใหม่ ไม่ให้นิสัยคนก่อนปนมา
            self.opp_history = []
            self.round_no = self.wins = self.losses = self.draws = 0
            self.boss = BOSSES[self.boss_idx]()
            print(f"✅ เปลี่ยนผู้เล่นเป็น: {self.player_name} แล้ว! รีเซ็ตคะแนนและความจำบอส ตาต่อไปจะบันทึกในชื่อนี้")

    def set_pace(self):
        print(f"\n⏱️ เวลาต่อตาปัจจุบัน: {self.pace_sec:.2f} วินาที")
        print("คำแนะนำ:")
        print("  0   = รู้ผลทันที (Instant Mode — เหมาะสำหรับเก็บ Data หลายสิบตาไวๆ)")
        print("  0.5 = เร็วมาก (กำลังสนุก)")
        print("  1.0 = ปานกลาง")
        print("  2.0 = ปกติ")
        print("  3.0 = โหมดช้า (เท่ากับหน้ากล้อง)")
        raw = input("ใส่เวลาต่อตาที่ต้องการ (วินาที): ").strip()
        try:
            val = float(raw)
            if val < 0:
                print("❌ เวลาต้องไม่ติดลบ")
            else:
                self.pace_sec = val
                print(f"✅ ปรับเวลาต่อตาเป็น {self.pace_sec:.2f} วินาทีแล้ว!")
        except ValueError:
            print("❌ กรุณาใส่ตัวเลข")

    def play_round(self, player_move: str, reaction_ms: Optional[float] = None):
        # 1. บอสตัดสินใจจากประวัติก่อนหน้านี้
        boss_move = self.boss.choose(self.opp_history)

        # 2. จำลองเวลาเป่ายิ้งฉุบตาม pace_sec ที่ผู้เล่นกำหนด
        if self.pace_sec > 0:
            steps = min(3, max(1, int(self.pace_sec / 0.4)))
            delay_step = self.pace_sec / steps
            for i in range(steps, 0, -1):
                print(f"  ⏱️ {i}...", end="\r", flush=True)
                self.sound.tick()
                time.sleep(delay_step)
            print("  🥊 ยิ้งฉุบ!         ")
            self.sound.shoot()
        else:
            self.sound.shoot()

        # 3. ตัดสินผลแพ้ชนะ
        result = judge(player_move, boss_move)
        self.opp_history.append(player_move)

        # 4. ให้บอสเรียนรู้
        reward = {"win": -1, "draw": 0, "lose": 1}[result]
        self.boss.learn(reward, self.opp_history)

        # 5. อัปเดตสถิติ
        self.round_no += 1
        if result == "win":
            self.wins += 1
            self.sound.win()
            tag = "คุณชนะ! 🎉"
        elif result == "lose":
            self.losses += 1
            self.sound.lose()
            tag = "บอสชนะ! 🤖"
        else:
            self.draws += 1
            self.sound.draw()
            tag = "เสมอ! 🤝"

        # 6. บันทึก Telemetry อัตโนมัติ
        self.log_telemetry(player_move, boss_move, result, reaction_ms)

        wr = (self.wins / self.round_no) * 100
        ai_wr = (self.losses / self.round_no) * 100
        print(f"👉 [ตา {self.round_no}] คุณ: {player_move}  vs  บอส: {boss_move}  ->  {tag}")
        print(f"   📊 ชนะ {self.wins}/{self.round_no} ({wr:.1f}%) | บอสชนะ: {ai_wr:.1f}% | เสมอ: {self.draws}")

    def run(self):
        move_map = {
            "1": "ค้อน",
            "2": "กรรไกร",
            "3": "กระดาษ",
            "r": "ค้อน",
            "s": "กรรไกร",  # scissors
            "p": "กระดาษ"   # paper
        }

        print("=" * 65)
        print("⚡ FAST ARENA — โหมดประลองด่วน (No-Cam + Custom Speed)")
        print(f"👤 ผู้เล่น: {self.player_name}")
        print(f"👾 บอสปัจจุบัน: {self.boss.name}")
        print(f"⏱️ เวลาต่อตา: {self.pace_sec:.2f} วินาที")
        print("=" * 65)
        print("ปุ่มกดเลือกท่า:")
        print("  [1] ค้อน ✊   [2] กรรไกร ✌️   [3] กระดาษ ✋")
        print("ปุ่มจัดการ:")
        print(f"  [u] เปลี่ยนชื่อผู้เล่น   [t] ปรับเวลาต่อตา    [b] สลับบอส (1-{len(BOSSES)})")
        print("  [s] บันทึกโมเดล        [l] โหลดโมเดล        [m] เปิด/ปิดเสียง")
        print("  [r] รีเซ็ตคะแนน        [q] สรุปผลและเลิกเล่น\n")

        while True:
            prompt = f"[{self.player_name} | ตา {self.round_no + 1} | {self.pace_sec:.1f}s] ออกท่า (1/2/3): "
            try:
                t_prompt = time.perf_counter()
                cmd = input(prompt).strip().lower()
                decision_ms = (time.perf_counter() - t_prompt) * 1000.0
            except (KeyboardInterrupt, EOFError):
                break

            if not cmd:
                continue

            if cmd == "q":
                break
            elif cmd in ["u", "user", "name"]:
                self.change_player_name()
            elif cmd == "t":
                self.set_pace()
            elif cmd == "b":
                print("\nรายชื่อบอส:")
                for i, b in enumerate(BOSSES, start=1):
                    print(f"  [{i}] {short_boss_name(b.name)}  —  {b.desc}")
                b_choice = input(f"เลือกหมายเลขบอส (1-{len(BOSSES)}): ").strip()
                if b_choice.isdigit() and 1 <= int(b_choice) <= len(BOSSES):
                    self.switch_boss(int(b_choice) - 1)
            elif cmd == "s":
                self.save_current_model()
            elif cmd == "l":
                self.load_custom_model()
            elif cmd == "m":
                self.sound.enabled = not self.sound.enabled
                print(f"🔊 เสียงเอฟเฟกต์: {'เปิด' if self.sound.enabled else 'ปิด'}")
            elif cmd == "r":
                self.opp_history = []
                self.round_no = 0
                self.wins = 0
                self.losses = 0
                self.draws = 0
                self.boss = BOSSES[self.boss_idx]()
                print("🔄 รีเซ็ตคะแนนและสมองบอสเรียบร้อย!")
            elif cmd in move_map:
                self.play_round(move_map[cmd], decision_ms)
            else:
                print("⚠️ กด 1 (ค้อน), 2 (กรรไกร), 3 (กระดาษ) หรือกด 't' เพื่อปรับเวลา")

        print("\n" + "=" * 50)
        print("🏁 สรุปผลการเล่น Fast Arena")
        if self.round_no > 0:
            wr = (self.wins / self.round_no) * 100
            ai_wr = (self.losses / self.round_no) * 100
            print(f"เล่นทั้งหมด: {self.round_no} ตา")
            print(f"คุณชนะ: {self.wins} ({wr:.1f}%) | บอสชนะ: {self.losses} ({ai_wr:.1f}%) | เสมอ: {self.draws}")
            print(f"📁 บันทึกข้อมูล Telemetry แล้วที่: {self.telemetry_path}")
            save_opt = input("\n💾 ต้องการเซฟโมเดลบอสตัวนี้ไว้สู้รอบหน้าไหม? (y/n): ").strip().lower()
            if save_opt == "y":
                self.save_current_model()
        else:
            print("ยังไม่มีการเล่น")
        print("ขอบคุณที่ร่วมสนุกครับ! 👋\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Arena — โหมดประลองด่วนเป่ายิ้งฉุบ (ไม่ต้องเปิดกล้อง)")
    parser.add_argument("--player", type=str, default=None, help="ชื่อผู้เล่น (เช่น --player Ken)")
    parser.add_argument("--model", type=str, default=None, help="พาธไฟล์โมเดล JSON ที่ต้องการโหลด (เช่น models/ds_champion_ken.json)")
    parser.add_argument("--pace", type=float, default=0.5, help="เวลาต่อตาเป็นวินาที (เช่น 0 = ทันที, 0.5, 1.0, 3.0)")
    parser.add_argument("--boss", type=int, default=DEFAULT_GAME_BOSS + 1, help="หมายเลขบอสเริ่มต้น: 1=Markov 2=N-gram 3=ML 4=Lite Ensemble (ค่าเริ่มต้น)")
    parser.add_argument("--no-sound", action="store_true", help="ปิดเสียงเอฟเฟกต์")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    player_name = args.player
    if player_name and validate_name(player_name):
        print(f"⚠️ {validate_name(player_name)}")
        player_name = None
    if not player_name:
        player_name = ask_player_name()          # Enter = visitor_<วันที่>_<ลำดับ> อัตโนมัติ

    model_path = args.model
    if not model_path and player_name:
        clean_name = player_name.strip().lower().replace(" ", "_")
        candidate = f"models/ds_champion_{clean_name}.json"
        if os.path.exists(candidate):
            try:
                ans = input(f"🤖 ตรวจพบโมเดลแชมเปียนเฉพาะบุคคล: '{candidate}'\n   ต้องการโหลดมาเป็นคู่ต่อสู้เลยไหม? (Y/n): ").strip().lower()
                if ans in ["", "y", "yes"]:
                    model_path = candidate
            except (KeyboardInterrupt, EOFError):
                pass

    boss_index = max(0, min(len(BOSSES) - 1, args.boss - 1))
    arena = FastArena(pace_sec=args.pace, boss_idx=boss_index, sound_on=not args.no_sound, player_name=player_name, model_path=model_path)
    arena.run()

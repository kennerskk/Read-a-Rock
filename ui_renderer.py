# =====================================================
#  ui_renderer.py — วาด UI ภาษาไทยและ HUD แดชบอร์ดใน OpenCV
# =====================================================
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIRenderer:
    def __init__(self):
        # ค้นหาฟอนต์ที่รองรับภาษาไทยในระบบต่างๆ (Windows, Linux, macOS)
        font_candidates = [
            "C:/Windows/Fonts/leelawad.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/System/Library/Fonts/Supplemental/Thonburi.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/thai/Garuda.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        self.font_path = None
        for fc in font_candidates:
            if os.path.exists(fc):
                self.font_path = fc
                break

        # แคชฟอนต์ขนาดต่างๆ เพื่อประสิทธิภาพสูงสุด
        self._font_cache = {}

    def get_font(self, size):
        if size not in self._font_cache:
            try:
                if self.font_path:
                    self._font_cache[size] = ImageFont.truetype(self.font_path, size)
                else:
                    self._font_cache[size] = ImageFont.load_default()
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def draw_thai_text(self, img, text, pos, font_size=22, color=(255, 255, 255), shadow=True, shadow_color=(0, 0, 0)):
        """
        วาดข้อความภาษาไทยลงบนภาพ OpenCV (BGR)
        pos: (x, y)
        color: (B, G, R)
        """
        # แปลง BGR เป็น RGB สำหรับ PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self.get_font(font_size)

        x, y = pos
        # วาดเงาเพื่อให้อ่านง่าย
        if shadow:
            s_rgb = (shadow_color[2], shadow_color[1], shadow_color[0])
            draw.text((x + 1, y + 1), text, font=font, fill=s_rgb)
            draw.text((x + 2, y + 2), text, font=font, fill=s_rgb)

        c_rgb = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=c_rgb)

        # แปลงกลับเป็น OpenCV BGR
        res_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np.copyto(img, res_bgr)
        return img

    def draw_panel(self, img, x, y, w, h, bg_color=(20, 24, 32), alpha=0.75, border_color=(0, 200, 255), border_thickness=2):
        """
        วาดกล่องโปร่งแสงพร้อมขอบสีสันสวยงาม
        """
        ih, iw, _ = img.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)

        if x2 <= x1 or y2 <= y1:
            return img

        sub_img = img[y1:y2, x1:x2]
        color_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        blended = cv2.addWeighted(sub_img, 1.0 - alpha, color_rect, alpha, 0)
        img[y1:y2, x1:x2] = blended

        if border_color and border_thickness > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness)

        return img

    def draw_badge(self, img, text, pos, bg_color=(40, 44, 52), text_color=(255, 255, 255), font_size=18, padding=6):
        """วาด Badge ข้อความพร้อมพื้นหลัง"""
        x, y = pos
        font = self.get_font(font_size)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        bx1, by1 = x, y
        bx2, by2 = x + tw + padding * 2, y + th + padding * 2
        self.draw_panel(img, bx1, by1, bx2 - bx1, by2 - by1, bg_color=bg_color, alpha=0.85, border_color=None)
        self.draw_thai_text(img, text, (x + padding, y + padding - 2), font_size=font_size, color=text_color, shadow=False)
        return bx2

    def draw_header_scoreboard(self, img, round_no, wins, losses, draws, winrate, boss_name, auto_mode=False, sound_on=True):
        """แถบสถานะด้านบน: ข้อมูลบอส และ คะแนนแพ้ชนะ"""
        w = img.shape[1]
        self.draw_panel(img, 10, 10, w - 20, 68, bg_color=(15, 18, 24), alpha=0.85, border_color=(70, 80, 95), border_thickness=1)

        # บอสที่กำลังเจอ
        self.draw_thai_text(img, f"🤖 บอส AI: {boss_name}", (25, 18), font_size=20, color=(0, 220, 255))

        # คะแนน
        stats_text = f"ตาที่ {round_no}  |  ชนะ: {wins}  แพ้: {losses}  เสมอ: {draws}  |  Winrate: {winrate:.1f}%"
        self.draw_thai_text(img, stats_text, (25, 46), font_size=16, color=(240, 240, 240))

        # Badge โหมดอัตโนมัติและเสียง
        auto_text = "AUTO: ON" if auto_mode else "AUTO: OFF"
        auto_color = (50, 220, 100) if auto_mode else (140, 140, 140)
        self.draw_badge(img, auto_text, (w - 230, 24), bg_color=(25, 30, 38), text_color=auto_color, font_size=16)

        sound_text = "SOUND: ON" if sound_on else "MUTED"
        sound_color = (0, 200, 255) if sound_on else (120, 120, 120)
        self.draw_badge(img, sound_text, (w - 120, 24), bg_color=(25, 30, 38), text_color=sound_color, font_size=16)

    def draw_player_card(self, img, detected_gesture, total_fingers, is_live=True):
        """การ์ดแสดงท่าที่ตรวจจับได้ของผู้เล่นแบบ Real-time"""
        h = img.shape[0]
        card_w = 310
        card_h = 100
        x = 20
        y = h - card_h - 60

        # เลือกสีตามท่า
        color_map = {
            "ค้อน": (0, 165, 255),    # สีส้มทอง
            "กรรไกร": (255, 120, 0),  # สีฟ้าคราม
            "กระดาษ": (0, 255, 128),  # สีเขียวสด
            "รอท่า...": (150, 150, 150)
        }
        b_color = color_map.get(detected_gesture, (150, 150, 150))

        self.draw_panel(img, x, y, card_w, card_h, bg_color=(18, 22, 30), alpha=0.85, border_color=b_color, border_thickness=2)

        icon_map = {"ค้อน": "✊", "กรรไกร": "✌️", "กระดาษ": "✋", "รอท่า...": "❓"}
        icon = icon_map.get(detected_gesture, "❓")

        self.draw_thai_text(img, "👤 ผู้เล่น (ตรวจจับนิ้วมือ):", (x + 14, y + 12), font_size=15, color=(180, 190, 205))
        self.draw_thai_text(img, f"{icon} {detected_gesture}", (x + 14, y + 36), font_size=28, color=b_color)
        self.draw_thai_text(img, f"นิ้วที่กาง: {total_fingers} นิ้ว (0=ค้อน, 2=กรรไกร, 5=กระดาษ)", (x + 14, y + 74), font_size=13, color=(160, 170, 185))

    def draw_boss_card(self, img, boss_move, boss_desc=""):
        """การ์ดแสดงท่าของ AI บอส"""
        w, h = img.shape[1], img.shape[0]
        card_w = 310
        card_h = 100
        x = w - card_w - 20
        y = h - card_h - 60

        color_map = {
            "ค้อน": (0, 165, 255),
            "กรรไกร": (255, 120, 0),
            "กระดาษ": (0, 255, 128),
            "-": (120, 120, 120)
        }
        b_color = color_map.get(boss_move, (120, 120, 120))

        self.draw_panel(img, x, y, card_w, card_h, bg_color=(18, 22, 30), alpha=0.85, border_color=b_color, border_thickness=2)

        icon_map = {"ค้อน": "✊", "กรรไกร": "✌️", "กระดาษ": "✋", "-": "🎲"}
        icon = icon_map.get(boss_move, "🎲")

        self.draw_thai_text(img, "🤖 AI บอส (Adaptive Arena):", (x + 14, y + 12), font_size=15, color=(180, 190, 205))
        if boss_move == "-":
            self.draw_thai_text(img, "กำลังคิดคำนวณ...", (x + 14, y + 38), font_size=22, color=(140, 150, 165))
        else:
            self.draw_thai_text(img, f"{icon} {boss_move}", (x + 14, y + 36), font_size=28, color=b_color)
            self.draw_thai_text(img, f"กลยุทธ์: {boss_desc[:28]}...", (x + 14, y + 74), font_size=13, color=(160, 170, 185))

    def draw_countdown(self, img, count_val):
        """แสดงเวลานับถอยหลัง 3.. 2.. 1.. เป่ายิ้งฉุบ! กลางหน้าจอ"""
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2

        # วงกลม/กล่องตรงกลาง
        box_w, box_h = 280, 140
        bx = cx - box_w // 2
        by = cy - box_h // 2 - 20

        color_stage = {
            "3": (0, 220, 255),    # เหลืองทอง
            "2": (0, 160, 255),    # ส้ม
            "1": (50, 80, 255),    # แดงส้ม
            "เป่ายิ้งฉุบ!": (0, 255, 100) # เขียวสด
        }
        color = color_stage.get(count_val, (255, 255, 255))

        self.draw_panel(img, bx, by, box_w, box_h, bg_color=(10, 12, 16), alpha=0.9, border_color=color, border_thickness=3)

        if count_val == "เป่ายิ้งฉุบ!":
            self.draw_thai_text(img, count_val, (bx + 26, by + 42), font_size=42, color=color)
        else:
            self.draw_thai_text(img, f"เตรียมตัว...", (bx + 75, by + 18), font_size=20, color=(180, 190, 205))
            self.draw_thai_text(img, count_val, (cx - 24, by + 46), font_size=64, color=color)

    def draw_result_banner(self, img, result, player_move, boss_move):
        """แบนเนอร์แสดงผลลัพธ์รอบล่าสุดแบบสะดุดตา"""
        h, w, _ = img.shape
        cx, cy = w // 2, h // 2

        banner_w = 460
        banner_h = 160
        bx = cx - banner_w // 2
        by = cy - banner_h // 2 - 20

        if result == "win":
            title = "🎉 คุณชนะ! (You Win!)"
            t_color = (0, 255, 100)
            bg_border = (0, 255, 120)
        elif result == "lose":
            title = "❌ คุณแพ้! (AI Wins!)"
            t_color = (60, 80, 255)
            bg_border = (50, 70, 240)
        else:
            title = "🤝 เสมอ! (Draw!)"
            t_color = (0, 220, 255)
            bg_border = (0, 200, 240)

        self.draw_panel(img, bx, by, banner_w, banner_h, bg_color=(12, 14, 20), alpha=0.92, border_color=bg_border, border_thickness=3)
        self.draw_thai_text(img, title, (bx + 75, by + 20), font_size=32, color=t_color)

        vs_text = f"คุณออก: {player_move}   ⚡   บอสออก: {boss_move}"
        self.draw_thai_text(img, vs_text, (bx + 55, by + 74), font_size=22, color=(240, 240, 240))
        self.draw_thai_text(img, "กด [SPACE] เล่นตาถัดไป หรือรอเล่นอัตโนมัติ", (bx + 85, by + 118), font_size=15, color=(170, 180, 195))

    def draw_idle_prompt(self, img):
        """ข้อความแนะนำเมื่อรอให้ผู้เล่นกดเริ่มเล่น"""
        h, w, _ = img.shape
        cx = w // 2
        prompt_w = 460
        prompt_h = 50
        px = cx - prompt_w // 2
        py = h // 2 - 25

        self.draw_panel(img, px, py, prompt_w, prompt_h, bg_color=(15, 18, 25), alpha=0.85, border_color=(0, 200, 255), border_thickness=1)
        self.draw_thai_text(img, "🎮 กด [SPACE] หรือเปิดโหมด [A] เพื่อเริ่มเป่ายิ้งฉุบ!", (px + 28, py + 12), font_size=20, color=(0, 240, 255))

    def draw_footer_controls(self, img):
        """แถบปุ่มคีย์ลัดด้านล่างสุด"""
        w, h = img.shape[1], img.shape[0]
        self.draw_panel(img, 10, h - 45, w - 20, 36, bg_color=(15, 18, 24), alpha=0.9, border_color=(50, 60, 75), border_thickness=1)

        help_text = "[SPACE] เริ่ม | [A] ออโต้ | [T] สปีด | [1-8] บอส | [S] เซฟ | [L] โหลด | [R] รีเซ็ต | [Q] ออก"
        self.draw_thai_text(img, help_text, (20, h - 38), font_size=14, color=(190, 200, 215))

    def draw_toast_notification(self, img, text, color=(0, 255, 120)):
        """แบนเนอร์แจ้งเตือนชั่วคราว (Toast Message) กลางบนหน้าจอ"""
        h, w, _ = img.shape
        font = self.get_font(18)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        box_w = max(400, tw + 40)
        box_h = th + 24
        bx = (w - box_w) // 2
        by = 86
        self.draw_panel(img, bx, by, box_w, box_h, bg_color=(15, 20, 28), alpha=0.92, border_color=color, border_thickness=2)
        self.draw_thai_text(img, text, (bx + 20, by + 10), font_size=18, color=color, shadow=True)

    def draw_boss_selection_menu(self, img, bosses, current_idx):
        """หน้าต่างเลือกบอสเมื่อผู้เล่นกดเปลี่ยนบอส"""
        h, w, _ = img.shape
        mw = 660
        mh = 480
        mx = (w - mw) // 2
        my = (h - mh) // 2

        self.draw_panel(img, mx, my, mw, mh, bg_color=(12, 15, 22), alpha=0.96, border_color=(0, 220, 255), border_thickness=2)
        self.draw_thai_text(img, "🧠 เลือกโมเดลสมองบอส AI (กด 1 - 8 หรือ [L] โหลดโมเดล)", (mx + 25, my + 18), font_size=22, color=(0, 240, 255))

        y_offset = my + 60
        for i, b_cls in enumerate(bosses):
            is_active = (i == current_idx)
            text_color = (0, 255, 128) if is_active else (220, 220, 220)
            prefix = "▶ " if is_active else "  "

            b_name = getattr(b_cls, "name", f"บอสตัวที่ {i+1}")
            b_desc = getattr(b_cls, "desc", "")
            self.draw_thai_text(img, f"{prefix}{b_name}", (mx + 25, y_offset), font_size=17, color=text_color)
            self.draw_thai_text(img, f"     {b_desc[:46]}", (mx + 25, y_offset + 22), font_size=12, color=(160, 170, 180))
            y_offset += 44

        self.draw_thai_text(img, "กดหมายเลข [1 - 8] เพื่อเลือกบอส | [L] โหลดโมเดลจากไฟล์ | [SPACE] กลับ", (mx + 30, my + mh - 30), font_size=14, color=(255, 200, 80))

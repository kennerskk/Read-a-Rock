# =====================================================
#  ui_renderer.py — วาด HUD ภาษาไทยบนภาพกล้อง (OpenCV + PIL)
#
#  หลักการออกแบบ:
#   - แต่ละสถานะของเกมมีข้อความกลางจอเพียงข้อความเดียว บอกว่า "ต้องทำอะไรต่อ"
#   - ใช้สีสื่อความหมายคงที่: เขียว = ชนะ, แดง = แพ้, เหลือง = เสมอ, ฟ้า = ข้อมูล/บอท
#   - ไม่ใช้ emoji (ฟอนต์ไทยใน Windows วาด emoji ไม่ได้ จะขึ้นเป็นกล่องสี่เหลี่ยม)
#   - วาดตัวอักษรเฉพาะบริเวณที่จำเป็น (ROI) ไม่แปลงภาพทั้งเฟรม -> เร็วขึ้นมาก
# =====================================================
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from i18n import t, bot_desc

# สี (B, G, R)
WHITE = (245, 245, 245)
INK2 = (190, 196, 206)        # ข้อความรอง
MUTED = (135, 142, 152)       # ข้อความจาง
PANEL = (22, 24, 30)
PANEL_LINE = (70, 76, 88)
ACCENT = (240, 190, 40)       # ฟ้า (ข้อมูล / บอส)
GREEN = (110, 210, 60)
RED = (70, 80, 235)
AMBER = (40, 190, 245)

MOVE_COLOR = {
    "ค้อน": (60, 150, 245),     # ส้ม
    "กรรไกร": (235, 150, 60),   # ฟ้า
    "กระดาษ": (120, 210, 90),   # เขียว
}
EMOJI = {"ค้อน": "✊", "กรรไกร": "✌", "กระดาษ": "✋"}
RESULT_STYLE = {
    "win":  ("res.win", GREEN),
    "lose": ("res.lose", RED),
    "draw": ("res.draw", AMBER),
}


def short_boss_name(name: str) -> str:
    """ '6. Ensemble/Meta (เลือก...)' -> 'Ensemble/Meta' """
    s = re.sub(r"^\s*\d+\.\s*", "", str(name))
    if not s.startswith("Lite Ensemble") or "+" not in s:
        s = s.split(" (")[0].strip()
    return s or str(name)


class UIRenderer:
    def __init__(self):
        fonts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
        regular = [os.path.join(fonts, "Prompt-Regular.ttf"), "C:/Windows/Fonts/LeelawUI.ttf", "C:/Windows/Fonts/leelawad.ttf",
                   "C:/Windows/Fonts/tahoma.ttf",
                   "/System/Library/Fonts/Supplemental/Thonburi.ttc",
                   "/usr/share/fonts/truetype/thai/Garuda.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        bold = [os.path.join(fonts, "Prompt-Bold.ttf"), "C:/Windows/Fonts/LeelaUIb.ttf", "C:/Windows/Fonts/leelawdb.ttf",
                "C:/Windows/Fonts/tahomabd.ttf",
                "/usr/share/fonts/truetype/thai/Garuda-Bold.ttf"]
        self.font_path = next((f for f in regular if os.path.exists(f)), None)
        self.bold_path = next((f for f in bold if os.path.exists(f)), self.font_path)
        emoji = ["C:/Windows/Fonts/seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc",
                 "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf"]
        self.emoji_path = next((f for f in emoji if os.path.exists(f)), None)
        self._font_cache = {}
        self.s = 1.0        # พิกเซลจริงต่อ 1 หน่วยเลย์เอาต์ (เลย์เอาต์ทั้งหมดออกแบบที่กว้าง 1280)

    def dims(self, img):
        """ขนาดภาพในหน่วยเลย์เอาต์ (h, w)"""
        return int(img.shape[0] / self.s), int(img.shape[1] / self.s)

    def p(self, *v):
        """แปลงพิกัด/รัศมี/ความหนาจากหน่วยเลย์เอาต์เป็นพิกเซลจริง"""
        r = tuple(round(x * self.s) for x in v)
        return r if len(r) > 1 else r[0]

    # ------------------------------------------------------------------
    #  พื้นฐาน
    # ------------------------------------------------------------------
    def get_font(self, size, bold=False):
        key = (size, bold)
        if key not in self._font_cache:
            path = self.bold_path if bold else self.font_path
            try:
                self._font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
            except Exception:
                self._font_cache[key] = ImageFont.load_default(size)
        return self._font_cache[key]

    def text_width(self, text, size, bold=False):
        l, _, r, _ = self.get_font(round(size * self.s), bold).getbbox(text, anchor="la")
        return (r - l) / self.s

    def fit(self, text, size, max_w, bold=False):
        """ตัดข้อความให้พอดีความกว้าง max_w พิกเซล (เติม ...)"""
        if self.text_width(text, size, bold) <= max_w:
            return text
        while text and self.text_width(text + "...", size, bold) > max_w:
            text = text[:-1]
        return text.rstrip() + "..."

    def text(self, img, text, pos, size=20, color=WHITE, bold=False, align="l", shadow=True):
        """วาดข้อความ (รองรับไทย) โดยแปลงเฉพาะพื้นที่ข้อความ  pos=(x, y บนสุดของบรรทัด)
        align: 'l' ซ้าย, 'c' กึ่งกลางที่ x, 'r' ชิดขวาที่ x  คืนค่าความกว้างข้อความ"""
        if not text:
            return 0
        font = self.get_font(round(size * self.s), bold)
        l, t, r, b = font.getbbox(text, anchor="la")
        w = r - l
        x, y = round(pos[0] * self.s), round(pos[1] * self.s)
        if align == "c":
            x -= w // 2
        elif align == "r":
            x -= w
        pad = 4
        ih, iw = img.shape[:2]
        x0, y0 = max(0, x + l - pad), max(0, y + t - pad)
        x1, y1 = min(iw, x + r + pad + 2), min(ih, y + b + pad + 2)
        if x1 <= x0 or y1 <= y0:
            return w / self.s
        roi = img[y0:y1, x0:x1]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        ox, oy = x - x0, y - y0
        if shadow:
            d.text((ox + 1, oy + 1), text, font=font, fill=(0, 0, 0), anchor="la")
        d.text((ox, oy), text, font=font, fill=(color[2], color[1], color[0]), anchor="la")
        img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return w / self.s

    # ชื่อเดิม (เผื่อสคริปต์อื่นเรียกใช้)
    def draw_thai_text(self, img, text, pos, font_size=22, color=WHITE, shadow=True, shadow_color=(0, 0, 0)):
        self.text(img, text, pos, size=font_size, color=color, shadow=shadow)
        return img

    def panel(self, img, x, y, w, h, bg=PANEL, alpha=0.82, border=PANEL_LINE, thickness=1):
        x, y, w, h = x * self.s, y * self.s, w * self.s, h * self.s
        thickness = self.p(thickness) if thickness > 0 else 0
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return
        sub = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1.0 - alpha, np.full_like(sub, bg), alpha, 0)
        if border is not None and thickness > 0:
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border, thickness, cv2.LINE_AA)

    def draw_panel(self, img, x, y, w, h, bg_color=PANEL, alpha=0.82, border_color=PANEL_LINE, border_thickness=1):
        self.panel(img, x, y, w, h, bg_color, alpha, border_color, border_thickness)
        return img

    def chip(self, img, text, x, y, color, size=16, filled=False):
        """ป้ายสถานะเล็ก ๆ คืนค่าพิกัด x ขอบขวา"""
        w = self.text_width(text, size, bold=True) + 20
        h = size + 14
        self.panel(img, x, y, w, h, bg=color if filled else PANEL, alpha=0.9 if filled else 0.8,
                   border=color, thickness=1)
        self.text(img, text, (x + 10, y + 5), size=size, bold=True,
                  color=(20, 20, 20) if filled else color, shadow=False)
        return x + w

    def keycap(self, img, key, label, x, y, size=15):
        """ปุ่มคีย์บอร์ด + คำอธิบาย คืนค่าพิกัด x ถัดไป"""
        kw = max(26, self.text_width(key, size, bold=True) + 14)
        kh = size + 12
        self.panel(img, x, y, kw, kh, bg=(60, 64, 74), alpha=0.95, border=(120, 126, 138))
        self.text(img, key, (x + kw // 2, y + 4), size=size, bold=True, align="c", shadow=False)
        lw = self.text(img, label, (x + kw + 7, y + 4), size=size, color=INK2, shadow=False)
        return x + kw + 7 + lw + 18

    # ------------------------------------------------------------------
    #  แถบบน: คะแนน + บอส + สถานะ
    # ------------------------------------------------------------------
    def draw_top_bar(self, img, round_no, wins, losses, draws, boss_name, auto_mode,
                     sound_on, countdown_sec, recent_results):
        _, w = self.dims(img)
        self.panel(img, 0, 0, w, 66, alpha=0.86, border=None)
        cv2.line(img, self.p(0, 66), self.p(w, 66), PANEL_LINE, self.p(1))

        # ซ้าย: ตาที่ + คะแนน
        self.text(img, t("top.round", n=round_no), (20, 8), size=15, color=MUTED)
        x = 20
        for label, val, col in (("top.wins", wins, GREEN), ("top.draws", draws, AMBER), ("top.losses", losses, RED)):
            x += self.text(img, f"{t(label)} ", (x, 28), size=19, color=INK2)
            x += self.text(img, str(val), (x, 26), size=22, bold=True, color=col) + 20
        if round_no:
            self.text(img, t("top.winrate", p=f"{wins / round_no * 100:.0f}"), (x, 30), size=16, color=MUTED)

        # กลาง: บอส
        cx = w // 2
        self.text(img, t("top.opponent"), (cx, 8), size=14, color=MUTED, align="c")
        self.text(img, short_boss_name(boss_name)[:34], (cx, 27), size=21, bold=True, color=ACCENT, align="c")

        # ขวา: สถานะ
        chips = [
            (t("top.sec", s=f"{countdown_sec:.1f}"), INK2, False),
            (t("top.sound") if sound_on else t("top.muted"), INK2 if sound_on else MUTED, False),
            (t("top.auto"), GREEN, True) if auto_mode else (t("top.auto_off"), MUTED, False),
        ]
        widths = [self.text_width(s, 15, True) + 20 for s, _, _ in chips]
        x = w - 20 - sum(widths) - 8 * (len(chips) - 1)
        for (s, c, f), cw in zip(chips, widths):
            self.chip(img, s, x, 18, c, size=15, filled=f)
            x += cw + 8

        # ผล 12 ตาล่าสุด (สี่เหลี่ยมเล็กใต้แถบ)
        if recent_results:
            x = 20
            x += self.text(img, t("top.recent"), (x, 74), size=14, color=INK2) + 8
            for r in recent_results[-12:]:
                col = {"win": GREEN, "lose": RED, "draw": AMBER}.get(r, MUTED)
                cv2.rectangle(img, self.p(x, 79), self.p(x + 13, 92), col, -1, cv2.LINE_AA)
                x += 18

    # ------------------------------------------------------------------
    #  การ์ดผู้เล่น / บอส (ล่างซ้าย / ล่างขวา)
    # ------------------------------------------------------------------
    def draw_player_card(self, img, gesture, fingers, hand_visible, player_name="ผู้เล่น", locked=False):
        h, _ = self.dims(img)
        cw, ch = 330, 128
        x, y = 20, h - ch - 62
        col = MOVE_COLOR.get(gesture, MUTED)
        self.panel(img, x, y, cw, ch, border=col if gesture in MOVE_COLOR else PANEL_LINE,
                   thickness=2 if gesture in MOVE_COLOR else 1)
        self.text(img, t("card.you", name=player_name), (x + 16, y + 10), size=15, color=INK2)

        # ไฟสถานะกล้องเห็นมือ
        dot = GREEN if hand_visible else RED
        cv2.circle(img, self.p(x + cw - 22, y + 20), self.p(6), dot, -1, cv2.LINE_AA)
        self.text(img, t("card.hand") if hand_visible else t("card.no_hand"), (x + cw - 34, y + 10), size=13,
                  color=dot, align="r")

        if gesture in MOVE_COLOR:
            self.text(img, t(gesture), (x + 16, y + 34), size=40, bold=True, color=col)
            if locked:
                self.chip(img, t("card.locked"), x + cw - 108, y + 48, GREEN, size=14, filled=True)
        else:
            self.text(img, t("card.no_move"), (x + 16, y + 44), size=26, color=MUTED)

        # นิ้วที่กาง (5 จุด)
        self.text(img, t("card.fingers"), (x + 16, y + 98), size=13, color=MUTED)
        for i in range(5):
            cxp = x + 96 + i * 22
            on = hand_visible and i < fingers
            cv2.circle(img, self.p(cxp, y + 107), self.p(7), col if on else (70, 74, 82), -1 if on else self.p(1), cv2.LINE_AA)

    # ------------------------------------------------------------------
    #  Emoji มือของบอส (Segoe UI Emoji สี) — วาดครั้งเดียวแล้วเก็บเป็น sprite
    # ------------------------------------------------------------------
    def _emoji_sprite(self, move, size):
        key = ("emoji", move, size)
        if key in self._font_cache:
            return self._font_cache[key]
        sprite = None
        if self.emoji_path:
            try:
                try:
                    fs = size
                    font = ImageFont.truetype(self.emoji_path, fs)
                except OSError:
                    fs = 109        # Noto Color Emoji (Linux) เป็นบิตแมปที่โหลดได้แค่ 109px แล้วค่อยย่อ
                    font = ImageFont.truetype(self.emoji_path, fs)
                ch = EMOJI.get(move, EMOJI["ค้อน"])
                canvas = Image.new("RGBA", (fs * 2, fs * 2), (0, 0, 0, 0))
                ImageDraw.Draw(canvas).text((fs // 4, fs // 4), ch, font=font, embedded_color=True)
                bbox = canvas.getbbox()
                if bbox:
                    rgba = np.asarray(canvas.crop(bbox))
                    sprite = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                    if fs != size:
                        sprite = cv2.resize(sprite, None, fx=size / fs, fy=size / fs, interpolation=cv2.INTER_AREA)
            except Exception:
                sprite = None
        self._font_cache[key] = sprite
        return sprite

    def blit_emoji(self, img, move, cx, cy, size=110, scale=1.0, alpha=1.0):
        """วาด emoji ท่ามือ ให้จุดกึ่งกลางอยู่ที่ (cx, cy) คืนค่า False ถ้าไม่มีฟอนต์ emoji"""
        cx, cy, size = cx * self.s, cy * self.s, round(size * self.s)
        sprite = self._emoji_sprite(move, size)
        if sprite is None:
            return False
        if abs(scale - 1.0) > 1e-3:
            sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        sh, sw = sprite.shape[:2]
        x0, y0 = int(cx - sw / 2), int(cy - sh / 2)
        ih, iw = img.shape[:2]
        ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(iw, x0 + sw), min(ih, y0 + sh)
        if ax1 <= ax0 or ay1 <= ay0:
            return True
        spr = sprite[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        a = (spr[:, :, 3:4].astype(np.float32) / 255.0) * alpha
        roi = img[ay0:ay1, ax0:ax1].astype(np.float32)
        img[ay0:ay1, ax0:ax1] = (spr[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
        return True

    def draw_boss_card(self, img, boss_move, boss_desc="", mode="idle", anim_t=0.0):
        """
        การ์ดบอส พร้อม emoji มือที่ "ออกท่าจริง"
        mode: 'idle'   = รอผู้เล่น (กำมือนิ่ง ๆ)
              'pump'   = กำลังนับถอยหลัง (กำมือเขย่าขึ้นลงตามจังหวะ 3-2-1)  anim_t = progress 0..1
              'reveal' = ออกท่าแล้ว (ขยายเด้งขึ้นมา)  anim_t = วินาทีตั้งแต่ออกท่า
        """
        h, w = self.dims(img)
        cw, ch = 330, 190
        x, y = w - cw - 20, h - ch - 62
        reveal = mode == "reveal" and boss_move in MOVE_COLOR
        col = MOVE_COLOR[boss_move] if reveal else ACCENT
        self.panel(img, x, y, cw, ch, border=col, thickness=2 if reveal else 1)
        self.text(img, t("card.bot"), (x + 16, y + 10), size=15, color=INK2)

        ex, ey = x + 92, y + 98
        if reveal:
            a = anim_t
            if a < 0.12:
                scale = 0.55 + 0.65 * a / 0.12
            elif a < 0.26:
                scale = 1.2 - 0.2 * (a - 0.12) / 0.14
            else:
                scale = 1.0
            drawn = self.blit_emoji(img, boss_move, ex, ey, 104, scale)
            self.text(img, t(boss_move), (x + 182, y + 70), size=34, bold=True, color=col)
            if not drawn:
                self.text(img, t("card.thrown"), (x + 16, y + 80), size=20, color=col)
        else:
            if mode == "pump":
                bob = -abs(np.sin(min(1.0, anim_t) * 3 * np.pi)) * 26     # เขย่ามือ 3 จังหวะ
                drawn = self.blit_emoji(img, "ค้อน", ex, ey + 10 + bob, 104, 0.92)
                label, lcol = t("card.pump"), WHITE
            else:
                drawn = self.blit_emoji(img, "ค้อน", ex, ey + 10, 104, 0.92, alpha=0.55)
                label, lcol = t("card.waiting"), MUTED
            self.text(img, label, (x + 172, y + 82), size=18, color=lcol)
            if not drawn:
                self.text(img, t("card.hidden"), (x + 16, y + 80), size=24, color=MUTED)
        self.text(img, self.fit(boss_desc, 12, cw - 32), (x + 16, y + ch - 28), size=12, color=MUTED)

    # ------------------------------------------------------------------
    #  ข้อความกลางจอตามสถานะ
    # ------------------------------------------------------------------
    def _center_box(self, img, w, h, border, dy=0):
        ih, iw = self.dims(img)
        x, y = (iw - w) // 2, (ih - h) // 2 - 30 + dy
        self.panel(img, x, y, w, h, alpha=0.88, border=border, thickness=2)
        return x, y

    def draw_idle_prompt(self, img):
        x, y = self._center_box(img, 520, 110, ACCENT)
        cx = x + 260
        self.text(img, t("idle.title"), (cx, y + 14), size=28, bold=True, color=WHITE, align="c")
        self.text(img, t("idle.hint"), (cx, y + 62), size=18, color=INK2, align="c")

    def draw_wait_gesture(self, img, carry_move, waited_sec, lock_progress=0.0, repeat_progress=0.0):
        """รอให้ผู้เล่นทำท่า — ยังไม่นับถอยหลังจนกว่าจะเห็นท่าชัด
        carry_move = ท่าที่ค้างมาจากตาที่แล้ว (ถ้ามี): เปลี่ยนท่า = ออกทันที, ค้างไว้ = ออกท่าเดิมซ้ำ"""
        x, y = self._center_box(img, 580, 140, ACCENT)
        cx = x + 290
        if carry_move:
            self.text(img, t("wait.change"), (cx, y + 12), size=28, bold=True, color=WHITE, align="c")
            self.text(img, t("wait.hold", m=t(carry_move)), (cx, y + 56), size=18,
                      color=INK2, align="c")
        else:
            self.text(img, t("wait.show"), (cx, y + 12), size=28, bold=True, color=WHITE, align="c")
            self.text(img, t("wait.moves"), (cx, y + 56), size=18,
                      color=INK2, align="c")
        self.text(img, t("wait.timer", s=f"{waited_sec:.1f}"), (cx, y + 92), size=14, color=MUTED, align="c")
        prog, col = (repeat_progress, AMBER) if repeat_progress > lock_progress else (lock_progress, GREEN)
        if prog > 0:
            bw = 420
            bx = cx - bw // 2
            cv2.rectangle(img, self.p(bx, y + 122), self.p(bx + bw, y + 127), (60, 64, 74), -1)
            cv2.rectangle(img, self.p(bx, y + 122), self.p(bx + int(bw * min(1.0, prog)), y + 127), col, -1)

    def draw_countdown(self, img, number, progress, locked_move=None, reaction_ms=None):
        """ตัวเลขนับถอยหลังพร้อมวงแหวนความคืบหน้า"""
        ih, iw = self.dims(img)
        cx, cy = iw // 2, ih // 2 - 40
        r = 86
        overlay = img.copy()
        cv2.circle(overlay, self.p(cx, cy), self.p(r + 16), PANEL, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
        cv2.circle(img, self.p(cx, cy), self.p(r), (60, 64, 74), self.p(6), cv2.LINE_AA)
        cv2.ellipse(img, self.p(cx, cy), self.p(r, r), -90, 0, int(360 * min(1.0, progress)), ACCENT, self.p(6), cv2.LINE_AA)
        if number == 0:
            self.text(img, t("cd.go"), (cx, cy - 34), size=52, bold=True, color=GREEN, align="c")
        else:
            self.text(img, str(number), (cx, cy - 58), size=96, bold=True, color=WHITE, align="c")
        if locked_move:
            self.panel(img, cx - 240, cy + r + 26, 480, 64, alpha=0.88, border=None)
            note = t("cd.locked") if number == 0 else t("cd.changeable")
            self.text(img, t("cd.your_move", m=t(locked_move), note=note), (cx, cy + r + 32),
                      size=17, color=WHITE, align="c")
            if reaction_ms is not None:
                self.text(img, t("reaction", s=f"{reaction_ms / 1000:.2f}"), (cx, cy + r + 60), size=14,
                          color=INK2, align="c")

    def draw_result_banner(self, img, result, player_move, boss_move, reaction_ms=None, auto_mode=False):
        title, col = RESULT_STYLE.get(result, ("-", WHITE))
        title = t(title)
        x, y = self._center_box(img, 560, 196, col)
        cx = x + 280
        self.text(img, title, (cx, y + 12), size=46, bold=True, color=col, align="c")
        # คุณ vs บอส
        self.text(img, t("res.you"), (cx - 140, y + 82), size=15, color=MUTED, align="c")
        self.text(img, t(player_move), (cx - 140, y + 100), size=30, bold=True,
                  color=MOVE_COLOR.get(player_move, WHITE), align="c")
        self.text(img, "VS", (cx, y + 104), size=22, bold=True, color=MUTED, align="c")
        self.text(img, t("res.bot"), (cx + 140, y + 82), size=15, color=MUTED, align="c")
        self.text(img, t(boss_move), (cx + 140, y + 100), size=30, bold=True,
                  color=MOVE_COLOR.get(boss_move, WHITE), align="c")
        foot = t("res.next_auto") if auto_mode else t("res.next_space")
        if reaction_ms is not None:
            foot = t("reaction", s=f"{reaction_ms / 1000:.2f}") + "   ·   " + foot
        self.text(img, foot, (cx, y + 160), size=15, color=INK2, align="c")

    def draw_message(self, img, text, color=AMBER):
        """ข้อความเตือนกลางจอแบบสั้น"""
        x, y = self._center_box(img, 520, 70, color)
        self.text(img, text, (x + 260, y + 18), size=22, bold=True, color=color, align="c")

    # ------------------------------------------------------------------
    #  แถบปุ่มด้านล่าง + คู่มือ + เมนูบอส + toast
    # ------------------------------------------------------------------
    def draw_footer_controls(self, img):
        h, w = self.dims(img)
        self.panel(img, 0, h - 46, w, 46, alpha=0.86, border=None)
        cv2.line(img, self.p(0, h - 46), self.p(w, h - 46), PANEL_LINE, self.p(1))
        x = 20
        for k, label in (("SPACE", "key.play"), ("A", "key.auto"), ("T", "key.speed"), ("B", "key.bots"),
                         ("4", "Lite Ensemble"), ("N", "key.new"), ("E", "key.lang"), ("H", "key.all"), ("Q", "key.quit")):
            x = self.keycap(img, k, t(label), x, h - 36)

    def draw_help_overlay(self, img):
        ih, iw = self.dims(img)
        mw, mh = 640, 547
        mx, my = (iw - mw) // 2, (ih - mh) // 2
        self.panel(img, mx, my, mw, mh, alpha=0.95, border=ACCENT, thickness=2)
        self.text(img, t("help.title"), (mx + 28, my + 18), size=26, bold=True, color=ACCENT)
        rows = [
            ("SPACE", "help.space"), ("A", "help.auto"), ("T", "help.speed"), ("B", "help.bots"),
            ("4", "help.combo"), ("S / L", "help.save"), ("N", "help.new"), ("R", "help.reset"),
            ("M", "help.mute"), ("G", "help.landmarks"), ("E", "help.lang"), ("Q / ESC", "help.quit"),
        ]
        y = my + 72
        for k, label in rows:
            self.keycap(img, k, t(label), mx + 32, y, size=16)
            y += 37
        self.text(img, t("help.close"), (mx + mw // 2, my + mh - 34), size=14, color=MUTED, align="c")

    def draw_boss_selection_menu(self, img, bosses, current_idx):
        ih, iw = self.dims(img)
        mw, mh = 700, 112 + 46 * len(bosses)
        mx, my = (iw - mw) // 2, max(70, (ih - mh) // 2)
        self.panel(img, mx, my, mw, mh, alpha=0.95, border=ACCENT, thickness=2)
        self.text(img, t("menu.title"), (mx + 28, my + 16), size=26, bold=True, color=ACCENT)
        self.text(img, t("menu.hint"), (mx + mw - 28, my + 24), size=15, color=MUTED, align="r")
        y = my + 66
        for i, b in enumerate(bosses):
            active = i == current_idx
            if active:
                self.panel(img, mx + 16, y - 4, mw - 32, 44, bg=(50, 44, 30), alpha=0.9, border=ACCENT)
            name = getattr(b, "name", t("menu.bot_n", n=i + 1))
            desc = bot_desc(b)
            self.keycap(img, str(i + 1), "", mx + 26, y + 4, size=15)
            self.text(img, self.fit(short_boss_name(name), 18, mw - 100, True), (mx + 70, y), size=18,
                      bold=True, color=ACCENT if active else WHITE)
            self.text(img, self.fit(desc, 13, mw - 100), (mx + 70, y + 22), size=13, color=MUTED)
            y += 46

    def draw_toast_notification(self, img, text, color=GREEN):
        ih, iw = self.dims(img)
        text = self.fit(text, 17, 540)
        tw = self.text_width(text, 17) + 40
        bx, by = (iw - tw) // 2, ih - 46 - 58
        self.panel(img, bx, by, tw, 40, alpha=0.92, border=color, thickness=2)
        self.text(img, text, (iw // 2, by + 8), size=17, color=color, align="c")

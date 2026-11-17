# Hallmark · redesign: full · scope: app · theme: Arcade (dealt 1/3, seed webcam-rock-paper-scissors-dce1·f8eccb14·2026-11-17·r0) · dials: ember/none
# Hallmark · signature: the showdown band, your move in ember VS the bot's in acid, opening and sliding in
# Hallmark · type: titles Prompt Bold + hard ember shadow (readable, both languages) · display Bungee / Prompt Black for short punch words · figures Bungee · body Prompt
# Hallmark · swap: no RGB split on titles (it smeared, worst on Thai); kept on the countdown digit only · figures in Bungee, Silkscreen's 2 reads as Z
# Hallmark · keys: one keycap everywhere (paper-3, ink-2 border, acid offset); emphasis by motion, never by a second style
# Hallmark · motion: panel entrance, digit pop, showdown open + slide, score pop, tile press-down, key press, idle attract, toast in/out · --reduce-motion renders final frames
# Hallmark · waive: sloplint, scaffold, log · the HUD is OpenCV/PIL, not HTML; the Floor is applied by hand (tokens below, no emoji icon, no dot-joined strings, no caps labels)
# =====================================================
#  ui_renderer.py — HUD บนภาพกล้อง (OpenCV + PIL)
#
#  หลักการ:
#   - กลางจอมีคำสั่งเดียวต่อสถานะ บอกว่า "ต้องทำอะไรต่อ"
#   - สีมีความหมายคงที่: ส้ม (ember) = คุณ, เขียวมะนาว (acid) = บอท
#   - เลย์เอาต์คิดในหน่วย 1280 กว้าง แล้วคูณ self.s ตอนวาดจริง (ขยายเต็มจอแล้วยังคม)
#   - แอนิเมชันอิงเวลา (self.clock) ไม่อิงจำนวนเฟรม, ปิดได้ด้วย self.motion = False
# =====================================================
import os
import re
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from i18n import t, bot_desc

# ---------- tokens (B, G, R) — Arcade, แปลงจาก oklch ----------
PAPER = (21, 4, 15)            # oklch(14% 0.04 312)
PAPER_2 = (32, 13, 25)         # oklch(18.5% 0.042 312)
PAPER_3 = (44, 22, 36)         # oklch(23% 0.045 312)
INK = (245, 236, 242)          # oklch(95% 0.014 315)
INK_2 = (221, 209, 217)        # oklch(87% 0.018 315)
MUTED = (172, 157, 167)        # oklch(71% 0.024 315)
RULE = (75, 48, 65)            # oklch(34% 0.05 312)
EMBER = (0, 82, 248)           # oklch(66% 0.215 40)   = คุณ
ACID = (27, 229, 213)          # oklch(88% 0.195 115)  = บอท
ACCENT_INK = (21, 4, 15)       # ตัวอักษรบนพื้น ember / acid
CRT_R = (52, 36, 249)          # oklch(63% 0.24 25)    เงา RGB ซ้าย
CRT_C = (218, 209, 0)          # oklch(78% 0.14 200)   เงา RGB ขวา
SHADE = (4, 1, 3)              # oklch(8% 0.02 315)    เงาตัวอักษร
ERROR = CRT_R

YOU, BOT = EMBER, ACID
RESULT_INK = {"win": YOU, "lose": BOT, "draw": INK_2}
RESULT_KEY = {"win": "res.win", "lose": "res.lose", "draw": "res.draw"}
MOVES = ("ค้อน", "กรรไกร", "กระดาษ")
MOVE_HINT = {"ค้อน": "move.hint.rock", "กรรไกร": "move.hint.scissors", "กระดาษ": "move.hint.paper"}
THAI = re.compile(r"[\u0E00-\u0E7F]")

FOOTER_H = 40
TOP_H = 56
PRESS_SEC = 0.15               # เวลาที่ปุ่มจมลงหลังกด


def short_boss_name(name: str) -> str:
    """ '6. Ensemble/Meta (เลือก...)' -> 'Ensemble/Meta' """
    s = re.sub(r"^\s*\d+\.\s*", "", str(name))
    if not s.startswith("Lite Ensemble") or "+" not in s:
        s = s.split(" (")[0].strip()
    return s or str(name)


class UIRenderer:
    def __init__(self):
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
        first = lambda paths: next((f for f in paths if os.path.exists(f)), None)
        body = first([os.path.join(here, "Prompt-Regular.ttf"), "C:/Windows/Fonts/LeelawUI.ttf",
                      "C:/Windows/Fonts/leelawad.ttf", "C:/Windows/Fonts/tahoma.ttf",
                      "/System/Library/Fonts/Supplemental/Thonburi.ttc"])
        bold = first([os.path.join(here, "Prompt-Bold.ttf"), "C:/Windows/Fonts/LeelaUIb.ttf",
                      "C:/Windows/Fonts/leelawdb.ttf", "C:/Windows/Fonts/tahomabd.ttf"]) or body
        black = first([os.path.join(here, "Prompt-Black.ttf")]) or bold
        self.faces = {
            "body": body,
            "bold": bold,
            "black": black,                                                       # display ภาษาไทย
            "bungee": first([os.path.join(here, "Bungee-Regular.ttf")]) or black,  # display ภาษาอังกฤษ + ตัวเลข
        }
        self.font_path, self.bold_path = body, bold
        self._font_cache = {}
        self.s = 1.0        # พิกเซลจริงต่อ 1 หน่วยเลย์เอาต์ (เลย์เอาต์ทั้งหมดออกแบบที่กว้าง 1280)
        self.clock = time.time
        self.motion = True  # False = ไม่มีแอนิเมชัน (--reduce-motion)
        self._anim = {}     # key -> (ค่าล่าสุด, เวลาที่ค่าเปลี่ยน)
        self._pressed = {}  # ชื่อปุ่ม -> เวลาที่กด

    # ------------------------------------------------------------------
    #  แอนิเมชัน
    # ------------------------------------------------------------------
    def _since(self, key, value):
        """วินาทีตั้งแต่ value ของ key นี้เปลี่ยนครั้งล่าสุด"""
        now = self.clock()
        old = self._anim.get(key)
        if old is None or old[0] != value:
            self._anim[key] = (value, now)
            return 0.0
        return now - old[1]

    def ease_out(self, t, dur):
        if not self.motion:
            return 1.0
        p = min(1.0, max(0.0, t / dur))
        return 1 - (1 - p) ** 3

    def ease_back(self, t, dur):
        """เกินเป้านิดหนึ่งแล้วเด้งกลับ (ใช้กับของที่ 'กระแทก' เข้ามา)"""
        if not self.motion:
            return 1.0
        p = min(1.0, max(0.0, t / dur)) - 1
        c = 1.70158
        return 1 + (c + 1) * p ** 3 + c * p ** 2

    def after(self, t, delay):
        return not self.motion or t >= delay

    @staticmethod
    def _fade(img, a, draw):
        """วาดแบบโปร่ง a (0..1) — คัดลอกเฟรมเฉพาะช่วงที่กำลังเฟด"""
        if a >= 0.999:
            draw(img)
        elif a > 0.001:
            tmp = img.copy()
            draw(tmp)
            cv2.addWeighted(tmp, a, img, 1 - a, 0, dst=img)

    def _enter(self, name):
        """แผงกลางจอเลื่อนขึ้น + เฟดเข้าเมื่อเปลี่ยนสถานะ คืน (ความทึบ, ระยะเลื่อน)"""
        e = self.ease_out(self._since("panel", name), 0.22)
        return e, 24 * (1 - e)

    def press(self, key):
        """เรียกเมื่อผู้เล่นกดปุ่ม: ปุ่มที่ตรงกันบนจอจะจมลงชั่วครู่"""
        self._pressed[key] = self.clock()

    def _is_down(self, key):
        now = self.clock()
        return self.motion and any(now - self._pressed.get(k, -9.0) < PRESS_SEC for k in key.split(" / "))

    # ------------------------------------------------------------------
    #  พื้นฐาน
    # ------------------------------------------------------------------
    def dims(self, img):
        """ขนาดภาพในหน่วยเลย์เอาต์ (h, w)"""
        return int(img.shape[0] / self.s), int(img.shape[1] / self.s)

    def p(self, *v):
        """แปลงพิกัด/รัศมี/ความหนาจากหน่วยเลย์เอาต์เป็นพิกเซลจริง"""
        r = tuple(round(x * self.s) for x in v)
        return r if len(r) > 1 else r[0]

    @staticmethod
    def _face(text, face):
        # Bungee ไม่มีอักษรไทย: ข้อความที่มีไทยใช้ Prompt Black แทน (เลือกทั้งข้อความ ไม่ผสมฟอนต์)
        if face == "display":
            return "black" if THAI.search(text) else "bungee"
        return "bungee" if face == "figure" else face

    def get_font(self, size, face="body"):
        key = (size, face)
        if key not in self._font_cache:
            path = self.faces.get(face)
            try:
                self._font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
            except Exception:
                self._font_cache[key] = ImageFont.load_default(size)
        return self._font_cache[key]

    def text_width(self, text, size, face="body"):
        l, _, r, _ = self.get_font(max(1, round(size * self.s)), self._face(text, face)).getbbox(text, anchor="la")
        return (r - l) / self.s

    def fit(self, text, size, max_w, face="body"):
        """ตัดข้อความให้พอดีความกว้าง max_w (เติม ...)"""
        if self.text_width(text, size, face) <= max_w:
            return text
        while text and self.text_width(text + "...", size, face) > max_w:
            text = text[:-1]
        return text.rstrip() + "..."

    def text(self, img, text, pos, size=20, color=INK, face="body", align="l", shadow=True, split=False,
             valign="t", hard=None):
        """วาดข้อความ pos=(x, y บนสุดของบรรทัด) หน่วยเลย์เอาต์  align: l / c / r
        hard  = สีเงาแข็งเยื้อง 3px (หัวข้อ)   split = เงา RGB แบบจอ CRT (เฉพาะตัวเลขนับถอยหลัง)
        valign="m" = y คือกึ่งกลางตัวอักษร   คืนค่าความกว้าง (หน่วยเลย์เอาต์)"""
        if not text:
            return 0
        font = self.get_font(max(1, round(size * self.s)), self._face(text, face))
        l, tp, r, b = font.getbbox(text, anchor="la")
        w = r - l
        x, y = round(pos[0] * self.s), round(pos[1] * self.s)
        if align == "c":
            x -= w // 2
        elif align == "r":
            x -= w
        if valign == "m":
            y -= (tp + b) // 2
        off = max(1, round(2 * self.s)) if split else 0
        hoff = max(1, round(3 * self.s)) if hard is not None else 0
        pad = 4 + off + hoff
        ih, iw = img.shape[:2]
        x0, y0 = max(0, x + l - pad), max(0, y + tp - pad)
        x1, y1 = min(iw, x + r + pad + 2), min(ih, y + b + pad + 2)
        if x1 <= x0 or y1 <= y0:
            return w / self.s
        roi = img[y0:y1, x0:x1]
        pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        ox, oy = x - x0, y - y0
        rgb = lambda c: (c[2], c[1], c[0])
        if hard is not None:
            d.text((ox + hoff, oy + hoff), text, font=font, fill=rgb(hard), anchor="la")
        elif shadow:
            sh = max(1, round(self.s))
            d.text((ox + sh, oy + sh), text, font=font, fill=rgb(SHADE), anchor="la")
        if split:
            d.text((ox - off, oy), text, font=font, fill=rgb(CRT_R), anchor="la")
            d.text((ox + off, oy), text, font=font, fill=rgb(CRT_C), anchor="la")
        d.text((ox, oy), text, font=font, fill=rgb(color), anchor="la")
        img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return w / self.s

    def title(self, img, text, pos, size=30, align="c"):
        """หัวข้อคำสั่ง: Prompt Bold + เงาแข็งสีส้ม อ่านง่ายทั้งไทยและอังกฤษ"""
        return self.text(img, text, pos, size=size, face="bold", color=INK, align=align, hard=EMBER)

    def rect(self, img, x, y, w, h, color):
        """สี่เหลี่ยมทึบ (หน่วยเลย์เอาต์)"""
        cv2.rectangle(img, self.p(x, y), self.p(x + w, y + h), color, -1)

    def panel(self, img, x, y, w, h, bg=PAPER_2, alpha=0.9, border=RULE, thickness=2, shadow=None, offset=6):
        """พื้นหลังโปร่ง + ขอบ + เงาแข็งไม่เบลอ (shadow = สีเงา)"""
        if shadow is not None and offset > 0:
            self.rect(img, x + offset, y + offset, w, h, shadow)
        x, y, w, h = x * self.s, y * self.s, w * self.s, h * self.s
        ih, iw = img.shape[:2]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(iw, int(x + w)), min(ih, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return
        sub = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1.0 - alpha, np.full_like(sub, bg), alpha, 0)
        if border is not None and thickness > 0:
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border, self.p(thickness))

    # ชื่อเดิม (เผื่อสคริปต์อื่นเรียกใช้)
    def draw_thai_text(self, img, text, pos, font_size=22, color=INK, shadow=True, shadow_color=SHADE):
        self.text(img, text, pos, size=font_size, color=color, shadow=shadow)
        return img

    def draw_panel(self, img, x, y, w, h, bg_color=PAPER_2, alpha=0.9, border_color=RULE, border_thickness=2):
        self.panel(img, x, y, w, h, bg_color, alpha, border_color, border_thickness)
        return img

    def tag(self, img, text, x, y, color, size=14, filled=False):
        """ป้ายสี่เหลี่ยม ขอบ 2px คืนค่าพิกัด x ขอบขวา"""
        w = self.text_width(text, size, "bold") + 18
        h = size + 14
        self.panel(img, x, y, w, h, bg=color if filled else PAPER_2, alpha=1.0 if filled else 0.9,
                   border=color, thickness=2)
        self.text(img, text, (x + 9, y + 5), size=size, face="bold",
                  color=ACCENT_INK if filled else color, shadow=False)
        return x + w

    def keycap_width(self, key, size):
        return max(size + 14, self.text_width(key, size, "bungee") + 16)

    def keycap(self, img, key, label, x, y, size=15, down=False):
        """ปุ่มคีย์บอร์ดแบบเดียวทั้งเกม: พื้นม่วง ขอบอ่อน เงาเขียวแข็ง
        เมื่อกด (หรือ down=True) ปุ่มจมลงไปทับเงาและเป็นสีส้มชั่วครู่  คืนค่าพิกัด x ถัดไป"""
        kw, kh, o = self.keycap_width(key, size), size + 14, 3
        down = down or self._is_down(key)
        self.rect(img, x + o, y + o, kw, kh, ACID)
        kx, ky = (x + o, y + o) if down else (x, y)
        self.panel(img, kx, ky, kw, kh, bg=EMBER if down else PAPER_3, alpha=1.0, border=INK_2, thickness=2)
        self.text(img, key, (kx + kw / 2, ky + 7), size=size, face="bungee", align="c", shadow=False,
                  color=ACCENT_INK if down else INK)
        lw = self.text(img, label, (x + kw + o + 9, y + 5), size=size, color=INK_2)
        return x + kw + o + 9 + lw + 20

    def well(self, img, cx, cy, w, h, bar=None):
        """แผงกลางจอ (cx, cy = จุดกึ่งกลาง) คืน (x, y) มุมซ้ายบน"""
        x, y = cx - w // 2, cy - h // 2
        self.panel(img, x, y, w, h, alpha=0.92)
        if bar is not None:
            self.rect(img, x, y, w, 4, bar)
        return x, y

    def _stage_center(self, img):
        h, w = self.dims(img)
        return w // 2, TOP_H + (h - FOOTER_H - 140 - TOP_H) // 2     # กึ่งกลางพื้นที่ระหว่างแถบบนกับการ์ด

    # ------------------------------------------------------------------
    #  แถบบน: สกอร์ + คู่ต่อสู้ + สถานะ
    # ------------------------------------------------------------------
    def draw_top_bar(self, img, round_no, wins, losses, draws, boss_name, auto_mode,
                     sound_on, countdown_sec, recent_results):
        _, w = self.dims(img)
        self.panel(img, 0, 0, w, TOP_H, alpha=0.92, border=None)
        self.rect(img, 0, TOP_H, w, 2, RULE)

        # ซ้าย: คุณ 2 | เสมอ 1 | 2 บอท  (ตัวเลขเด้งเมื่อคะแนนเปลี่ยน)
        pop = lambda key, v: 1 + 0.4 * (1 - self.ease_out(self._since(key, v), 0.3))
        x = 20
        x += self.text(img, t("res.you"), (x, 20), size=16, face="display", color=YOU) + 10
        sw = self.text_width(str(wins), 30, "figure")      # จองที่ตามขนาดปกติ ไม่ให้ทั้งแถวขยับตอนเด้ง
        self.text(img, str(wins), (x + sw / 2, TOP_H / 2), size=round(30 * pop("wins", wins)), face="figure",
                  color=YOU, align="c", valign="m")
        x += sw + 16
        self.rect(img, x, 14, 2, 30, RULE)
        x += 14
        x += self.text(img, f"{t('top.draws')} {draws}", (x, 18), size=16, face="bold", color=INK_2) + 14
        self.rect(img, x, 14, 2, 30, RULE)
        x += 16
        sl = self.text_width(str(losses), 30, "figure")
        self.text(img, str(losses), (x + sl / 2, TOP_H / 2), size=round(30 * pop("losses", losses)),
                  face="figure", color=BOT, align="c", valign="m")
        x += sl + 10
        x += self.text(img, t("res.bot"), (x, 20), size=16, face="display", color=BOT) + 18
        if round_no:
            self.text(img, t("top.winrate", p=f"{wins / round_no * 100:.0f}"), (x, 20), size=14, color=MUTED)

        # กลาง: คู่ต่อสู้
        cx = w // 2
        self.text(img, t("top.opponent"), (cx, 6), size=13, color=MUTED, align="c")
        self.text(img, short_boss_name(boss_name)[:34], (cx, 24), size=20, face="bold", color=BOT, align="c")

        # ขวา: ตาที่ + ป้ายสถานะ (เติมสีเมื่อเปิด)
        tags = [(t("top.sec", s=f"{countdown_sec:.1f}"), INK_2, False),
                (t("top.sound") if sound_on else t("top.muted"), ACID if sound_on else MUTED, sound_on),
                (t("top.auto") if auto_mode else t("top.auto_off"), ACID if auto_mode else MUTED, auto_mode)]
        widths = [self.text_width(s, 14, "bold") + 18 for s, _, _ in tags]
        x = w - 20 - sum(widths) - 8 * (len(tags) - 1)
        rw = self.text_width(t("top.round", n=round_no), 14)
        self.text(img, t("top.round", n=round_no), (x - 16 - rw, 18), size=14, color=MUTED)
        for (s, c, f), tw in zip(tags, widths):
            self.tag(img, s, x, 14, c, size=14, filled=f)
            x += tw + 8

        # ผลล่าสุด 12 ตา (ช่องใหม่ขยายเข้ามา)
        if recent_results:
            x = 20
            x += self.text(img, t("top.recent"), (x, TOP_H + 10), size=13, color=INK_2) + 10
            shown = recent_results[-12:]
            grow = self.ease_back(self._since("recent", len(recent_results)), 0.3)
            for i, r in enumerate(shown):
                k = grow if i == len(shown) - 1 else 1.0
                sz = 12 * max(0.0, k)
                self.rect(img, x + (12 - sz) / 2, TOP_H + 14 + (12 - sz) / 2, sz, sz,
                          {"win": YOU, "lose": BOT}.get(r, INK_2))
                x += 17

    # ------------------------------------------------------------------
    #  การ์ดผู้เล่น / บอท (ล่างซ้าย / ล่างขวา)
    # ------------------------------------------------------------------
    CARD_W, CARD_H = 320, 124

    def _card(self, img, left, ink):
        h, w = self.dims(img)
        x = 20 if left else w - self.CARD_W - 20
        y = h - FOOTER_H - 16 - self.CARD_H
        self.panel(img, x, y, self.CARD_W, self.CARD_H, alpha=0.92, border=None)
        self.rect(img, x, y, self.CARD_W, 4, ink)
        return x, y

    def draw_player_card(self, img, gesture, fingers, hand_visible, player_name="ผู้เล่น", locked=False):
        x, y = self._card(img, True, YOU)
        cw = self.CARD_W
        self.text(img, self.fit(t("card.you", name=player_name), 15, cw - 140, "bold"), (x + 16, y + 14),
                  size=15, face="bold", color=INK_2)
        # สถานะกล้องเห็นมือ เป็นป้ายข้อความ
        label, col = (t("card.hand"), INK_2) if hand_visible else (t("card.no_hand"), ERROR)
        self.tag(img, label, x + cw - 16 - self.text_width(label, 13, "bold") - 18, y + 12, col, size=13)

        if gesture in MOVES:
            k = 1 + 0.25 * (1 - self.ease_out(self._since("player_move", gesture), 0.18))
            self.text(img, t(gesture), (x + 16, y + 42 + 16 * (1 - k)), size=round(32 * k), face="display",
                      color=YOU)
            if locked:
                lw = self.text_width(t("card.locked"), 13, "bold") + 18
                self.tag(img, t("card.locked"), x + cw - 16 - lw, y + 50, YOU, size=13, filled=True)
        else:
            self._since("player_move", None)
            self.text(img, t("card.no_move"), (x + 16, y + 48), size=20, color=MUTED)

        # นิ้วที่กาง (5 ช่อง)
        lx = x + 16 + self.text(img, t("card.fingers"), (x + 16, y + 96), size=13, color=MUTED) + 12
        for i in range(5):
            on = hand_visible and i < fingers
            if on:
                self.rect(img, lx + i * 20, y + 99, 13, 13, YOU)
            else:
                cv2.rectangle(img, self.p(lx + i * 20, y + 99), self.p(lx + i * 20 + 13, y + 112), RULE, self.p(2))

    def draw_boss_card(self, img, boss_move, boss_desc="", mode="idle", anim_t=0.0):
        """
        การ์ดบอท
        mode: 'idle'   = รอผู้เล่น
              'pump'   = กำลังนับถอยหลัง (คำว่า เป่า ยิ้ง ฉุบ เด้งตามจังหวะ)  anim_t = progress 0..1
              'reveal' = ออกท่าแล้ว (คำท่ากระแทกเข้ามา)  anim_t = วินาทีตั้งแต่ออกท่า
        """
        x, y = self._card(img, False, BOT)
        cw = self.CARD_W
        self.text(img, t("card.bot"), (x + 16, y + 14), size=15, face="bold", color=INK_2)
        if mode == "reveal" and boss_move in MOVES:
            k = 0.6 + 0.4 * self.ease_back(anim_t, 0.3)
            self.text(img, t(boss_move), (x + 16, y + 42 + 16 * (1 - k)), size=max(1, round(32 * k)),
                      face="display", color=BOT)
        elif mode == "pump":
            bob = -abs(np.sin(min(1.0, anim_t) * 3 * np.pi)) * 10 if self.motion else 0     # เด้ง 3 จังหวะ
            self.text(img, t("card.pump"), (x + 16, y + 50 + bob), size=20, face="bold", color=INK)
        else:
            self.text(img, t("card.waiting"), (x + 16, y + 48), size=20, color=MUTED)
        self.text(img, self.fit(boss_desc, 12, cw - 32), (x + 16, y + 98), size=12, color=MUTED)

    # ------------------------------------------------------------------
    #  กลางจอ: คำสั่งเดียวต่อสถานะ (เลื่อนขึ้น + เฟดเข้าเมื่อเปลี่ยนสถานะ)
    # ------------------------------------------------------------------
    def draw_idle_prompt(self, img):
        a, dy = self._enter("idle")

        def draw(im):
            cx, cy = self._stage_center(im)
            x, y = self.well(im, cx, cy + dy, 560, 200, bar=YOU)
            self.title(im, t("idle.title"), (cx, y + 28), size=34)
            # SPACE กดตัวเองเป็นจังหวะ (โหมดเรียกผู้เล่นแบบตู้เกม) ส่วนหน้าตาปุ่มเหมือนปุ่มอื่นทุกอย่าง
            attract = self.motion and (self.clock() % 1.4) < PRESS_SEC
            for i, (key, label) in enumerate((("SPACE", t("idle.play")), ("A", t("idle.auto")))):
                total = self.keycap_width(key, 18) + 3 + 9 + self.text_width(label, 18)
                self.keycap(im, key, label, cx - total / 2, y + 96 + i * 50, size=18, down=attract and i == 0)

        self._fade(img, a, draw)

    def draw_wait_gesture(self, img, carry_move, waited_sec, lock_progress=0.0, repeat_progress=0.0,
                          active_move=None):
        """รอให้ผู้เล่นทำท่า — ยังไม่นับถอยหลังจนกว่าจะเห็นท่าชัด
        carry_move = ท่าที่ค้างมาจากตาที่แล้ว, active_move = ท่าที่กล้องกำลังจับ (ช่องนั้นจมลงตามความคืบหน้า)"""
        a, dy = self._enter("wait")
        prog, col = (repeat_progress, BOT) if repeat_progress > lock_progress else (lock_progress, YOU)

        def draw(im):
            cx, cy = self._stage_center(im)
            x, y = self.well(im, cx, cy + dy, 760, 236, bar=YOU)
            self.title(im, t("wait.change") if carry_move else t("wait.show"), (cx, y + 24), size=30)

            # ช่องท่า 3 ช่อง: ช่องที่กำลังจับ = ส้ม และจมลงไปทับเงาตามความคืบหน้าการล็อก
            tw, th, gap, o = 216, 70, 18, 4
            tx = cx - (3 * tw + 2 * gap) // 2
            ty = y + 80
            for i, m in enumerate(MOVES):
                bx = tx + i * (tw + gap)
                active = m == active_move
                if active:
                    sink = o * min(1.0, prog) if self.motion else 0
                    self.rect(im, bx + o, ty + o, tw, th, ACID)
                    bx, by = bx + sink, ty + sink
                    self.panel(im, bx, by, tw, th, bg=YOU, alpha=1.0, border=INK)
                else:
                    by = ty
                    self.panel(im, bx, by, tw, th, bg=PAPER_3, alpha=0.95, border=YOU if m == carry_move else RULE)
                self.text(im, t(m), (bx + tw / 2, by + 8), size=24, face="bold", align="c", shadow=not active,
                          color=ACCENT_INK if active else INK)
                self.text(im, t(MOVE_HINT[m]), (bx + tw / 2, by + 42), size=14, align="c", shadow=not active,
                          color=ACCENT_INK if active else MUTED)

            # แถบล็อก: ส้ม = ท่าใหม่กำลังล็อก, เขียว = ค้างท่าเดิมเพื่อออกซ้ำ
            bw = 3 * tw + 2 * gap
            self.rect(im, tx, ty + th + 16, bw, 6, RULE)
            if prog > 0:
                self.rect(im, tx, ty + th + 16, bw * min(1.0, prog), 6, col)
            foot = t("wait.timer", s=f"{waited_sec:.1f}")
            if carry_move:
                foot = t("wait.hold", m=t(carry_move)) + "    " + foot
            self.text(im, foot, (cx, ty + th + 34), size=14, color=INK_2, align="c")

        self._fade(img, a, draw)

    def draw_countdown(self, img, number, progress, locked_move=None, reaction_ms=None):
        """ตัวเลขนับถอยหลังในกรอบสี่เหลี่ยม เด้งทุกครั้งที่เลขเปลี่ยน แถบความคืบหน้าที่ขอบล่าง"""
        a, dy = self._enter("countdown")
        td = self._since("digit", number)

        def draw(im):
            cx, cy = self._stage_center(im)
            size = 220
            x, y = self.well(im, cx, cy - 20 + dy, size, size)
            if number == 0:
                k = 0.5 + 0.5 * self.ease_back(td, 0.3)
                self.text(im, t("cd.go"), (cx, y + size / 2), size=max(1, round(44 * k)), face="display", color=YOU,
                          align="c", valign="m", hard=SHADE)
            else:
                k = 1 + 0.35 * (1 - self.ease_out(td, 0.18))
                self.text(im, str(number), (cx, y + size / 2), size=round(130 * k), face="figure", color=INK,
                          align="c", split=True, valign="m")
            self.rect(im, x, y + size - 8, size * min(1.0, progress), 8, YOU)
            if locked_move:
                note = t("cd.locked") if number == 0 else t("cd.changeable")
                label = t("cd.your_move", m=t(locked_move), note=note)
                lw = self.text_width(label, 16, "bold") + 18
                self.tag(im, label, cx - lw / 2, y + size + 14, YOU, size=16, filled=True)
                if reaction_ms is not None:
                    self.text(im, t("reaction", s=f"{reaction_ms / 1000:.2f}"), (cx, y + size + 54), size=14,
                              color=INK_2, align="c")

        self._fade(img, a, draw)

    def draw_result_banner(self, img, result, player_move, boss_move, reaction_ms=None, auto_mode=False):
        """ลายเซ็นของเกม: แถบเต็มความกว้างเปิดออกจากกึ่งกลาง ผลกระแทกเข้ามา
        ท่าคุณ (ส้ม) เลื่อนมาจากซ้าย ท่าบอท (เขียว) เลื่อนมาจากขวา"""
        tr = self._since("panel", "result")
        _, w = self.dims(img)
        cx, cy = self._stage_center(img)
        ink = RESULT_INK.get(result, INK)
        bh = 250
        hh = bh * self.ease_out(tr, 0.2)                 # แถบเปิดจากเส้นกึ่งกลาง
        if hh < 2:
            return
        self.panel(img, 0, cy - hh / 2, w, hh, alpha=0.94, border=None)
        self.rect(img, 0, cy - hh / 2, w, 4, ink)
        y = cy - bh // 2
        if self.after(tr, 0.08):
            k = 1.5 - 0.5 * self.ease_back(tr - 0.08, 0.35)
            self.text(img, t(RESULT_KEY.get(result, "-")), (cx, y + 50), size=round(46 * k), face="display",
                      color=ink, align="c", valign="m", hard=SHADE)
        if self.after(tr, 0.1):
            slide = 220 * (1 - self.ease_out(tr - 0.1, 0.28))
            for side, label, move, col in ((-1, t("res.you"), player_move, YOU), (1, t("res.bot"), boss_move, BOT)):
                mx = cx + side * (260 + slide)
                self.text(img, label, (mx, y + 92), size=15, face="bold", color=MUTED, align="c")
                self.text(img, t(move), (mx, y + 114), size=54, face="display", color=col, align="c", hard=SHADE)
        if self.after(tr, 0.2):
            self.text(img, "VS", (cx, y + 132), size=26, face="bungee", color=MUTED, align="c")
        if self.after(tr, 0.3):
            foot = t("res.next_auto") if auto_mode else t("res.next_space")
            if reaction_ms is not None:
                foot = t("reaction", s=f"{reaction_ms / 1000:.2f}") + "    " + foot
            self.text(img, foot, (cx, y + bh - 36), size=15, color=INK_2, align="c")

    def draw_message(self, img, text, color=EMBER):
        """ข้อความเตือนกลางจอแบบสั้น"""
        cx, cy = self._stage_center(img)
        x, y = self.well(img, cx, cy, 520, 72, bar=color)
        self.text(img, text, (cx, y + 20), size=22, face="bold", color=color, align="c")

    # ------------------------------------------------------------------
    #  แถบปุ่มด้านล่าง + คู่มือ + เมนูบอท + toast
    # ------------------------------------------------------------------
    def draw_footer_controls(self, img):
        h, w = self.dims(img)
        self.panel(img, 0, h - FOOTER_H, w, FOOTER_H, alpha=0.92, border=None)
        self.rect(img, 0, h - FOOTER_H - 2, w, 2, RULE)
        x = 20
        for k, label in (("SPACE", "key.play"), ("A", "key.auto"), ("B", "key.bots"),
                         ("E", "key.lang"), ("H", "key.all"), ("Q", "key.quit")):
            x = self.keycap(img, k, t(label), x, h - FOOTER_H + 5, size=14)

    def draw_help_overlay(self, img):
        a, dy = self._enter("help")
        ih, iw = self.dims(img)
        rows = [("SPACE", "help.space"), ("A", "help.auto"), ("T", "help.speed"), ("B", "help.bots"),
                ("4", "help.combo"), ("S / L", "help.save"), ("N", "help.new"), ("R", "help.reset"),
                ("M", "help.mute"), ("G", "help.landmarks"), ("E", "help.lang"), ("Q / ESC", "help.quit")]
        rh = 40
        mw, mh = 660, 84 + rh * len(rows) + 40
        mx, my = (iw - mw) // 2, max(TOP_H + 8, (ih - mh) // 2)

        def draw(im):
            y0 = my + dy
            self.panel(im, mx, y0, mw, mh, alpha=0.96, border=RULE, shadow=ACID)
            self.title(im, t("help.title"), (mx + 28, y0 + 22), size=28, align="l")
            y = y0 + 76
            for k, label in rows:
                self.keycap(im, k, t(label), mx + 28, y + 4, size=15)
                self.rect(im, mx + 28, y + rh - 1, mw - 56, 1, RULE)
                y += rh
            self.text(im, t("help.close"), (mx + mw // 2, y0 + mh - 32), size=14, color=MUTED, align="c")

        self._fade(img, a, draw)

    def draw_boss_selection_menu(self, img, bosses, current_idx):
        a, dy = self._enter("menu")
        ih, iw = self.dims(img)
        rh = 56
        mw, mh = 720, 92 + rh * len(bosses) + 20
        mx, my = (iw - mw) // 2, max(TOP_H + 12, (ih - mh) // 2)

        def draw(im):
            y0 = my + dy
            self.panel(im, mx, y0, mw, mh, alpha=0.96, border=RULE, shadow=ACID)
            self.title(im, t("menu.title"), (mx + 28, y0 + 22), size=28, align="l")
            self.text(im, t("menu.hint"), (mx + mw - 28, y0 + 32), size=14, color=MUTED, align="r")
            y = y0 + 82
            for i, b in enumerate(bosses):
                active = i == current_idx
                if active:      # แถวที่เลือกอยู่ = แถวเคอร์เซอร์สีส้ม
                    self.panel(im, mx + 14, y, mw - 28, rh - 4, bg=YOU, alpha=1.0, border=None)
                else:
                    self.rect(im, mx + 28, y + rh - 3, mw - 56, 1, RULE)
                name = getattr(b, "name", t("menu.bot_n", n=i + 1))
                ink, sub = (ACCENT_INK, ACCENT_INK) if active else (INK, MUTED)
                self.keycap(im, str(i + 1), "", mx + 28, y + 9, size=15)
                self.text(im, self.fit(short_boss_name(name), 18, mw - 116, "bold"), (mx + 82, y + 4), size=18,
                          face="bold", color=ink, shadow=not active)
                self.text(im, self.fit(bot_desc(b), 13, mw - 116), (mx + 82, y + 30), size=13, color=sub,
                          shadow=not active)
                y += rh

        self._fade(img, a, draw)

    def draw_toast_notification(self, img, text, color=ACID, remaining=None):
        """toast เลื่อนขึ้น + เฟดเข้า, เฟดออกช่วง 0.3 วินาทีสุดท้าย (remaining = วินาทีที่เหลือ)"""
        tt = self._since("toast", text)
        e = self.ease_out(tt, 0.18)
        a = e
        if remaining is not None and self.motion:
            a = min(a, max(0.0, remaining / 0.3))
        ih, iw = self.dims(img)
        text = self.fit(text, 17, 560)
        tw = self.text_width(text, 17) + 40
        bx, by = (iw - tw) / 2, ih - FOOTER_H - 16 - 48 + 16 * (1 - e)

        def draw(im):
            self.panel(im, bx, by, tw, 42, alpha=0.96, border=color, shadow=ACID if color != ACID else EMBER,
                       offset=4)
            self.text(im, text, (iw / 2, by + 9), size=17, color=color, align="c")

        self._fade(img, a, draw)

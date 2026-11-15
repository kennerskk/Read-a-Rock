# =====================================================
#  sound_effects.py — เอฟเฟกต์เสียงผ่าน winsound (Non-blocking)
# =====================================================
import threading
import time

try:
    import winsound
except ImportError:
    winsound = None


class SoundManager:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def _play_async(self, func):
        if not self.enabled or winsound is None:
            return
        t = threading.Thread(target=func, daemon=True)
        t.start()

    def tick(self):
        """เสียงนับถอยหลัง 3, 2, 1"""
        def _action():
            try:
                winsound.Beep(900, 70)
            except Exception:
                pass
        self._play_async(_action)

    def shoot(self):
        """เสียงเมื่อถึงจังหวะ 'เป่ายิ้งฉุบ!'"""
        def _action():
            try:
                winsound.Beep(1400, 150)
            except Exception:
                pass
        self._play_async(_action)

    def win(self):
        """เสียงเมื่อผู้เล่นชนะ"""
        def _action():
            try:
                winsound.Beep(1046, 90)  # C6
                time.sleep(0.04)
                winsound.Beep(1318, 90)  # E6
                time.sleep(0.04)
                winsound.Beep(1567, 180) # G6
            except Exception:
                pass
        self._play_async(_action)

    def lose(self):
        """เสียงเมื่อผู้เล่นแพ้"""
        def _action():
            try:
                winsound.Beep(520, 120)
                time.sleep(0.05)
                winsound.Beep(380, 220)
            except Exception:
                pass
        self._play_async(_action)

    def draw(self):
        """เสียงเมื่อเสมอ"""
        def _action():
            try:
                winsound.Beep(700, 100)
                time.sleep(0.04)
                winsound.Beep(700, 100)
            except Exception:
                pass
        self._play_async(_action)

    def change_boss(self):
        """เสียงเปลี่ยนบอส"""
        def _action():
            try:
                winsound.Beep(1200, 70)
            except Exception:
                pass
        self._play_async(_action)

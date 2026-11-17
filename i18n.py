# =====================================================
#  i18n.py — ข้อความในเกม 2 ภาษา (ไทย / English)
#  ท่ามือในระบบยังเป็นภาษาไทยเสมอ (MOVES, telemetry) แปลเฉพาะตอนแสดงผล: t("ค้อน")
# =====================================================

LANG = "th"

STRINGS = {
    "th": {
        # ผล
        "res.win": "คุณชนะ!", "res.lose": "บอทชนะ", "res.draw": "เสมอ",
        # แถบบน
        "top.round": "ตาที่ {n}", "top.wins": "ชนะ", "top.draws": "เสมอ", "top.losses": "แพ้",
        "top.winrate": "คุณชนะ {p}%", "top.opponent": "คู่ต่อสู้", "top.sec": "{s} วิ",
        "top.sound": "เสียง", "top.muted": "ปิดเสียง", "top.auto": "AUTO", "top.auto_off": "AUTO ปิด",
        "top.recent": "ล่าสุด",
        # การ์ด
        "card.you": "คุณ  ·  {name}", "card.hand": "เห็นมือ", "card.no_hand": "ไม่เห็นมือ",
        "card.locked": "ล็อกแล้ว", "card.no_move": "ยังไม่เห็นท่า", "card.fingers": "นิ้วที่กาง",
        "card.bot": "บอท AI", "card.thrown": "ออกท่าแล้ว", "card.pump": "เป่า... ยิ้ง... ฉุบ!",
        "card.waiting": "รอคุณออกท่า", "card.hidden": "ซ่อนท่าอยู่...",
        # กลางจอ
        "idle.title": "พร้อมเล่นแล้ว", "idle.hint": "กด SPACE เล่น 1 ตา   ·   กด A เล่นต่อเนื่อง",
        "wait.change": "เปลี่ยนท่าได้เลย", "wait.hold": "หรือค้าง \"{m}\" ไว้เพื่อออกท่าเดิมซ้ำ",
        "wait.show": "ทำท่าให้กล้องเห็น",
        "wait.moves": "ค้อน (กำมือ)  ·  กรรไกร (2 นิ้ว)  ·  กระดาษ (แบมือ)",
        "wait.timer": "จับเวลาตอบสนอง {s} วิ",
        "cd.go": "ออก!", "cd.locked": "(ล็อกแล้ว)", "cd.changeable": "(เปลี่ยนได้จนกว่าจะนับจบ)",
        "cd.your_move": "ท่าของคุณ: {m}   {note}", "reaction": "เวลาตอบสนอง {s} วิ",
        "res.you": "คุณ", "res.bot": "บอท",
        "res.next_auto": "ตาถัดไปจะเริ่มเองเมื่อเห็นท่าใหม่", "res.next_space": "กด SPACE เล่นตาถัดไป",
        # ปุ่ม
        "key.play": "เล่น", "key.auto": "ออโต้", "key.speed": "ความเร็ว", "key.bots": "เลือกบอท",
        "key.new": "คนใหม่", "key.all": "ปุ่มทั้งหมด", "key.quit": "ออก", "key.lang": "EN",
        "help.title": "ปุ่มทั้งหมด",
        "help.space": "เริ่มตาใหม่ (แล้วทำท่าให้กล้องเห็น)",
        "help.auto": "โหมดออโต้: เริ่มตาใหม่เองเมื่อเห็นท่า",
        "help.speed": "ความเร็วนับถอยหลัง 3 / 2 / 1.5 / 1 วิ",
        "help.bots": "เปิดรายชื่อบอท  ·  1-4 เปลี่ยนบอท",
        "help.combo": "Lite Ensemble (กดซ้ำ: 2 / 3 / 4 โมเดล / Warm start)",
        "help.save": "บันทึก / โหลดสมองบอท",
        "help.new": "ผู้เล่นคนใหม่ (ล้างความจำบอท + แยกไฟล์ข้อมูล)",
        "help.reset": "รีเซ็ตคะแนนและความจำบอท",
        "help.mute": "เปิด/ปิดเสียง", "help.landmarks": "แสดง/ซ่อนเส้นโครงมือ",
        "help.lang": "เปลี่ยนภาษา ไทย / English", "help.quit": "ออกจากเกม",
        "help.close": "กด H อีกครั้งเพื่อปิด",
        "menu.title": "เลือกบอท", "menu.hint": "กดตัวเลขเพื่อเลือก  ·  B ปิด", "menu.bot_n": "บอท {n}",
        # toast / console (เกมกล้อง)
        "toast.loaded": "โหลดโมเดล: {name}", "toast.loaded_cycle": "โหลดโมเดล: {name}  (กด L ซ้ำเพื่อเปลี่ยน)",
        "log.loaded": "โหลดโมเดลคู่ปรับสำเร็จ: {name} ({path})", "log.load_fail": "ไม่สามารถโหลดโมเดล {path} ได้: {e}",
        "toast.switched": "เปลี่ยนบอท: {name}", "log.switched": ">>> เปลี่ยนเป็นบอท: {name}",
        "meta.custom_name": "บอทแก้ทาง {player} (ชนะ AI {p}%)",
        "toast.saved": "บันทึกโมเดลแล้ว: {f}", "toast.no_models": "ยังไม่มีโมเดลใน models/ — กด S เพื่อบันทึกก่อน",
        "toast.load_fail": "โหลดโมเดลไม่สำเร็จ: {e}", "toast.speed": "นับถอยหลัง {s} วินาที",
        "toast.reset": "รีเซ็ตคะแนนและความจำบอทแล้ว",
        "log.summary": "\n[สรุป {name}] {n} ตา | ผู้เล่นชนะ {w} | บอทชนะ {l} ({p}%) | เสมอ {d}",
        "toast.new_player": "ผู้เล่นใหม่: {name}  (บอทเริ่มเรียนรู้ใหม่)", "log.new_player": ">>> ผู้เล่นใหม่: {name} -> {path}",
        "log.no_camera": "[ERROR] ไม่สามารถเปิดกล้อง ID {id} ได้",
        "log.round": "[ตา {n}] คุณ: {p} | บอท: {b} -> {r} | reaction {rt}",
        "log.ready": "\nเกมพร้อมแล้ว — กด H ในหน้าต่างเกมเพื่อดูปุ่มทั้งหมด, Q เพื่อออก\n",
        "log.cam_read": "[ERROR] อ่านภาพจากกล้องไม่ได้", "log.thanks": "\nขอบคุณที่เล่นครับ!",
        "toast.auto_on": "ออโต้: เริ่มตาใหม่เองเมื่อเห็นท่า", "toast.auto_off": "ปิดออโต้",
        "toast.lang": "ภาษา: ไทย",
        "ask.champion": "พบโมเดลเฉพาะบุคคล '{path}' — โหลดมาเป็นคู่ต่อสู้ไหม? (Y/n): ",
        # telemetry
        "name.mistyped": "ชื่อ '{name}' มีอักษรไทยปนอังกฤษ — ลืมเปลี่ยนภาษาคีย์บอร์ดหรือเปล่า?",
        "name.reserved": "ชื่อ '{name}' สงวนไว้สำหรับข้อมูลจำลอง ใช้ชื่ออื่นแทน",
        "name.existing": "ผู้เล่นที่มีข้อมูลอยู่แล้ว: {names}", "name.ask": "ชื่อผู้เล่น (Enter = '{default}'): ",
        "name.retry": "  ! {msg} พิมพ์ใหม่อีกครั้ง",
        # fast mode
        "fm.loaded": "✅ โหลดโมเดลคู่ปรับสำเร็จ: {name} ({path})", "fm.load_fail": "⚠️ ไม่สามารถโหลดโมเดล {path} ได้: {e}",
        "fm.switched": "\n🔄 สลับไปเจอบอท: {name}", "fm.no_models": "\n⚠️ ยังไม่มีโมเดลที่บันทึกไว้ในโฟลเดอร์ models/",
        "fm.models": "\n📂 รายการโมเดลที่พบใน models/:", "fm.model_row": "  [{i}] {name}  (ไฟล์: {f})",
        "fm.pick_model": "เลือกหมายเลขโมเดลที่ต้องการโหลด (หรือกด Enter เพื่อยกเลิก): ",
        "fm.model_ok": "✅ โหลดโมเดลสำเร็จ: {f}", "fm.model_fail": "❌ โหลดโมเดลไม่สำเร็จ: {e}",
        "fm.name_model": "ตั้งชื่อโมเดล (กด Enter เพื่อใช้ชื่ออัตโนมัติ): ",
        "fm.custom_name": "บอทแก้ทาง {player} (Winrate {p}%)", "fm.saved": "💾 บันทึกโมเดลเรียบร้อยที่: {path}",
        "fm.cur_player": "\n👤 ผู้เล่นปัจจุบัน: {name}", "fm.new_name": "ใส่ชื่อผู้เล่นใหม่ (กด Enter เพื่อยกเลิก): ",
        "fm.bad_name": "⚠️ {msg} ยังไม่เปลี่ยนชื่อ",
        "fm.summary": "📋 สรุป {name}: {n} ตา | ชนะ {w} | บอทชนะ {l} ({p}%)",
        "fm.player_changed": "✅ เปลี่ยนผู้เล่นเป็น: {name} แล้ว! รีเซ็ตคะแนนและความจำบอท ตาต่อไปจะบันทึกในชื่อนี้",
        "fm.cur_pace": "\n⏱️ เวลาต่อตาปัจจุบัน: {s} วินาที",
        "fm.pace_help": "คำแนะนำ:\n  0   = รู้ผลทันที (Instant Mode — เหมาะสำหรับเก็บ Data หลายสิบตาไวๆ)\n"
                        "  0.5 = เร็วมาก (กำลังสนุก)\n  1.0 = ปานกลาง\n  2.0 = ปกติ\n  3.0 = โหมดช้า (เท่ากับหน้ากล้อง)",
        "fm.ask_pace": "ใส่เวลาต่อตาที่ต้องการ (วินาที): ", "fm.neg_pace": "❌ เวลาต้องไม่ติดลบ",
        "fm.pace_set": "✅ ปรับเวลาต่อตาเป็น {s} วินาทีแล้ว!", "fm.not_number": "❌ กรุณาใส่ตัวเลข",
        "fm.shoot": "  🥊 ยิ้งฉุบ!         ",
        "fm.win": "คุณชนะ! 🎉", "fm.lose": "บอทชนะ! 🤖", "fm.draw": "เสมอ! 🤝",
        "fm.round": "👉 [ตา {n}] คุณ: {p}  vs  บอท: {b}  ->  {tag}",
        "fm.stats": "   📊 ชนะ {w}/{n} ({wr}%) | บอทชนะ: {ai}% | เสมอ: {d}",
        "fm.title": "⚡ FAST ARENA — โหมดประลองด่วน (No-Cam + Custom Speed)",
        "fm.player": "👤 ผู้เล่น: {name}", "fm.cur_bot": "👾 บอทปัจจุบัน: {name}",
        "fm.pace": "⏱️ เวลาต่อตา: {s} วินาที",
        "fm.keys": "ปุ่มกดเลือกท่า:\n  [1] ค้อน ✊   [2] กรรไกร ✌️   [3] กระดาษ ✋\nปุ่มจัดการ:\n"
                   "  [u] เปลี่ยนชื่อผู้เล่น   [t] ปรับเวลาต่อตา    [b] สลับบอท (1-{n})\n"
                   "  [s] บันทึกโมเดล        [l] โหลดโมเดล        [m] เปิด/ปิดเสียง\n"
                   "  [r] รีเซ็ตคะแนน        [e] ภาษา ไทย/English  [q] สรุปผลและเลิกเล่น\n",
        "fm.prompt": "[{name} | ตา {n} | {s}s] ออกท่า (1/2/3): ",
        "fm.bot_list": "\nรายชื่อบอท:", "fm.pick_bot": "เลือกหมายเลขบอท (1-{n}): ",
        "fm.sound": "🔊 เสียงเอฟเฟกต์: {state}", "fm.on": "เปิด", "fm.off": "ปิด",
        "fm.reset": "🔄 รีเซ็ตคะแนนและสมองบอทเรียบร้อย!",
        "fm.bad_key": "⚠️ กด 1 (ค้อน), 2 (กรรไกร), 3 (กระดาษ) หรือกด 't' เพื่อปรับเวลา",
        "fm.end_title": "🏁 สรุปผลการเล่น Fast Arena", "fm.total": "เล่นทั้งหมด: {n} ตา",
        "fm.final": "คุณชนะ: {w} ({wr}%) | บอทชนะ: {l} ({ai}%) | เสมอ: {d}",
        "fm.telemetry": "📁 บันทึกข้อมูล Telemetry แล้วที่: {path}",
        "fm.ask_save": "\n💾 ต้องการเซฟโมเดลบอทตัวนี้ไว้สู้รอบหน้าไหม? (y/n): ",
        "fm.no_play": "ยังไม่มีการเล่น", "fm.bye": "ขอบคุณที่ร่วมสนุกครับ! 👋\n",
        "fm.lang": "🌐 ภาษา: ไทย",
        "fm.champion": "🤖 ตรวจพบโมเดลแชมเปียนเฉพาะบุคคล: '{path}'\n   ต้องการโหลดมาเป็นคู่ต่อสู้เลยไหม? (Y/n): ",
    },
    "en": {
        "ค้อน": "Rock", "กรรไกร": "Scissors", "กระดาษ": "Paper",
        "res.win": "You win!", "res.lose": "Bot wins", "res.draw": "Draw",
        "top.round": "Round {n}", "top.wins": "Win", "top.draws": "Draw", "top.losses": "Lose",
        "top.winrate": "You win {p}%", "top.opponent": "Opponent", "top.sec": "{s} s",
        "top.sound": "Sound", "top.muted": "Muted", "top.auto": "AUTO", "top.auto_off": "AUTO off",
        "top.recent": "Recent",
        "card.you": "You  ·  {name}", "card.hand": "Hand seen", "card.no_hand": "No hand",
        "card.locked": "Locked", "card.no_move": "No move yet", "card.fingers": "Fingers",
        "card.bot": "AI Bot", "card.thrown": "Thrown", "card.pump": "Rock... Paper... Shoot!",
        "card.waiting": "Waiting for you", "card.hidden": "Hiding its move...",
        "idle.title": "Ready to play", "idle.hint": "SPACE = play 1 round   ·   A = keep playing",
        "wait.change": "Change your move", "wait.hold": "or hold \"{m}\" to throw it again",
        "wait.show": "Show your hand to the camera",
        "wait.moves": "Rock (fist)  ·  Scissors (2 fingers)  ·  Paper (open hand)",
        "wait.timer": "Reaction timer {s} s",
        "cd.go": "Shoot!", "cd.locked": "(locked)", "cd.changeable": "(change until the count ends)",
        "cd.your_move": "Your move: {m}   {note}", "reaction": "Reaction time {s} s",
        "res.you": "You", "res.bot": "Bot",
        "res.next_auto": "Next round starts when a new move is seen", "res.next_space": "Press SPACE for the next round",
        "key.play": "Play", "key.auto": "Auto", "key.speed": "Speed", "key.bots": "Bots",
        "key.new": "New player", "key.all": "All keys", "key.quit": "Quit", "key.lang": "ไทย",
        "help.title": "All keys",
        "help.space": "Start a round (then show your move)",
        "help.auto": "Auto mode: start rounds when a move is seen",
        "help.speed": "Countdown speed 3 / 2 / 1.5 / 1 s",
        "help.bots": "Open bot list  ·  1-4 switch bot",
        "help.combo": "Lite Ensemble (press again: 2 / 3 / 4 models / Warm start)",
        "help.save": "Save / load bot brain",
        "help.new": "New player (clear bot memory + new data file)",
        "help.reset": "Reset score and bot memory",
        "help.mute": "Sound on/off", "help.landmarks": "Show/hide hand landmarks",
        "help.lang": "Switch language ไทย / English", "help.quit": "Quit game",
        "help.close": "Press H again to close",
        "menu.title": "Choose a bot", "menu.hint": "Press a number to pick  ·  B close", "menu.bot_n": "Bot {n}",
        "toast.loaded": "Loaded model: {name}", "toast.loaded_cycle": "Loaded model: {name}  (press L again to change)",
        "log.loaded": "Loaded opponent model: {name} ({path})", "log.load_fail": "Could not load model {path}: {e}",
        "toast.switched": "Bot: {name}", "log.switched": ">>> Switched to bot: {name}",
        "meta.custom_name": "Counter-bot for {player} (AI wins {p}%)",
        "toast.saved": "Model saved: {f}", "toast.no_models": "No models in models/ yet — press S to save one",
        "toast.load_fail": "Could not load model: {e}", "toast.speed": "Countdown {s} seconds",
        "toast.reset": "Score and bot memory reset",
        "log.summary": "\n[Summary {name}] {n} rounds | player won {w} | bot won {l} ({p}%) | draws {d}",
        "toast.new_player": "New player: {name}  (bot starts learning again)", "log.new_player": ">>> New player: {name} -> {path}",
        "log.no_camera": "[ERROR] Could not open camera ID {id}",
        "log.round": "[Round {n}] you: {p} | bot: {b} -> {r} | reaction {rt}",
        "log.ready": "\nGame ready — press H in the game window for all keys, Q to quit\n",
        "log.cam_read": "[ERROR] Could not read a frame from the camera", "log.thanks": "\nThanks for playing!",
        "toast.auto_on": "Auto: rounds start when a move is seen", "toast.auto_off": "Auto off",
        "toast.lang": "Language: English",
        "ask.champion": "Found personal model '{path}' — use it as your opponent? (Y/n): ",
        "name.mistyped": "Name '{name}' mixes Thai and English letters — wrong keyboard layout?",
        "name.reserved": "Name '{name}' is reserved for simulated data, pick another",
        "name.existing": "Players with data: {names}", "name.ask": "Player name (Enter = '{default}'): ",
        "name.retry": "  ! {msg} Try again.",
        "fm.loaded": "✅ Loaded opponent model: {name} ({path})", "fm.load_fail": "⚠️ Could not load model {path}: {e}",
        "fm.switched": "\n🔄 Now facing bot: {name}", "fm.no_models": "\n⚠️ No saved models in models/ yet",
        "fm.models": "\n📂 Models found in models/:", "fm.model_row": "  [{i}] {name}  (file: {f})",
        "fm.pick_model": "Model number to load (Enter to cancel): ",
        "fm.model_ok": "✅ Model loaded: {f}", "fm.model_fail": "❌ Could not load model: {e}",
        "fm.name_model": "Model name (Enter for an automatic name): ",
        "fm.custom_name": "Counter-bot for {player} (Winrate {p}%)", "fm.saved": "💾 Model saved to: {path}",
        "fm.cur_player": "\n👤 Current player: {name}", "fm.new_name": "New player name (Enter to cancel): ",
        "fm.bad_name": "⚠️ {msg} Name not changed.",
        "fm.summary": "📋 Summary {name}: {n} rounds | won {w} | bot won {l} ({p}%)",
        "fm.player_changed": "✅ Player is now {name}! Score and bot memory reset; next rounds are saved under this name",
        "fm.cur_pace": "\n⏱️ Current time per round: {s} seconds",
        "fm.pace_help": "Suggestions:\n  0   = instant result (good for collecting lots of data fast)\n"
                        "  0.5 = very fast\n  1.0 = medium\n  2.0 = normal\n  3.0 = slow (same as the camera game)",
        "fm.ask_pace": "Time per round (seconds): ", "fm.neg_pace": "❌ Time can't be negative",
        "fm.pace_set": "✅ Time per round set to {s} seconds!", "fm.not_number": "❌ Please enter a number",
        "fm.shoot": "  🥊 Shoot!         ",
        "fm.win": "You win! 🎉", "fm.lose": "Bot wins! 🤖", "fm.draw": "Draw! 🤝",
        "fm.round": "👉 [Round {n}] you: {p}  vs  bot: {b}  ->  {tag}",
        "fm.stats": "   📊 Won {w}/{n} ({wr}%) | bot wins: {ai}% | draws: {d}",
        "fm.title": "⚡ FAST ARENA — quick match (no camera + custom speed)",
        "fm.player": "👤 Player: {name}", "fm.cur_bot": "👾 Current bot: {name}",
        "fm.pace": "⏱️ Time per round: {s} seconds",
        "fm.keys": "Move keys:\n  [1] Rock ✊   [2] Scissors ✌️   [3] Paper ✋\nControls:\n"
                   "  [u] change player   [t] time per round   [b] switch bot (1-{n})\n"
                   "  [s] save model      [l] load model       [m] sound on/off\n"
                   "  [r] reset score     [e] ไทย/English      [q] summary and quit\n",
        "fm.prompt": "[{name} | round {n} | {s}s] your move (1/2/3): ",
        "fm.bot_list": "\nBots:", "fm.pick_bot": "Bot number (1-{n}): ",
        "fm.sound": "🔊 Sound effects: {state}", "fm.on": "on", "fm.off": "off",
        "fm.reset": "🔄 Score and bot brain reset!",
        "fm.bad_key": "⚠️ Press 1 (Rock), 2 (Scissors), 3 (Paper) or 't' to change the time",
        "fm.end_title": "🏁 Fast Arena summary", "fm.total": "Rounds played: {n}",
        "fm.final": "You won: {w} ({wr}%) | bot won: {l} ({ai}%) | draws: {d}",
        "fm.telemetry": "📁 Telemetry saved to: {path}",
        "fm.ask_save": "\n💾 Save this bot to fight next time? (y/n): ",
        "fm.no_play": "No rounds played", "fm.bye": "Thanks for playing! 👋\n",
        "fm.lang": "🌐 Language: English",
        "fm.champion": "🤖 Found personal champion model: '{path}'\n   Use it as your opponent? (Y/n): ",
        # คำอธิบายบอท (ภาษาไทยใช้ desc เดิมในคลาส)
        "desc.RandomBoss": "Plays 1/3 each at random — no learning, baseline",
        "desc.FrequencyBoss": "Counts your most-played move all game and counters it",
        "desc.DecayBoss": "Weights recent moves more, adapts when you change style",
        "desc.MarkovBoss": "Short sequences: what you usually play after your last move",
        "desc.NgramBoss": "Treats your last k moves as a state, catches longer patterns than Markov",
        "desc.EnsembleBoss": "Runs several models and trusts whichever is most accurate lately",
        "desc.QLearningBoss": "Learns from real win/loss rewards: state=your last move, action=bot move",
        "desc.MLPredictorBoss": "WSLS + Markov + N-gram features with real-time weight updates",
        "desc.PopulationPriorBoss": "Knows typical player habits from round one, from all real players",
        "desc.ComboEnsembleBoss": "N-gram + ML: won every benchmark combo (press again for 2 / 3 / 4 models / Warm start)",
        "desc.ComboEnsembleBoss.inst": "Uses only {n} models ({label})",
        "desc.ComboEnsembleBoss.warm": " — starts from typical habits, then adapts to you",
        "desc.CustomSavedBoss": "A bot brain loaded from a file",
    },
}


def set_lang(code):
    global LANG
    LANG = code if code in STRINGS else "th"


def toggle_lang():
    set_lang("en" if LANG == "th" else "th")
    return LANG


def t(key, **kw):
    """แปลข้อความ: ไม่มีในภาษาที่เลือก -> ใช้ภาษาไทย -> ใช้ key ตรง ๆ (ท่ามือภาษาไทยจึงผ่านได้เลย)"""
    s = STRINGS[LANG].get(key) or STRINGS["th"].get(key, key)
    return s.format(**kw) if kw else s


def bot_desc(bot):
    """คำอธิบายบอท (รับได้ทั้งคลาสและ instance)"""
    if LANG == "th":
        return getattr(bot, "desc", "")
    cls = bot if isinstance(bot, type) else type(bot)
    members = getattr(bot, "members", None) if not isinstance(bot, type) else None
    if members and cls.__name__ == "ComboEnsembleBoss":
        label = bot.name.split("(", 1)[-1].rstrip(")")
        warm = t("desc.ComboEnsembleBoss.warm") if "prior" in members else ""
        return t("desc.ComboEnsembleBoss.inst", n=len(members), label=label) + warm
    return STRINGS["en"].get("desc." + cls.__name__, getattr(bot, "desc", ""))


if __name__ == "__main__":
    missing = set(STRINGS["th"]) - set(STRINGS["en"])
    assert not missing, f"missing en keys: {missing}"
    set_lang("en")
    assert t("ค้อน") == "Rock" and t("top.round", n=3) == "Round 3"
    set_lang("th")
    assert t("ค้อน") == "ค้อน" and t("res.lose") == "บอทชนะ"
    print("i18n OK")

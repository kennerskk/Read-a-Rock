# =====================================================
#  telemetry.py — เก็บข้อมูลการเล่นแยกไฟล์ตามผู้เล่น
#
#  data/players/<ชื่อผู้เล่น>.csv   หนึ่งไฟล์ต่อหนึ่งคน (ชื่อไฟล์ = ชื่อตัวพิมพ์เล็ก)
#  data/players/guest.csv          ข้อมูลจำลองจาก simulator (ไม่ใช่คนจริง)
#
#  read_telemetry() อ่านรวมทุกไฟล์แล้วเรียงตามเวลา ได้ตารางเดียวกับไฟล์รวมแบบเดิม
#  สคริปต์วิเคราะห์ทุกตัวจึงใช้ได้เหมือนเดิม (ส่งพาธไฟล์ .csv เดี่ยวก็ยังได้)
# =====================================================
import os
import re
import csv
import glob
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ตั้ง RPS_PLAYERS_DIR เพื่อชี้ไปโฟลเดอร์อื่น (เช่นตอนทดสอบ จะได้ไม่ปนข้อมูลจริง)
PLAYERS_DIR = os.environ.get("RPS_PLAYERS_DIR") or os.path.join(BASE_DIR, "data", "players")
LEGACY_FILE = os.path.join(BASE_DIR, "data", "gameplay_telemetry.csv")
DEFAULT_SOURCE = "data/players"
COLUMNS = ["timestamp", "round_id", "player_name", "player_move", "ai_move", "result", "reaction_time_ms"]

_THAI = re.compile(r"[฀-๿]")
_LATIN = re.compile(r"[A-Za-z]")


def player_key(name: str) -> str:
    """ชื่อผู้เล่น -> ชื่อไฟล์ (ตัวพิมพ์เล็ก, ช่องว่างเป็น _, ตัดอักขระที่ใช้ในชื่อไฟล์ไม่ได้)"""
    key = str(name or "player").strip().lower()
    key = re.sub(r"\s+", "_", key)
    key = re.sub(r'[\\/:*?"<>|]', "", key)
    return key or "player"


def player_file(name: str) -> str:
    return os.path.join(PLAYERS_DIR, f"{player_key(name)}.csv")


def looks_mistyped(name: str) -> bool:
    """ชื่อที่มีทั้งอักษรไทยและอังกฤษปนกัน (เช่น 'ธThree' จากลืมเปลี่ยนภาษาคีย์บอร์ด)"""
    return bool(_THAI.search(name or "")) and bool(_LATIN.search(name or ""))


def known_players() -> List[str]:
    """ชื่อผู้เล่นจริงที่มีไฟล์อยู่แล้ว (ไม่รวม guest)"""
    names = []
    for path in sorted(glob.glob(os.path.join(PLAYERS_DIR, "*.csv"))):
        key = os.path.splitext(os.path.basename(path))[0]
        if key == "guest":
            continue
        display = key
        try:
            with open(path, encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
                if row and row.get("player_name"):
                    display = row["player_name"]
        except Exception:
            pass
        names.append(display)
    return names


RESERVED_KEYS = {"guest"}           # guest.csv = ข้อมูลจำลอง ห้ามคนจริงใช้ชื่อนี้


def is_reserved(name: str) -> bool:
    return player_key(name) in RESERVED_KEYS


def next_visitor_name() -> str:
    """ชื่ออัตโนมัติไม่ซ้ำสำหรับผู้เล่นที่ไม่ใส่ชื่อ เช่น visitor_1117_01 (เดือนวัน_ลำดับ)"""
    from datetime import datetime
    stamp = datetime.now().strftime("%m%d")
    n = 1
    while os.path.exists(player_file(f"visitor_{stamp}_{n:02d}")):
        n += 1
    return f"visitor_{stamp}_{n:02d}"


def validate_name(name: str) -> Optional[str]:
    """คืนข้อความเตือนถ้าชื่อใช้ไม่ได้ หรือ None ถ้าใช้ได้"""
    if looks_mistyped(name):
        return f"ชื่อ '{name}' มีอักษรไทยปนอังกฤษ — ลืมเปลี่ยนภาษาคีย์บอร์ดหรือเปล่า?"
    if is_reserved(name):
        return f"ชื่อ '{name}' สงวนไว้สำหรับข้อมูลจำลอง ใช้ชื่ออื่นแทน"
    return None


def ask_player_name(default: Optional[str] = None) -> str:
    """ถามชื่อผู้เล่นในเทอร์มินัล: โชว์ชื่อที่มีอยู่, เตือนชื่อผิด, กด Enter = ชื่อ visitor อัตโนมัติ"""
    existing = known_players()
    if existing:
        print("ผู้เล่นที่มีข้อมูลอยู่แล้ว: " + ", ".join(existing))
    default = default or next_visitor_name()
    while True:
        try:
            name = input(f"ชื่อผู้เล่น (Enter = '{default}'): ").strip() or default
        except (KeyboardInterrupt, EOFError):
            return default
        msg = validate_name(name)
        if msg:
            print(f"  ! {msg} พิมพ์ใหม่อีกครั้ง")
            continue
        return name


def canonical_name(name: str) -> str:
    """ใช้การสะกดตามที่บันทึกไว้แล้วในไฟล์ของผู้เล่นคนนั้น (กัน Ken / ken กลายเป็นคนละแบบ)"""
    path = player_file(name)
    try:
        with open(path, encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)
            if row and row.get("player_name"):
                return row["player_name"]
    except Exception:
        pass
    return str(name).strip()


def append_round(player_name: str, row: list) -> str:
    """เพิ่ม 1 แถว (ตาม COLUMNS) ลงไฟล์ของผู้เล่น คืนค่าพาธไฟล์"""
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    path = player_file(player_name)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow(row)
    return path


def read_telemetry(source: Optional[str] = None, include_synthetic: bool = True):
    """
    อ่านข้อมูลการเล่นเป็น DataFrame
      source = None / โฟลเดอร์ -> รวมทุกไฟล์ในโฟลเดอร์ (ค่าเริ่มต้น data/players) แล้วเรียงตามเวลา
      source = ไฟล์ .csv       -> อ่านไฟล์นั้นไฟล์เดียว
    รองรับพาธเก่า data/gameplay_telemetry.csv: ถ้าไฟล์นั้นไม่มีแล้ว จะอ่านจาก data/players แทน
    """
    import pandas as pd
    if source is None:
        source = PLAYERS_DIR
    if not os.path.isabs(source):
        source = os.path.join(BASE_DIR, source) if not os.path.exists(source) else source
    if os.path.isfile(source):
        return pd.read_csv(source)
    if not os.path.isdir(source):
        if os.path.normcase(os.path.abspath(source)) == os.path.normcase(LEGACY_FILE) and os.path.isdir(PLAYERS_DIR):
            source = PLAYERS_DIR
        else:
            return pd.DataFrame(columns=COLUMNS)
    frames = []
    for path in sorted(glob.glob(os.path.join(source, "*.csv"))):
        if not include_synthetic and os.path.basename(path).lower() == "guest.csv":
            continue
        d = pd.read_csv(path)
        d["_order"] = range(len(d))
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["_ts", "_order"], kind="stable").drop(columns=["_ts", "_order"]).reset_index(drop=True)
    return df


def split_legacy_file(src: str = LEGACY_FILE, rename: Optional[dict] = None) -> dict:
    """
    แยกไฟล์รวมแบบเดิมเป็นไฟล์ต่อผู้เล่น (ใช้ครั้งเดียวตอนย้ายข้อมูล)
    rename: แก้ชื่อผู้เล่น เช่น {"ธThree": "Three"}
    ชื่อที่ต่างกันแค่ตัวพิมพ์เล็กใหญ่ (Ken/ken) รวมเป็นคนเดียว ใช้การสะกดที่พบบ่อยที่สุด
    """
    import pandas as pd
    df = pd.read_csv(src)
    if rename:
        df["player_name"] = df["player_name"].replace(rename)
    df["_key"] = df["player_name"].map(player_key)
    out = {}
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    for key, g in df.groupby("_key", sort=True):
        canon = g["player_name"].value_counts().idxmax()
        g = g.assign(player_name=canon)[COLUMNS]
        g["reaction_time_ms"] = pd.to_numeric(g["reaction_time_ms"], errors="coerce").round().astype("Int64")
        path = os.path.join(PLAYERS_DIR, f"{key}.csv")
        if os.path.exists(path):
            raise FileExistsError(f"มีไฟล์ {path} อยู่แล้ว — ไม่เขียนทับ")
        g.to_csv(path, index=False, encoding="utf-8")
        out[canon] = (path, len(g))
    return out

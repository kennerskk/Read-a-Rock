# =====================================================
#  model_io.py — ระบบ Import / Export โมเดล AI บอส
#  รองรับการบันทึกสถานะสมอง, สถิติการเรียนรู้, และโหลดกลับมาสู้ใหม่
# =====================================================
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List


MODELS_DIR = "models"


def ensure_models_dir(directory: str = MODELS_DIR) -> str:
    """ตรวจสอบและสร้างโฟลเดอร์เก็บโมเดลหากยังไม่มี"""
    os.makedirs(directory, exist_ok=True)
    return directory


def save_boss_model(boss, filepath: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    ส่งออก (Export) โมเดล AI บอสลงไฟล์ JSON พร้อม Metadata ละเอียด
    คืนค่า path ของไฟล์ที่ถูกบันทึก
    """
    ensure_models_dir(MODELS_DIR)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_boss_type = boss.__class__.__name__

    if filepath is None:
        filename = f"{safe_boss_type}_{timestamp_str}.json"
        filepath = os.path.join(MODELS_DIR, filename)

    # ดึงสถานะภายในของโมเดล
    if hasattr(boss, "get_state"):
        state_data = boss.get_state()
    else:
        # Fallback กรณีไม่มี get_state
        state_data = getattr(boss, "__dict__", {})

    meta = {
        "model_type": safe_boss_type,
        "boss_name": getattr(boss, "name", safe_boss_type),
        "boss_desc": getattr(boss, "desc", ""),
        "exported_at": datetime.now().isoformat(),
        "timestamp_epoch": time.time(),
        "version": "1.0.0",
        "author": "Antigravity Data Science Pipeline"
    }
    if metadata:
        meta.update(metadata)

    payload = {
        "metadata": meta,
        "state": state_data
    }

    # บันทึกเป็น JSON แบบ Human-readable
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath


def load_boss_model(filepath: str, boss_registry: Optional[Dict[str, Any]] = None):
    """
    นำเข้า (Import) โมเดล AI บอสจากไฟล์ JSON
    คืนค่า instance ของ Boss พร้อมฟื้นฟูสมองที่เคยเรียนรู้ไว้
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload.get("metadata", {})
    state = payload.get("state", {})
    model_type = meta.get("model_type", "Boss")

    # นำเข้า adaptive_arena แบบ dynamic เพื่อป้องกัน Circular Import
    import adaptive_arena as arena

    registry = {
        "RandomBoss": arena.RandomBoss,
        "FrequencyBoss": arena.FrequencyBoss,
        "DecayBoss": arena.DecayBoss,
        "MarkovBoss": arena.MarkovBoss,
        "NgramBoss": arena.NgramBoss,
        "EnsembleBoss": arena.EnsembleBoss,
        "QLearningBoss": arena.QLearningBoss,
    }
    if hasattr(arena, "MLPredictorBoss"):
        registry["MLPredictorBoss"] = arena.MLPredictorBoss
    if hasattr(arena, "CustomSavedBoss"):
        registry["CustomSavedBoss"] = arena.CustomSavedBoss

    if boss_registry:
        registry.update(boss_registry)

    boss_class = registry.get(model_type)
    if boss_class is None:
        # ถ้าไม่มีคลาสเฉพาะ ให้ใช้ CustomSavedBoss
        if hasattr(arena, "CustomSavedBoss"):
            boss_class = arena.CustomSavedBoss
        else:
            raise ValueError(f"ไม่พบคลาสโมเดลสำหรับประเภท '{model_type}' ในระบบ")

    # สร้างอินสแตนซ์โมเดล
    try:
        boss = boss_class()
    except TypeError:
        boss = boss_class(filepath=filepath)

    # โหลดสถานะกลับคืน
    if hasattr(boss, "load_state"):
        boss.load_state(state)
    elif hasattr(boss, "Q") and "Q" in state:
        boss.Q.update(state["Q"])

    # แนบ metadata กลับไว้กับตัวโมเดล
    boss.loaded_metadata = meta
    boss.model_filepath = filepath
    if "custom_name" in meta:
        boss.name = meta["custom_name"]
    elif "boss_name" in meta:
        boss.name = f"[LOADED] {meta['boss_name']}"

    return boss


def list_saved_models(directory: str = MODELS_DIR) -> List[Dict[str, Any]]:
    """
    ค้นหารายการโมเดลทั้งหมดที่มีอยู่ในโฟลเดอร์
    คืนค่าเป็น list ของ metadata เรียงลำดับจากใหม่สุดไปเก่าสุด
    """
    if not os.path.exists(directory):
        return []

    models = []
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                meta["filename"] = fname
                meta["filepath"] = fpath
                models.append(meta)
            except Exception:
                continue

    # เรียงตามเวลาบันทึกล่าสุด
    models.sort(key=lambda m: m.get("timestamp_epoch", 0), reverse=True)
    return models


def get_latest_saved_model(directory: str = MODELS_DIR) -> Optional[str]:
    """ดึง path ของโมเดลล่าสุดที่ถูกบันทึกไว้"""
    models = list_saved_models(directory)
    if models:
        return models[0]["filepath"]
    return None

import os
import json
import joblib
import numpy as np

# ------- FastAPI -------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------- Firebase (optional สำหรับอ่าน preference จาก Firestore) -------
import firebase_admin
from firebase_admin import credentials, firestore

# =========================
# 1) โหลดเมตา + โมเดล
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "model_data.json"), "r", encoding="utf-8") as f:
    MD = json.load(f)

GENRES = MD["genres"]                    # ลำดับแนวทั้งหมด (16 แนว)
FREQ_MAP = MD["freq_map"]                # {"Never":0,"Rarely":1,"Sometimes":2,"Very frequently":3}
LABEL_NAMES = MD["label_names"]          # index -> genre name
NAME2IDX = MD["name_to_index"]           # genre name -> index

# Logistic Regression (Model-1: favorite-genre classification)
LOGREG = joblib.load(os.path.join(HERE, "logreg_model.joblib"))

# KNN CF: fit จาก X_train (เวกเตอร์พฤติกรรมฟัง) + y_train (label เป็น index แนว)
X_TRAIN = np.load(os.path.join(HERE, "X_train.npy"))
Y_TRAIN = np.load(os.path.join(HERE, "y_train.npy"))

from sklearn.neighbors import NearestNeighbors
KNN = NearestNeighbors(n_neighbors=25, metric="cosine")
KNN.fit(X_TRAIN)

# =========================
# 2) Utils
# =========================
def _coerce_freq(v):
    """รับทั้งข้อความ Likert หรือเลข แล้วแปลงเป็น 0..3"""
    if isinstance(v, (int, float)):
        x = int(v)
        return max(0, min(3, x))
    if isinstance(v, str):
        v = v.strip()
        if v in FREQ_MAP:
            return FREQ_MAP[v]
        # เผื่อส่ง "Never (0)" จาก UI เข้ามา
        for k in FREQ_MAP:
            if v.startswith(k):
                return FREQ_MAP[k]
        # เผื่อส่ง "0","1","2","3" แบบ string
        try:
            return _coerce_freq(int(v))
        except Exception:
            return 0
    return 0

def preprocess_input(user_pref_dict):
    """
    รับ dict: {genre_name: Likert/number} -> เวกเตอร์ความยาวเท่าจำนวน GENRES
    """
    x = np.zeros(len(GENRES), dtype=float)
    for g in GENRES:
        if g in user_pref_dict:
            x[NAME2IDX[g]] = _coerce_freq(user_pref_dict[g])
    nonzero = int((x > 0).sum())
    return x, nonzero

def normalize_minmax(vec):
    v = np.asarray(vec, dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo + 1e-8)

def knn_scores(x, k=25):
    """
    ให้คะแนนทุกแนวด้วยการโหวตจากเพื่อนบ้าน (CF แบบง่าย)
    คืนเวกเตอร์ความยาว = n_genres
    """
    # หาเพื่อนบ้าน
    distances, indices = KNN.kneighbors([x], n_neighbors=min(k, len(X_TRAIN)))
    idxs = indices[0]

    # นับ label ของเพื่อนบ้านเป็น score
    scores = np.zeros(len(GENRES), dtype=float)
    for i in idxs:
        c = int(Y_TRAIN[i])           # class index (0..15)
        scores[c] += 1.0
    # ทำให้เป็นเวกเตอร์เต็มตาม GENRES
    return scores

def recommend_with_strategy(user_pref_dict, strategy="auto", topk=3, return_debug=False):
    """
    กลยุทธ์เลือกโมเดล:
      - auto: ถ้าผู้ใช้กรอก >= threshold (max(3, len(GENRES)//4)) -> logreg, น้อยกว่านั้น -> knn
      - self/logreg/knn: บังคับใช้โมเดลนั้น ๆ
    """
    topk = max(1, min(topk, len(GENRES)))
    x, nonzero = preprocess_input(user_pref_dict)

    # ----- ตัดสินใจเลือกโมเดล -----
    threshold = max(3, len(GENRES) // 4)  # 16 แนว -> 4
    chosen = strategy
    if strategy == "auto":
        chosen = "logreg" if nonzero >= threshold else "knn"

    # ----- คำนวณคะแนนทุกแนว (เวกเตอร์เต็ม) -----
    if chosen == "self":
        full = x.astype(float)
        norm = normalize_minmax(full)

    elif chosen == "logreg":
        if nonzero == 0:
            # fallback -> self
            full = x.astype(float)
            norm = normalize_minmax(full)
            chosen = "self"
        else:
            proba = LOGREG.predict_proba([x])[0]  # ความยาว = n_classes (=16)
            # map class index -> ตำแหน่งใน GENRES
            # LOGREG.classes_ เป็น index 0..15 ตรงกับ LABEL_NAMES
            class_index_to_pos = {int(i): i for i in range(len(GENRES))}
            full = np.array([proba[class_index_to_pos[i]] for i in range(len(GENRES))], dtype=float)
            # prob อยู่แล้ว -> clip 0..1
            norm = np.clip(full, 0.0, 1.0)

    elif chosen == "knn":
        if nonzero == 0:
            # เลี่ยงเวกเตอร์ศูนย์: ใส่ค่าเล็ก ๆ ให้เท่ากันทุกแนว
            x = np.ones_like(x) * 1e-3
        full = knn_scores(x, k=25)
        norm = normalize_minmax(full)

    else:
        raise ValueError("Unknown strategy")

    order = np.argsort(-norm)[:topk]
    items = [GENRES[i] for i in order]   # คืนเป็นชื่อแนว (ฝั่งเว็บนำไปค้นเพลงเอง)

    if return_debug:
        dbg = {"chosen_strategy": chosen, "nonzero": nonzero, "threshold": threshold}
        return {"items": items, "used": dbg}
    return {"items": items, "used": {"chosen_strategy": chosen}}

# =========================
# 3) FastAPI
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Firebase init (optional – ถ้ามี GOOGLE_APPLICATION_CREDENTIALS)
if not firebase_admin._apps:
    try:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        else:
            firebase_admin.initialize_app()  # ใช้ default credentials ได้ในบางสภาพแวดล้อม
    except Exception:
        # ถ้า init ไม่ได้ เราจะยังใช้ endpoint แบบส่ง preference ตรง ๆ ได้
        pass

# ----- Schemas -----
class RecommendByUserId(BaseModel):
    userId: str
    topk: int = 3
    strategy: str = "auto"  # "auto" | "self" | "logreg" | "knn"

class RecommendByPrefs(BaseModel):
    preferences: dict   # {"EDM":3, "Hip hop":"Sometimes", ...}
    topk: int = 3
    strategy: str = "auto"

# ----- Endpoints -----
@app.get("/")
def root():
    return {"status": "ok", "n_genres": len(GENRES)}

@app.post("/get_recommendations")
def get_recommendations(payload: RecommendByUserId):
    """
    เวอร์ชันใช้งานจริง: อ่าน preference จาก Firestore ที่ users/{uid}.preference
    """
    try:
        if not firebase_admin._apps:
            raise HTTPException(status_code=500, detail="Firebase not initialized")

        db = firestore.client()
        doc = db.collection("users").document(payload.userId).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        data = doc.to_dict() or {}
        prefs = data.get("preference") or {}
        if not isinstance(prefs, dict):
            raise HTTPException(status_code=400, detail="Invalid preference format")

        res = recommend_with_strategy(prefs, strategy=payload.strategy, topk=payload.topk, return_debug=True)
        return {"recommendations": res["items"], "used": res["used"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

@app.post("/get_recommendations_from_prefs")
def get_recommendations_from_prefs(payload: RecommendByPrefs):
    """
    เวอร์ชันทดลอง/ทดสอบ: ส่ง preferences มาโดยตรง (ไม่ต้องพึ่ง Firestore)
    """
    try:
        res = recommend_with_strategy(payload.preferences, strategy=payload.strategy, topk=payload.topk, return_debug=True)
        return {"recommendations": res["items"], "used": res["used"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}")

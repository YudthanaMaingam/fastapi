import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================
# 🔥 Firebase Initialization
# =========================
import firebase_admin
from firebase_admin import credentials, firestore

try:
    cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin initialized successfully.")
    db = firestore.client()
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {e}")
    db = None

# =========================
# 🧠 Load Model and Metadata
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model_data.json"), "r", encoding="utf-8") as f:
    MD = json.load(f)

GENRES = MD["genres"]
FREQ_MAP = MD["freq_map"]
LABEL_NAMES = MD["label_names"]
NAME2IDX = MD["name_to_index"]

LOGREG = joblib.load(os.path.join(BASE_DIR, "logreg_model.joblib"))
X_TRAIN = np.load(os.path.join(BASE_DIR, "X_train.npy"))
Y_TRAIN = np.load(os.path.join(BASE_DIR, "y_train.npy"))

from sklearn.neighbors import NearestNeighbors
KNN = NearestNeighbors(n_neighbors=25, metric="cosine")
KNN.fit(X_TRAIN)

# =========================
# 🧩 Helper Functions
# =========================
def _coerce_freq(v):
    """แปลงค่าจากข้อความ Likert หรือเลข ให้เป็น 0..3"""
    if isinstance(v, (int, float)):
        return max(0, min(3, int(v)))
    if isinstance(v, str):
        v = v.strip()
        if v in FREQ_MAP:
            return FREQ_MAP[v]
        for k in FREQ_MAP:
            if v.startswith(k):
                return FREQ_MAP[k]
        try:
            return int(v)
        except Exception:
            return 0
    return 0

def preprocess_input(user_pref_dict):
    """รับ dict {genre: Likert/number} → คืนเวกเตอร์ความชอบของผู้ใช้"""
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
    """คำนวณคะแนนแนวเพลงจากเพื่อนบ้าน"""
    distances, indices = KNN.kneighbors([x], n_neighbors=min(k, len(X_TRAIN)))
    idxs = indices[0]
    scores = np.zeros(len(GENRES), dtype=float)
    for i in idxs:
        c = int(Y_TRAIN[i])
        scores[c] += 1.0
    return scores

def recommend_with_strategy(user_pref_dict, strategy="auto", topk=3, return_debug=False):
    """ระบบเลือกกลยุทธ์การแนะนำเพลง"""
    topk = max(1, min(topk, len(GENRES)))
    x, nonzero = preprocess_input(user_pref_dict)
    threshold = max(3, len(GENRES)//4)  # ปกติ = 4

    # auto: เลือก logreg หรือ knn ตามจำนวนข้อมูลที่ผู้ใช้กรอก
    chosen = strategy
    if strategy == "auto":
        chosen = "logreg" if nonzero >= threshold else "knn"

    if chosen == "self":
        full = x.astype(float)
        norm = normalize_minmax(full)

    elif chosen == "logreg":
        if nonzero == 0:
            full = x.astype(float)
            norm = normalize_minmax(full)
            chosen = "self"
        else:
            proba = LOGREG.predict_proba([x])[0]
            class_index_to_pos = {int(i): i for i in range(len(GENRES))}
            full = np.array([proba[class_index_to_pos[i]] for i in range(len(GENRES))], dtype=float)
            norm = np.clip(full, 0.0, 1.0)

    elif chosen == "knn":
        if nonzero == 0:
            x = np.ones_like(x) * 1e-3
        full = knn_scores(x, k=25)
        norm = normalize_minmax(full)

    else:
        raise ValueError("Unknown strategy")

    order = np.argsort(-norm)[:topk]
    items = [GENRES[i] for i in order]

    if return_debug:
        dbg = {"chosen_strategy": chosen, "nonzero": nonzero, "threshold": threshold}
        return {"items": items, "used": dbg}
    return {"items": items, "used": {"chosen_strategy": chosen}}

# =========================
# 🚀 FastAPI
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class RecommendByUserId(BaseModel):
    userId: str
    topk: int = 3
    strategy: str = "auto"

class RecommendByPrefs(BaseModel):
    preferences: dict
    topk: int = 3
    strategy: str = "auto"

@app.get("/")
def root():
    return {"status": "ok", "n_genres": len(GENRES)}

@app.post("/get_recommendations")
def get_recommendations(payload: RecommendByUserId):
    """เวอร์ชันใช้งานจริง: อ่าน preference จาก Firestore"""
    if not db:
        raise HTTPException(status_code=503, detail="Firebase not initialized")
    try:
        doc = db.collection("users").document(payload.userId).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        prefs = doc.to_dict().get("preference", {})
        if not isinstance(prefs, dict):
            raise HTTPException(status_code=400, detail="Invalid preference format")
        res = recommend_with_strategy(prefs, strategy=payload.strategy, topk=payload.topk, return_debug=True)
        return {"recommendations": res["items"], "used": res["used"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_recommendations_from_prefs")
def get_recommendations_from_prefs(payload: RecommendByPrefs):
    """เวอร์ชันทดสอบ: ส่ง preferences ตรง ๆ"""
    try:
        res = recommend_with_strategy(payload.preferences, strategy=payload.strategy, topk=payload.topk, return_debug=True)
        return {"recommendations": res["items"], "used": res["used"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

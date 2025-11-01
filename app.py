import os
import joblib
import numpy as np
import pandas as pd
import json

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # ✅ [เพิ่มใหม่] 1. Import CORS

# --- Firebase Admin Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------------------------------
# 1. 🔑 Initialize Firebase Admin
# ----------------------------------------------------
# เราจะใช้ไฟล์ 'serviceAccountKey.json' 
# ที่คุณดาวน์โหลดมาจาก Firebase
#
# *** สำหรับ Render.com (ตอน Deploy) ***
# 1. ห้ามอัปโหลด serviceAccountKey.json ขึ้น GitHub
# 2. ให้ Copy เนื้อหาในไฟล์ .json ทั้งหมด
# 3. ไปที่ Dashboard ของ Render -> App -> Environment
# 4. สร้าง "Secret File"
# 5. ตั้งชื่อ Path เป็น `serviceAccountKey.json`
# 6. วางเนื้อหา JSON ที่ copy มาลงไป
#
# Render จะสร้างไฟล์นี้ให้บน Server ตอนรันจริง
#
# *** สำหรับทดสอบบนคอม (Local Test) ***
# แค่วางไฟล์ `serviceAccountKey.json` ไว้ในโฟลเดอร์เดียวกับ `app.py`
# ----------------------------------------------------

try:
    cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin Initialized.")
except Exception as e:
    print(f"Error initializing Firebase Admin: {e}")
    # ถ้า Deploy บน Render แล้ว Error ตรงนี้ 
    # แปลว่าคุณยังไม่ได้ตั้งค่า Secret File
    db = None 

# ----------------------------------------------------
# 2. 🧠 Load Models
# ----------------------------------------------------
# โค้ดส่วนนี้จะรันแค่ครั้งแรกที่ Server ตื่น
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    logreg_model = joblib.load(os.path.join(BASE_DIR, 'logreg_model.joblib'))
    X_train_knn = np.load(os.path.join(BASE_DIR, 'X_train.npy'))
    y_train_knn = np.load(os.path.join(BASE_DIR, 'y_train.npy'))
    with open(os.path.join(BASE_DIR, 'model_data.json'), 'r') as f:
        all_genres = json.load(f)

    # Re-create the kNN model from loaded data
    from sklearn.neighbors import NearestNeighbors
    knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn_model.fit(X_train_knn)
    print("All models loaded successfully!")
except Exception as e:
    print(f"CRITICAL: Error loading models: {e}")

# ----------------------------------------------------
# 3. 📋 Copy Helper Functions from Notebook
# ----------------------------------------------------
# (!!!!)
# (!!!!)  สำคัญมาก: คัดลอกโค้ดฟังก์ชันทั้งหมด 
# (!!!!)  จากไฟล์ model1.ipynb มาวางแทนที่ '...' ตรงนี้
# (!!!!)
# ----------------------------------------------------

def preprocess_input(user_input, all_genres_list):
    """
    (!!!) คัดลอกโค้ดจาก Cell 'preprocess_input' ใน Notebook มาวางทับตรงนี้ (!!!)
    """
    # --- เริ่มโค้ดตัวอย่างจาก Notebook ---
    # (นี่คือโค้ดจาก Cell 2 ใน Notebook ของคุณ)
    freq_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    
    # แปลง user_input (dict) ให้อยู่ในรูป DataFrame แถวเดียว
    # (ต้องแน่ใจว่า all_genres_list คือ list ของชื่อแนวเพลงทั้งหมด)
    
    # สร้าง dict ว่างสำหรับ DataFrame โดยมี key เป็นแนวเพลงทั้งหมด
    df_data = {genre: 0 for genre in all_genres_list} # เริ่มด้วย 0 ทั้งหมด
    
    # อัปเดตค่าจาก user_input
    for genre, freq in user_input.items():
        if genre in df_data:
            if isinstance(freq, str):
                df_data[genre] = freq_map.get(freq, 0)
            elif isinstance(freq, (int, float)):
                df_data[genre] = freq
            
    # สร้าง DataFrame
    user_df = pd.DataFrame([df_data], columns=all_genres_list)
    
    # ตรวจสอบค่า NaN (ถ้ามี)
    user_df = user_df.fillna(0).astype(int)
    
    return user_df
    # --- จบโค้ดตัวอย่าง ---


def knn_recommend_topk(user_vector, k=5):
    """
    (!!!) คัดลอกโค้ดจาก Cell 'knn_recommend_topk' ใน Notebook มาวางทับตรงนี้ (!!!)
    """
    # --- เริ่มโค้ดตัวอย่างจาก Notebook ---
    # (นี่คือโค้ดจาก Cell 5 ใน Notebook ของคุณ)
    if not hasattr(knn_model, 'kneighbors'):
        raise Exception("knn_model is not fitted or loaded correctly")
        
    distances, indices = knn_model.kneighbors(user_vector)
    
    # ดึงแนวเพลงของผู้ใช้ที่ใกล้เคียง
    # y_train_knn คือ array ของแนวเพลงที่โหลดมาจาก .npy
    recommended_genres = y_train_knn[indices[0]]
    
    # นับความถี่
    from collections import Counter
    genre_counts = Counter(recommended_genres)
    
    # จัดลำดับ
    top_k_genres = [genre for genre, count in genre_counts.most_common(k)]
    return top_k_genres
    # --- จบโค้ดตัวอย่าง ---


def logreg_recommend(user_vector, k=5):
    """
    (!!!) คัดลอกโค้ดจาก Cell 'logreg_recommend' ใน Notebook มาวางทับตรงนี้ (!!!)
    """
    # --- เริ่มโค้ดตัวอย่างจาก Notebook ---
    # (นี่คือโค้ดจาก Cell 8 ใน Notebook ของคุณ)
    if not hasattr(logreg_model, 'predict_proba'):
        raise Exception("logreg_model is not loaded correctly")
        
    # ทำนายความน่าจะเป็นของทุก Class (แนวเพลง)
    proba = logreg_model.predict_proba(user_vector)[0]
    
    # จับคู่แนวเพลงกับความน่าจะเป็น
    # logreg_model.classes_ คือรายชื่อแนวเพลงที่โมเดลรู้จัก
    top_k_indices = np.argsort(proba)[::-1][:k] # เอา index ของ top k
    top_k_genres = logreg_model.classes_[top_k_indices]
    
    return list(top_k_genres)
    # --- จบโค้ดตัวอย่าง ---


def recommend_with_strategy(user_input_dict, strategy="auto", k=5):
    """
    (!!!) คัดลอกโค้ดจาก Cell 'recommend_with_strategy' ใน Notebook มาวางทับตรงนี้ (!!!)
    """
    # --- เริ่มโค้ดตัวอย่างจาก Notebook ---
    # (นี่คือโค้ดจาก Cell 9 ใน Notebook ของคุณ)
    
    # 1. Preprocess
    user_vector_df = preprocess_input(user_input_dict, all_genres)
    
    # นับจำนวนการให้เรตติ้ง (ที่ไม่ใช่ 'Never')
    num_ratings = sum(1 for v in user_input_dict.values() if (isinstance(v, str) and v != 'Never') or (isinstance(v, int) and v > 0))

    current_strategy = strategy
    
    # 2. Auto Strategy Logic
    if strategy == "auto":
        if num_ratings == 0:
            return [] # ไม่มีข้อมูล
        elif num_ratings < 10:
            current_strategy = "knn"
        else:
            current_strategy = "logreg"
    elif strategy == "self":
        current_strategy = "knn" # 'self' ใน Notebook คือ kNN
        
    # 3. Get recommendations
    if current_strategy == "knn":
        return knn_recommend_topk(user_vector_df, k=k)
    elif current_strategy == "logreg":
        return logreg_recommend(user_vector_df, k=k)
    else:
        return [] # หรือ Default
    # --- จบโค้ดตัวอย่าง ---

# ----------------------------------------------------
# 5. 🚀 FastAPI App
# ----------------------------------------------------
app = FastAPI()

# ✅ [เพิ่มใหม่] 2. เพิ่มการตั้งค่า CORS
origins = [
    "*"  # อนุญาตทั้งหมด (เหมาะสำหรับ Development)
    # "http://localhost",
    # "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # อนุญาตทุก Method (POST, GET)
    allow_headers=["*"], # อนุญาตทุก Header
)

# Pydantic model สำหรับรับข้อมูลจาก Flutter
class RequestBody(BaseModel):
    userId: str # รับ 'userId' (ถูกต้อง)

# Endpoint หลักที่ Flutter จะเรียก
@app.post("/get_recommendations")
async def handle_recommendation_request(body: RequestBody):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase Admin is not initialized.")

    try:
        # 1. ดึง Preferences จาก Firestore
        
        # (!!!) แก้ไขตรงนี้ (!!!)
        # เราจะใช้ body.userId (ถูกต้อง)
        doc_ref = db.collection('users').document(body.userId) 
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
            
        # (!!!) แก้ไขตรงนี้ (!!!)
        # เปลี่ยน 'genres' (ที่เป็น Array)
        # เป็น 'preferences' (ที่เป็น Map)
        user_preferences_map = doc.to_dict().get('preference') 
        
        if not user_preferences_map:
            raise HTTPException(status_code=404, detail="User 'preferences' field (Map) not found")

        # 2. รันโมเดล Preference
        # ส่ง Map ที่ได้จาก Firestore เข้าโมเดล
        recommendations = recommend_with_strategy(user_preferences_map, strategy="auto")
        
        # 3. ส่งผลลัพธ์กลับ
        return {"recommendations": recommendations}

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint สำหรับเช็คว่า Server ตื่น
@app.get("/")
def read_root():
    return {"status": "Music Recommender API is running!"}
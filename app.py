import os
import joblib
import numpy as np
import pandas as pd
import json

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- Firebase Admin Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------------------------------
# 1. 🔑 Initialize Firebase Admin
# ----------------------------------------------------
try:
    cred_path = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin Initialized.")
except Exception as e:
    print(f"Error initializing Firebase Admin: {e}")
    db = None 

# ----------------------------------------------------
# 2. 🧠 Load Models
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    logreg_model = joblib.load(os.path.join(BASE_DIR, 'logreg_model.joblib'))
    X_train_knn = np.load(os.path.join(BASE_DIR, 'X_train.npy'))
    y_train_knn = np.load(os.path.join(BASE_DIR, 'y_train.npy'))
    
    # ✅ [แก้ไข] BUG FIX: เราต้อง "เปิด" ไฟล์ JSON แล้ว "ดึง" ลิสต์ genres ออกมา
    with open(os.path.join(BASE_DIR, 'model_data.json'), 'r') as f:
        # โหลด "ตู้เอกสาร" ทั้งหมด (ที่มี 4 ลิ้นชัก)
        json_data = json.load(f) 
        # ดึงเฉพาะ "แฟ้ม" ที่เราต้องการ (ที่มี 16 แนวเพลง)
        all_genres_list = json_data['genres'] 

    from sklearn.neighbors import NearestNeighbors
    knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn_model.fit(X_train_knn)
    
    # ✅ [แก้ไข] พิมพ์ Log เพื่อยืนยัน
    print(f"All models loaded successfully! Found {len(all_genres_list)} genres.") 

except Exception as e:
    print(f"CRITICAL: Error loading models: {e}")

# ----------------------------------------------------
# 3. 📋 Copy Helper Functions from Notebook
# ----------------------------------------------------
# (ฟังก์ชันเหล่านี้ถูกต้องแล้ว ไม่ต้องแก้)

def preprocess_input(user_input, all_genres_list): # (all_genres_list จะได้รับ 16 รายการแล้ว)
    freq_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
    
    # ✅ [แก้ไข] BUG FIX: ตรงนี้จะสร้าง "กรอบรูป 16 ช่อง" ได้ถูกต้อง
    df_data = {genre: 0 for genre in all_genres_list} 
    
    for genre, freq in user_input.items():
        if genre in df_data:
            if isinstance(freq, str):
                df_data[genre] = freq_map.get(freq, 0)
            elif isinstance(freq, (int, float)):
                df_data[genre] = freq
    user_df = pd.DataFrame([df_data], columns=all_genres_list)
    user_df = user_df.fillna(0).astype(int)
    return user_df

def knn_recommend_topk(user_vector, k=5):
    if not hasattr(knn_model, 'kneighbors'):
        raise Exception("knn_model is not fitted or loaded correctly")
    distances, indices = knn_model.kneighbors(user_vector)
    recommended_genres = y_train_knn[indices[0]]
    from collections import Counter
    genre_counts = Counter(recommended_genres)
    top_k_genres = [genre for genre, count in genre_counts.most_common(k)]
    return top_k_genres

def logreg_recommend(user_vector, k=5):
    if not hasattr(logreg_model, 'predict_proba'):
        raise Exception("logreg_model is not loaded correctly")
    proba = logreg_model.predict_proba(user_vector)[0]
    top_k_indices = np.argsort(proba)[::-1][:k]
    top_k_genres = logreg_model.classes_[top_k_indices]
    return list(top_k_genres)

def recommend_with_strategy(user_input_dict, strategy="auto", k=5):
    # ✅ [แก้ไข] BUG FIX: ส่ง all_genres_list (16 รายการ) ไปให้
    user_vector_df = preprocess_input(user_input_dict, all_genres_list) 
    
    num_ratings = sum(1 for v in user_input_dict.values() if (isinstance(v, str) and v != 'Never') or (isinstance(v, int) and v > 0))
    current_strategy = strategy
    if strategy == "auto":
        if num_ratings == 0:
            return []
        elif num_ratings < 10:
            current_strategy = "knn"
        else:
            current_strategy = "logreg"
    elif strategy == "self":
        current_strategy = "knn"
    if current_strategy == "knn":
        return knn_recommend_topk(user_vector_df, k=k)
    elif current_strategy == "logreg":
        return logreg_recommend(user_vector_df, k=k)
    else:
        return []

# ----------------------------------------------------
# 5. 🚀 FastAPI App
# ----------------------------------------------------
app = FastAPI()

# (CORS Middleware - ถูกต้องแล้ว)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (Endpoint - ถูกต้องแล้ว)
class RequestBody(BaseModel):
    userId: str

@app.post("/get_recommendations")
async def handle_recommendation_request(body: RequestBody):
    if not db:
        raise HTTPException(status_code=503, detail="Firebase Admin is not initialized.")

    try:
        doc_ref = db.collection('users').document(body.userId)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
            
        user_preferences = doc.to_dict().get('preference') 
        if not user_preferences:
            raise HTTPException(status_code=404, detail="User preferences (Map) not found")

        # ✅ [แก้ไข] BUG FIX: ไม่ต้องส่ง all_genres_list อีก
        # เพราะฟังก์ชัน recommend_with_strategy จะไปใช้ตัวแปร global (all_genres_list) ที่เราโหลดไว้แล้ว
        recommendations = recommend_with_strategy(user_preferences, strategy="auto")
        
        return {"recommendations": recommendations}

    except Exception as e:
        print(f"Error during recommendation: {e}")
        # (พิมพ์ Error ที่แท้จริงออกมา)
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}") 

@app.get("/")
def read_root():
    return {"status": "Music Recommender API is running!"}


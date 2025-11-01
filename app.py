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
all_genres_list = [] # (ประกาศเป็น global)

try:
    logreg_model = joblib.load(os.path.join(BASE_DIR, 'logreg_model.joblib'))
    X_train_knn = np.load(os.path.join(BASE_DIR, 'X_train.npy'))
    y_train_knn = np.load(os.path.join(BASE_DIR, 'y_train.npy')) # <-- นี่คือตัวเลข (indexes)
    
    with open(os.path.join(BASE_DIR, 'model_data.json'), 'r') as f:
        json_data = json.load(f) 
        all_genres_list = json_data['genres'] # <-- นี่คือพจนานุกรม (names)

    from sklearn.neighbors import NearestNeighbors
    knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn_model.fit(X_train_knn)
    
    print(f"All models loaded successfully! Found {len(all_genres_list)} genres.") 

except Exception as e:
    print(f"CRITICAL: Error loading models: {e}")

# ----------------------------------------------------
# 3. 📋 Copy Helper Functions from Notebook
# ----------------------------------------------------

def preprocess_input(user_input, all_genres_list):
    # (ฟังก์ชันนี้ถูกต้องแล้ว)
    freq_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
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

# ✅ [แก้ไข] BUG FIX อยู่นี่ครับ
def knn_recommend_topk(user_vector, k=5):
    if not hasattr(knn_model, 'kneighbors'):
        raise Exception("knn_model is not fitted or loaded correctly")
        
    distances, indices = knn_model.kneighbors(user_vector)
    
    # 1. recommended_indexes คือ list ของตัวเลข (เช่น [11, 11, 14, 6, 11])
    recommended_indexes = y_train_knn[indices[0]] 
    
    from collections import Counter
    # 2. genre_counts คือ {11: 3, 14: 1, 6: 1}
    genre_counts = Counter(recommended_indexes) 
    
    # 3. top_k_indexes คือ [11, 14, 6] (เป็น numpy.int32)
    top_k_indexes = [genre_index for genre_index, count in genre_counts.most_common(k)]
    
    # 4. (!!!) แปลงตัวเลขกลับเป็นชื่อ (!!!)
    #    เราต้องใช้ all_genres_list (global) เป็นตัวแปล
    #    และใช้ int(idx) เพื่อแปลง numpy.int32 ให้เป็น Python int ธรรมดา
    try:
        top_k_names = [all_genres_list[int(idx)] for idx in top_k_indexes]
    except IndexError:
        # (กันเหนียว)
        print(f"Error: Index out of bound. Indexes: {top_k_indexes}, List size: {len(all_genres_list)}")
        return []
        
    # 5. คืนค่าเป็น List ของ String ที่ JSON-Safe
    return top_k_names

def logreg_recommend(user_vector, k=5):
    # (ฟังก์ชันนี้ถูกต้องแล้ว)
    if not hasattr(logreg_model, 'predict_proba'):
        raise Exception("logreg_model is not loaded correctly")
    proba = logreg_model.predict_proba(user_vector)[0]
    top_k_indices = np.argsort(proba)[::-1][:k]
    top_k_genres = logreg_model.classes_[top_k_indices]
    return list(top_k_genres)

def recommend_with_strategy(user_input_dict, strategy="auto", k=5):
    # (ฟังก์ชันนี้ถูกต้องแล้ว)
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

        recommendations = recommend_with_strategy(user_preferences, strategy="auto")
        
        # 'recommendations' ตอนนี้จะเป็น List[str] ที่ปลอดภัย
        return {"recommendations": recommendations}

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {e}") 

@app.get("/")
def read_root():
    return {"status": "Music Recommender API is running!"}


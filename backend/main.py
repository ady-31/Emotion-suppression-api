"""
Emotion Suppression Detection API
----------------------------------
Endpoint summary
  GET  /                  – health check
  POST /register-user     – save subject info to MongoDB
  POST /analyze-video     – upload a video, run full suppression pipeline,
                            return JSON with scores, emotions, timeline & latency
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from suppression.logic import run_video_pipeline
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
import tempfile
import os
from typing import Optional

app = FastAPI(title="Emotion Suppression Detection API")

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://UserData:TeraPass@cluster0.n7jxgnt.mongodb.net/?appName=Cluster0",
)
mongo_client     = MongoClient(MONGO_URI)
db               = mongo_client["emotion_suppression"]
users_collection = db["users"]

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class UserRegistration(BaseModel):
    name:   str
    email:  str
    phone:  str
    age:    Optional[str] = ""
    gender: Optional[str] = ""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/register-user")
async def register_user(user: UserRegistration):
    user_doc = user.dict()
    user_doc["created_at"] = datetime.utcnow()

    result = users_collection.update_one(
        {"email": user_doc["email"]},
        {"$set": user_doc},
        upsert=True,
    )

    user_id = str(result.upserted_id) if result.upserted_id else None
    return {
        "message": "User registered successfully",
        "user_id": user_id,
        "email":   user_doc["email"],
    }


@app.post("/analyze-video")
async def analyze_video(
    video: UploadFile = File(..., description="Video file to analyse (mp4, avi, mov, …)"),
):
    """
    Accepts a video upload and runs the full emotion-suppression pipeline:
      1. OpenFace  → Action Unit CSV
      2. LSTM      → per-window suppression scores
      3. DeepFace  → dominant visible emotion
      4. Speech    → latency events

    Response JSON shape
    -------------------
    {
      "suppression_score":  float,   // raw LSTM mean
      "normalized_score":   float,   // 0-1 clamped
      "level":              str,     // Low / Moderate / High Suppression
      "dominant_emotion":   str|null,
      "suppressed_emotion": str|null,
      "timeline":           [{time, score}, …],
      "latency_events":     [{time, duration}, …],
      "files_processed":    1
    }
    """
    # Preserve the original extension so OpenFace can detect the codec
    original_ext = os.path.splitext(video.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=original_ext)
    try:
        tmp.write(await video.read())
        tmp.close()
        result = run_video_pipeline(tmp.name)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ── Dev entry point ────────────────────────────────────────────────────────────
# Run with:  uvicorn main:app --reload --port 8000

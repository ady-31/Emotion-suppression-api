from fastapi import FastAPI, UploadFile, File
from suppression.logic import run_suppression_pipeline
import tempfile
import os
from typing import List

app = FastAPI(title="Emotion Suppression Detection API")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/analyze")
async def analyze(
    au_files: List[UploadFile] = File(..., description="Upload one or more AU CSV files"),
    valence_files: List[UploadFile] = File(..., description="Upload one or more valence CSV files")
):
    # Save all AU files
    au_paths = []
    for au_file in au_files:
        au_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        au_tmp.write(await au_file.read())
        au_tmp.close()
        au_paths.append(au_tmp.name)
    
    # Save all valence files
    val_paths = []
    for val_file in valence_files:
        val_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        val_tmp.write(await val_file.read())
        val_tmp.close()
        val_paths.append(val_tmp.name)
    
    # Run the pipeline with multiple files
    result = run_suppression_pipeline(au_paths, val_paths)
    
    # Cleanup temp files
    for path in au_paths + val_paths:
        try:
            os.unlink(path)
        except:
            pass
    
    return result

from io import BytesIO
import os

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model_selector import ModelSelector


app = FastAPI(title="Modelwise")
selector = ModelSelector()

frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        'FRONTEND_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    target: str = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")

    content = await file.read()

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "The maximum file size is 25 MB.")

    try:
        frame = pd.read_csv(BytesIO(content), low_memory=False)
        return selector.recommend(frame, target)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise HTTPException(
            500,
            "The dataset could not be analyzed.",
        ) from error

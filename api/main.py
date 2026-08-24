from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import shutil
import uuid
from api.models.summarizer import summarizer_instance
from utils.parsers import parse_file, parse_url
from utils.metrics import get_metrics, extract_keywords
from api.database import engine, get_db
from api.models import schemas
from api import crud

# Create database tables
schemas.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Text Distiller API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.get("/")
async def root():
    return {"message": "Welcome to Text Distiller API"}

@app.get("/summaries")
async def get_all_summaries(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    summaries = crud.get_summaries(db, skip=skip, limit=limit)
    return summaries

@app.delete("/summaries/{summary_id}")
async def delete_summary(summary_id: int, db: Session = Depends(get_db)):
    success = crud.delete_summary(db, summary_id)
    if not success:
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"message": "Summary deleted successfully"}

@app.post("/summarize")
async def summarize(
    text: str = Form(None),
    url: str = Form(None),
    file: UploadFile = File(None),
    style: str = Form("abstractive"),
    length: str = Form("medium"),
    db: Session = Depends(get_db)
):
    source_text = ""
    title = "Summarized Content"

    # 1. Extract text based on input type
    if file:
        file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        try:
            source_text = parse_file(file_path, file.filename)
            title = file.filename
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    elif url:
        try:
            source_text, title = parse_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse URL: {str(e)}")
    elif text:
        source_text = text
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    if not source_text or len(source_text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short to summarize")

    # 2. Set length parameters
    max_len = 150
    min_len = 50
    if length == "short":
        max_len, min_len = 80, 30
    elif length == "detailed":
        max_len, min_len = 300, 100

    # 3. Perform summarization
    try:
        if style == "abstractive":
            summary = summarizer_instance.summarize_abstractive(source_text, max_length=max_len, min_length=min_len)
        else:
            ratio = 0.2 if length == "short" else 0.4 if length == "detailed" else 0.3
            summary = summarizer_instance.summarize_extractive(source_text, ratio=ratio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    # 4. Extract insights
    metrics = get_metrics(source_text, summary)
    keywords = extract_keywords(source_text)

    # 5. Save to database
    db_summary = crud.create_summary(
        db=db, 
        title=title, 
        original_text=source_text, 
        summary=summary, 
        style=style, 
        length=length
    )

    return {
        "id": db_summary.id,
        "title": title,
        "original_text": source_text,
        "summary": summary,
        "metrics": metrics,
        "keywords": keywords,
        "style": style,
        "length": length,
        "created_at": db_summary.created_at
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

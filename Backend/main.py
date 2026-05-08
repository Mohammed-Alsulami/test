import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"


@app.get("/")
def home():
    return {"message": "Backend is running"}


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    from function import run_model

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_model(file_path)

    return result


@app.get("/download-report/{report_filename}")
def download_report(report_filename: str):
    report_path = os.path.join(OUTPUT_FOLDER, report_filename)

    if not os.path.exists(report_path):
        return {"error": "Report not found"}

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=report_filename
    )

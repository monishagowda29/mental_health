"""
app/api/endpoints/analysis.py
FastAPI routing handlers for asynchronous text classification tasks.
"""
import io
from typing import Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from celery.result import AsyncResult
from app.tasks.worker import queue_text_analysis, queue_image_scan
from app.services.pdf_generator import PDFReportGenerator

router = APIRouter()

class TextAnalysisRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient Name or ID profile key.", example="PAT-1082")
    raw_text: str = Field(..., max_length=2500, description="Raw input text to analyze.", example="I feel extremely sad.")
    language_hint: str = Field("auto", description="Source language BCP-47 hint (e.g. 'kn', 'hi', 'en', 'auto').", example="auto")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None



@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_text(payload: TextAnalysisRequest):
    """
    Submits a text analysis request. Enqueues an asynchronous worker task and returns its ID.
    """
    try:
        task = queue_text_analysis.delay(
            patient_id=payload.patient_id,
            text=payload.raw_text,
            lang_hint=payload.language_hint
        )
        return {"task_id": task.id, "status": "PENDING"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue background task: {exc}"
        ) from exc

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Queries the message broker for the execution state of the specified task ID.
    """
    res = AsyncResult(task_id)
    
    if res.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "SUCCESS",
            "result": res.result
        }
    elif res.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "result": {"error": str(res.result)}
        }
    else:
        return {
            "task_id": task_id,
            "status": res.state,
            "result": None
        }

@router.post("/scan", status_code=status.HTTP_202_ACCEPTED)
async def scan_image(
    patient_id: str = Form(...),
    mode: str = Form("general"),
    file: UploadFile = File(...)
):
    """
    Submits a scanned sheet image for perspective correction and clinical sentiment analysis.
    """
    try:
        content = await file.read()
        hex_content = content.hex()
        task = queue_image_scan.delay(
            patient_id=patient_id,
            image_bytes_hex=hex_content,
            mode=mode
        )
        return {"task_id": task.id, "status": "PENDING"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue image analysis task: {exc}"
        ) from exc

class ReportGenerationRequest(BaseModel):
    patient_id: str
    results: dict

@router.post("/report")
async def generate_encrypted_report(payload: ReportGenerationRequest):
    """
    Generates a ReportLab clinical screening PDF and encrypts it using AES-256 with the patient ID.
    """
    try:
        generator = PDFReportGenerator()
        pdf_bytes = generator.generate_report(payload.patient_id, payload.results)
        
        # Return as a downloadable streaming response
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=mindscan_report_{payload.patient_id}.pdf"
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate encrypted PDF report: {exc}"
        ) from exc

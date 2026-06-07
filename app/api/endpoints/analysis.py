"""
app/api/endpoints/analysis.py
FastAPI routing handlers for asynchronous text classification tasks.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from celery.result import AsyncResult
from app.tasks.worker import queue_text_analysis

router = APIRouter()

class TextAnalysisRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient Name or ID profile key.", example="PAT-1082")
    raw_text: str = Field(..., max_length=2500, description="Raw input text to analyze.", example="I feel extremely sad.")
    language_hint: str = Field("auto", description="Source language BCP-47 hint (e.g. 'kn', 'hi', 'en', 'auto').", example="auto")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None

from typing import Optional

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

"""
app/main.py
FastAPI backend service bootstrap for MindScan.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import analysis

app = FastAPI(
    title="MindScan AI Platform",
    description="HIPAA-compliant, local-first mental health text classification and translation backend.",
    version="2.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])

@app.get("/health")
def health_check():
    return {"status": "ONLINE", "message": "MindScan AI Platform operational."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

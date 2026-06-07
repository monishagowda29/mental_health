import os
import sys
import time
from pathlib import Path

# Adjust search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.tasks.worker import queue_text_analysis

def main():
    print("Submitting task to Celery...")
    # Queue the analysis task
    task = queue_text_analysis.delay(
        patient_id="PAT-TEST-001",
        text="I have been feeling extremely anxious and worried about my health lately.",
        lang_hint="en"
    )
    print(f"Task submitted. Task ID: {task.id}")
    print("Waiting for task completion...")
    
    # Wait for completion (timeout 30s)
    start_time = time.time()
    while not task.ready():
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print("Timeout waiting for task execution!")
            sys.exit(1)
            
    print(f"Task state: {task.state}")
    if task.state == "SUCCESS":
        result = task.result
        print("Task succeeded! Result:")
        print(result)
        # Verify result contains the expected keys
        assert "prediction" in result, "Result missing prediction"
        assert "patient_id" in result, "Result missing patient_id"
        assert result["patient_id"] == "PAT-TEST-001"
        print("[SUCCESS] Celery task transaction verified via Redis broker!")
    else:
        print(f"Task failed with state: {task.state}")
        print(f"Error: {task.result}")
        sys.exit(1)

if __name__ == "__main__":
    main()

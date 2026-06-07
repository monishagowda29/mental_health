"""
src/services/onnx_base.py
Base service helper class for optimizing local ONNX Runtime execution sessions.
"""
import logging
import os
from typing import List, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)

class ONNXBaseService:
    """
    Base service facilitating optimized onnxruntime Session initialization.
    Configures CPU execution providers, thread scaling, and session states.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self._init_session()

    def _init_session(self):
        if not os.path.exists(self.model_path):
            logger.error("ONNX model weights not found at target path: %s", self.model_path)
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")

        logger.info("Initializing ONNX Inference Session for %s ...", self.model_path)
        
        # Configure execution options for fast localized CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 1)  # Scale to safe execution thread limit
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load session using CPUExecutionProvider
        self.session = ort.InferenceSession(
            self.model_path,
            opts,
            providers=["CPUExecutionProvider"]
        )
        logger.info("ONNX Session successfully initialized for: %s", self.model_path)

    def run_inference(self, input_feed: dict) -> List[ort.NodeArg]:
        """
        Runs model forward pass inside the loaded ONNX runtime session.
        """
        if self.session is None:
            raise RuntimeError("ONNX Session is not initialized.")
        return self.session.run(None, input_feed)

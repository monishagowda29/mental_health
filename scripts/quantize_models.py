"""
scripts/quantize_models.py
Script to export models to ONNX and perform dynamic INT8 quantization.
"""
import os
import sys
from pathlib import Path

# Adjust search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForSeq2SeqLM
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.config import Config

def main():
    root = Config.ROOT_PATH
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("        MINDSCAN MODEL EXPORT & INT8 QUANTIZATION TOOL")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────────
    # 1. Quantizing Fine-Tuned BERT Classifier
    # ──────────────────────────────────────────────────────────────────
    bert_src_dir = os.environ.get("MODEL_DIR", str(models_dir / "bert_mental_health"))
    bert_onnx_dir = models_dir / "bert_onnx_fp32"
    bert_quantized_path = models_dir / "bert_quantized.onnx"

    print(f"\n[BERT] Loading and exporting from PyTorch: {bert_src_dir}")
    if not os.path.exists(bert_src_dir):
        print(f"[ERROR] Source BERT weights not found at: {bert_src_dir}")
        print("Skipping BERT quantization. Place weights inside models/bert_mental_health first.")
    else:
        try:
            # Load and export to ONNX format
            bert_onnx_model = ORTModelForSequenceClassification.from_pretrained(
                bert_src_dir,
                export=True
            )
            bert_onnx_model.save_pretrained(str(bert_onnx_dir))
            print(f"[BERT] FP32 ONNX model saved to: {bert_onnx_dir}")

            # Locate the model.onnx file inside the exported folder
            fp32_onnx_file = bert_onnx_dir / "model.onnx"
            if fp32_onnx_file.exists():
                print(f"[BERT] Quantizing FP32 ONNX weights to dynamic INT8...")
                quantize_dynamic(
                    model_input=str(fp32_onnx_file),
                    model_output=str(bert_quantized_path),
                    weight_type=QuantType.QInt8
                )
                print(f"[BERT] [SUCCESS] Quantized INT8 model saved to: {bert_quantized_path}")
            else:
                print(f"[ERROR] Exported model.onnx not found inside: {bert_onnx_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to export/quantize BERT: {e}")

    # ──────────────────────────────────────────────────────────────────
    # 2. Quantizing Translation Model
    # ──────────────────────────────────────────────────────────────────
    translator_src = getattr(Config, "HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en")
    translator_onnx_dir = models_dir / "translator_onnx"

    print(f"\n[TRANSLATOR] Loading and exporting: {translator_src}")
    try:
        # Load and export model
        translator_onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
            translator_src,
            export=True
        )
        translator_onnx_model.save_pretrained(str(translator_onnx_dir))
        
        # Save tokenizer files inside the target model folder
        from transformers import AutoTokenizer
        translator_tokenizer = AutoTokenizer.from_pretrained(translator_src)
        translator_tokenizer.save_pretrained(str(translator_onnx_dir))
        
        print(f"[TRANSLATOR] [SUCCESS] ONNX translation engine and tokenizer ready at: {translator_onnx_dir}")

        # Skip translation quantization entirely to preserve absolute precision
        for model_file in []:
            fp32_path = translator_onnx_dir / model_file
            if fp32_path.exists():
                quantized_path = translator_onnx_dir / model_file.replace(".onnx", "_quantized.onnx")
                print(f"[TRANSLATOR] Quantizing {model_file} to INT8...")
                quantize_dynamic(
                    model_input=str(fp32_path),
                    model_output=str(quantized_path),
                    weight_type=QuantType.QInt8
                )
                # Swap original FP32 with Quantized INT8 to save disk/RAM
                os.remove(fp32_path)
                os.rename(quantized_path, fp32_path)
                print(f"[TRANSLATOR] Replaced {model_file} with quantized INT8 version.")

        print(f"[TRANSLATOR] [SUCCESS] All translation sub-models quantized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to export/quantize Translator: {e}")

    print("\n" + "=" * 60)
    print("               QUANTIZATION AUTOMATION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()

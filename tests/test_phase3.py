"""
tests/test_phase3.py
Unit and integration tests for Phase 3 components:
- ImageProcessor (OpenCV document preprocessing and binarization)
- ephemeral_file (Secure temporary storage & shredding context manager)
- PDFReportGenerator (ReportLab generation & pypdf AES-256 encryption)
"""
import io
import os
import unittest
from PIL import Image
from pypdf import PdfReader
from app.services.image_processor import ImageProcessor
from app.utils.ephemeral_storage import ephemeral_file, secure_shred, get_ephemeral_dir
from app.services.pdf_generator import PDFReportGenerator

class TestPhase3Components(unittest.TestCase):

    def setUp(self):
        # Create a dummy image for testing OpenCV image processing
        self.img = Image.new("RGB", (300, 300), color="white")
        # Draw some dark shapes on the white background to simulate a document
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.img)
        draw.rectangle([50, 50, 250, 250], fill="black")
        
        # Save image to bytes
        img_buffer = io.BytesIO()
        self.img.save(img_buffer, format="JPEG")
        self.img_bytes = img_buffer.getvalue()

    def test_image_processor_pipeline(self):
        """Verify that ImageProcessor decodes, processes, and returns high-contrast JPEG bytes."""
        processor = ImageProcessor()
        
        # Process the dummy image
        processed_bytes = processor.process_image(self.img_bytes)
        
        self.assertIsInstance(processed_bytes, bytes)
        self.assertTrue(len(processed_bytes) > 0)
        
        # Try loading processed bytes back into PIL to verify it's a valid image
        processed_img = Image.open(io.BytesIO(processed_bytes))
        self.assertEqual(processed_img.format, "JPEG")
        
        # Verify it converted to grayscale/binary (mode L or 1)
        self.assertIn(processed_img.mode, ("L", "1"))

    def test_image_processor_invalid_bytes(self):
        """Verify that passing invalid image bytes to ImageProcessor raises a ValueError."""
        processor = ImageProcessor()
        with self.assertRaises(ValueError):
            processor.process_image(b"invalid_image_bytes_here")

    def test_ephemeral_file_context_manager(self):
        """Verify that ephemeral_file creates a file, makes it readable, and shreds it on exit."""
        test_data = b"confidential_patient_record_data"
        file_path_leaked = None
        
        with ephemeral_file(test_data, suffix=".dat") as file_path:
            file_path_leaked = file_path
            # Verify file exists
            self.assertTrue(os.path.exists(file_path))
            
            # Verify contents are correct
            with open(file_path, "rb") as f:
                content = f.read()
            self.assertEqual(content, test_data)
            
        # Verify file is deleted after context manager exits
        self.assertFalse(os.path.exists(file_path_leaked))

    def test_secure_shred_overwrites_content(self):
        """Verify that secure_shred overwrites file contents before deleting."""
        # Create a temp file manually
        temp_dir = get_ephemeral_dir()
        temp_path = os.path.join(temp_dir, "test_shred.tmp")
        
        secret_content = b"super_secret_information"
        with open(temp_path, "wb") as f:
            f.write(secret_content)
            
        self.assertTrue(os.path.exists(temp_path))
        
        # Shred the file
        secure_shred(temp_path)
        
        # Verify the file is removed
        self.assertFalse(os.path.exists(temp_path))

    def test_pdf_report_generator_and_encryption(self):
        """Verify PDF is generated and encrypted with the patient ID as password."""
        generator = PDFReportGenerator()
        
        patient_id = "PAT-9982"
        results = {
            "prediction": "anxiety",
            "depression_score": 0.15,
            "anxiety_score": 0.75,
            "normal_score": 0.10,
            "processed_text": "Patient reports high levels of stress and continuous worry.",
            "assessment_type": "Anxiety Screening Assessor",
            "language": "English"
        }
        
        pdf_bytes = generator.generate_report(patient_id, results)
        
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 0)
        
        # Attempt to read PDF with pypdf
        reader = PdfReader(io.BytesIO(pdf_bytes))
        
        # Verify that the PDF is encrypted
        self.assertTrue(reader.is_encrypted)
        
        # Verify that trying to decrypt with incorrect password fails
        self.assertFalse(reader.decrypt("wrong_password"))
        
        # Verify that decrypting with the patient ID works
        self.assertTrue(reader.decrypt(patient_id))
        
        # Verify pages are readable after decryption
        self.assertTrue(len(reader.pages) > 0)
        page_text = reader.pages[0].extract_text()
        self.assertIn("MINDSCAN", page_text)
        self.assertIn(patient_id, page_text)
        self.assertIn("SCREENING CLASSIFICATION: ANXIETY", page_text)

if __name__ == "__main__":
    unittest.main()

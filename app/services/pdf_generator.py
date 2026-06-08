"""
app/services/pdf_generator.py
PDF generation service using ReportLab.
Wraps the output document with AES-256 encryption using pypdf, with the patient ID as the decryption password.
"""
import io
import logging
import uuid
from datetime import datetime
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """
    Service to generate clinical-grade mental health screening PDF reports with AES-256 encryption.
    """

    def generate_report(self, patient_id: str, results: dict) -> bytes:
        """
        Generates a styled clinical report in PDF format and encrypts it with AES-256.
        
        Args:
            patient_id (str): The patient ID or name (used as the encryption password).
            results (dict): The screening result metrics dictionary.
            
        Returns:
            bytes: The encrypted PDF document bytes.
        """
        logger.info("Starting PDF report generation for patient ID: %s", patient_id)
        
        # 1. Create a memory buffer for ReportLab
        pdf_buffer = io.BytesIO()
        
        # 2. Build the ReportLab PDF document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )
        
        # Style sheet setup
        styles = getSampleStyleSheet()
        
        # Define clean, calming corporate styling colors
        primary_color = colors.HexColor("#4f46e5")    # Indigo
        text_dark = colors.HexColor("#1f2937")        # Dark slate
        text_light = colors.HexColor("#4b5563")       # Cool grey
        bg_light = colors.HexColor("#f3f4f6")         # Very light grey
        border_color = colors.HexColor("#e5e7eb")     # Light border
        
        # Custom Typography Styles
        title_style = ParagraphStyle(
            name="ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=primary_color,
            alignment=TA_LEFT,
            spaceAfter=15
        )
        
        subtitle_style = ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_light,
            spaceAfter=25
        )
        
        section_heading = ParagraphStyle(
            name="SectionHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=primary_color,
            spaceBefore=15,
            spaceAfter=10
        )
        
        body_style = ParagraphStyle(
            name="ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=text_dark,
            spaceAfter=10
        )
        
        bold_body_style = ParagraphStyle(
            name="ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold"
        )
        
        disclaimer_style = ParagraphStyle(
            name="ReportDisclaimer",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#9ca3af"),
            spaceBefore=30
        )
        
        story = []
        
        # --- Header Section ---
        story.append(Paragraph("MINDSCAN SCREENING ANALYSIS REPORT", title_style))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated on: {current_time} (UTC) | Clinical Confidential Document", subtitle_style))
        story.append(Spacer(1, 10))
        
        # --- Metadata Table ---
        meta_data = [
            [Paragraph("Patient Profile Key:", bold_body_style), Paragraph(patient_id, body_style)],
            [Paragraph("Assessment Modality:", bold_body_style), Paragraph(results.get("assessment_type", "Text-Based Mental Health Screening"), body_style)],
            [Paragraph("Language Mode:", bold_body_style), Paragraph(results.get("language", "English (or NMT Translated)"), body_style)],
            [Paragraph("Screening Pipeline Version:", bold_body_style), Paragraph("MindScan local-first v2.0.0", body_style)]
        ]
        
        meta_table = Table(meta_data, colWidths=[150, 350])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_light),
            ('TEXTCOLOR', (0, 0), (-1, -1), text_dark),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 1, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 25))
        
        # --- Screening Results Section ---
        story.append(Paragraph("Assessment Classification & Sentiment", section_heading))
        
        pred_label = results.get("prediction", "normal").upper()
        # Choose colored status box based on prediction
        if "SEVERE" in pred_label or pred_label == "DEPRESSION":
            status_color = colors.HexColor("#ef4444")  # Red
        elif "MODERATE" in pred_label or "MILD" in pred_label or pred_label == "ANXIETY":
            status_color = colors.HexColor("#f59e0b")  # Amber
        else:
            status_color = colors.HexColor("#10b981")  # Green
            
        status_text_style = ParagraphStyle(
            name="StatusText",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.white,
            alignment=TA_CENTER
        )
        
        status_box_data = [[Paragraph(f"SCREENING CLASSIFICATION: {pred_label}", status_text_style)]]
        status_box_table = Table(status_box_data, colWidths=[500])
        status_box_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), status_color),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(status_box_table)
        story.append(Spacer(1, 20))
        
        # --- Probability / Analytics Grid ---
        story.append(Paragraph("Model Confidence Distribution", section_heading))
        
        prob_data = [
            [Paragraph("Indicator Metric", bold_body_style), Paragraph("Confidence Probability", bold_body_style)],
            [Paragraph("Depression Score:", body_style), Paragraph(f"{results.get('depression_score', 0.0) * 100:.1f}%", body_style)],
            [Paragraph("Anxiety Score:", body_style), Paragraph(f"{results.get('anxiety_score', 0.0) * 100:.1f}%", body_style)],
            [Paragraph("Normal / Wellness Score:", body_style), Paragraph(f"{results.get('normal_score', 0.0) * 100:.1f}%", body_style)]
        ]
        
        prob_table = Table(prob_data, colWidths=[250, 250])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), bg_light),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('LINEBELOW', (0, 0), (-1, 0), 1, border_color),
            ('BOX', (0, 0), (-1, -1), 1, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 20))
        
        # --- Analyzed Content Section ---
        if "processed_text" in results:
            story.append(Paragraph("De-Identified Assessment Input Text", section_heading))
            story.append(Paragraph(results["processed_text"], body_style))
            story.append(Spacer(1, 15))
            
        # --- Disclaimers ---
        disclaimer_text = (
            "DISCLAIMER: This document is a screening aid generated using localized artificial intelligence "
            "classification models (quantized dynamic INT8 BERT architectures) and local neural machine translation "
            "modules. This report does NOT represent a clinical diagnosis, psychiatric consultation, or medical opinion. "
            "All model classification parameters represent statistical linguistic correlations rather than medical evaluation. "
            "Please review this document with a qualified, licensed healthcare provider."
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build document
        doc.build(story)
        
        # Get raw PDF bytes
        raw_pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        # 3. Encrypt the PDF using pypdf and AES-256
        logger.info("Applying AES-256 encryption to the generated PDF...")
        try:
            reader = PdfReader(io.BytesIO(raw_pdf_bytes))
            writer = PdfWriter()
            
            # Copy all pages to the writer
            for page in reader.pages:
                writer.add_page(page)
                
            # Set patient_id as the user password. Generate a secure fallback owner password.
            owner_pwd = f"{patient_id}_owner_{uuid.uuid4().hex}"
            
            # pypdf supports AES-256 since v3.0.0
            writer.encrypt(
                user_password=patient_id,
                owner_password=owner_pwd,
                algorithm="AES-256"
            )
            
            encrypted_buffer = io.BytesIO()
            writer.write(encrypted_buffer)
            encrypted_bytes = encrypted_buffer.getvalue()
            encrypted_buffer.close()
            
            logger.info("Successfully generated encrypted PDF (size: %d bytes).", len(encrypted_bytes))
            return encrypted_bytes
        except Exception as exc:
            logger.exception("Failed to apply AES-256 encryption to PDF: %s", exc)
            raise RuntimeError(f"Failed to encrypt generated PDF report: {exc}") from exc

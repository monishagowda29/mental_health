"""
app/services/image_processor.py
Document image scanning service using OpenCV.
Detects document boundaries, performs perspective warping, and applies adaptive threshold binarization
to optimize OCR readability.
"""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Service for processing and normalizing documents/scans of mental health screening sheets.
    """

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates as: top-left, top-right, bottom-right, bottom-left.
        pts shape: (4, 2)
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference, bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect

    @staticmethod
    def warp_perspective(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Applies a perspective warp to obtain a flat, top-down view of the document.
        """
        rect = ImageProcessor.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new warped image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Compute the height of the new warped image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Construct destination points for the top-down view
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # Calculate transformation matrix and warp
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped

    def process_image(self, img_bytes: bytes) -> bytes:
        """
        Main pipeline to decode image, detect document boundaries, warp perspective,
        and apply adaptive threshold binarization.
        
        Args:
            img_bytes: Input raw image bytes.
            
        Returns:
            bytes: Processed, high-contrast, warped black & white image as JPEG bytes.
        """
        logger.info("Decoding input image bytes (length: %d bytes)...", len(img_bytes))
        
        # 1. Decode image bytes to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("Failed to decode image bytes. Image is corrupted or invalid format.")
            raise ValueError("Invalid image bytes provided.")

        orig = image.copy()
        height, width = image.shape[:2]
        
        # 2. Downscale image for faster and more accurate contour detection
        ratio = height / 500.0
        resized = cv2.resize(image, (int(width / ratio), 500))
        
        # 3. Contour detection
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter smooths noise while keeping sharp edges
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection using Canny
        edged = cv2.Canny(smoothed, 75, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        doc_contour = None
        for contour in contours:
            # Approximate the contour with a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # If our approximated contour has four vertices, we can assume we found the document
            if len(approx) == 4:
                doc_contour = approx
                break
                
        # 4. Warp perspective if document boundary found
        if doc_contour is not None:
            logger.info("Document contour detected. Applying perspective warp...")
            # Scale coordinates back to original image size
            pts = doc_contour.reshape(4, 2) * ratio
            warped = self.warp_perspective(orig, pts)
        else:
            logger.warning("No document contour with 4 vertices found. Falling back to default image bounds.")
            warped = orig

        # 5. Adaptive Thresholding for Binarization (enhancing text contrast)
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray_warped,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # 6. Encode processed image back to JPEG bytes
        success, encoded_img = cv2.imencode(".jpg", binarized)
        if not success:
            logger.error("Failed to encode processed image to JPEG format.")
            raise RuntimeError("Failed to encode processed image.")
            
        logger.info("Image processing pipeline completed successfully.")
        return encoded_img.tobytes()

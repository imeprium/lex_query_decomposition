"""
PDF signing utilities for adding visual signatures to PDF documents.
"""
import os
import uuid
import datetime
import logging
from io import BytesIO
from functools import lru_cache
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, blue, gray
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger("pdf")


class VisualPDFSigner:
    """
    A PDF signer that adds visual signatures without requiring cryptographic libraries.
    Uses PyPDF2 and ReportLab to add a visual signature box to the last page.
    """

    def __init__(self, logo_path=None, signature_path=None):
        """
        Initialize the visual PDF signer

        Args:
            logo_path: Path to organization logo
            signature_path: Path to signature image
        """
        # Ensure directories exist
        Path("static").mkdir(exist_ok=True)
        Path("static/signatures").mkdir(exist_ok=True, parents=True)

        # Set paths with fallbacks
        self.logo_path = logo_path or "static/lexanalytics_logo.png"
        self.signature_path = signature_path or "static/signatures/signature.png"

        # Check if signature file exists
        if not os.path.exists(self.signature_path):
            logger.warning(f"Signature file not found at {self.signature_path}")

    def sign_pdf(self, pdf_data, reason="Document verification", location="Digital"):
        """
        Add a visual signature to the PDF

        Args:
            pdf_data: PDF document data as bytes
            reason: Reason for signing
            location: Location of signing

        Returns:
            PDF data with visual signature
        """
        try:
            # Load the PDF using PyPDF2
            reader = PdfReader(BytesIO(pdf_data))
            writer = PdfWriter()

            # Get the number of pages
            num_pages = len(reader.pages)

            # Process each page
            for i in range(num_pages):
                # Get the page
                page = reader.pages[i]

                # Add visual signature to last page only
                if i == num_pages - 1:
                    page = self._add_visual_signature(page, reason, location)

                # Add the page to the writer
                writer.add_page(page)

            # Save the modified PDF to a BytesIO object
            output = BytesIO()
            writer.write(output)

            logger.info("Successfully added visual signature to PDF")
            return output.getvalue()

        except Exception as e:
            logger.error(f"Error adding visual signature to PDF: {str(e)}", exc_info=True)
            # Return the original PDF on error
            return pdf_data

    def _add_visual_signature(self, page, reason, location):
        """
        Add a visual signature box to a PDF page

        Args:
            page: PDF page to add signature to
            reason: Reason for signature
            location: Location of signature

        Returns:
            Modified PDF page
        """
        try:
            # Create a new PDF with just the signature
            signature_pdf = BytesIO()
            c = canvas.Canvas(signature_pdf, pagesize=A4)

            # Get page size
            page_width, page_height = A4

            # Draw signature box at bottom of page
            self._draw_signature_box(c, page_width, page_height, reason, location)

            # Save the canvas
            c.save()

            # Merge the signature with the original page
            signature_pdf.seek(0)
            signature_reader = PdfReader(signature_pdf)
            signature_page = signature_reader.pages[0]

            # Merge the signature page with the original page
            page.merge_page(signature_page)

            return page

        except Exception as e:
            logger.error(f"Error adding visual signature elements: {str(e)}")
            return page

    def _draw_signature_box(self, c, page_width, page_height, reason, location):
        """
        Draw signature box on canvas

        Args:
            c: Canvas to draw on
            page_width: Page width
            page_height: Page height
            reason: Reason for signature
            location: Location of signature
        """
        # Draw signature box
        c.setStrokeColor(gray)
        c.setFillColor(gray)
        c.setLineWidth(0.5)
        c.rect(2 * cm, 2 * cm, page_width - 4 * cm, 3 * cm, fill=0)

        # Add signature information
        c.setFont("Helvetica", 10)
        c.setFillColor(black)

        # Add document ID
        document_id = str(uuid.uuid4())
        c.drawString(2.5 * cm, 4.5 * cm, f"Document ID: {document_id}")

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(2.5 * cm, 4 * cm, f"Timestamp: {timestamp}")

        # Add reason and location
        c.drawString(2.5 * cm, 3.5 * cm, f"Reason: {reason}")
        c.drawString(2.5 * cm, 3 * cm, f"Location: {location}")

        # Add "DIGITALLY VERIFIED" text
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(blue)
        c.drawString(page_width - 10 * cm, 4.5 * cm, "DIGITALLY VERIFIED")

        # Add signature image if available
        if os.path.exists(self.signature_path):
            try:
                c.drawImage(
                    self.signature_path,
                    page_width - 6 * cm,
                    2.5 * cm,
                    width=3 * cm,
                    height=1.5 * cm,
                    preserveAspectRatio=True
                )
            except Exception as e:
                logger.error(f"Error drawing signature image: {str(e)}")


# Factory function
@lru_cache(maxsize=1)
def get_pdf_signer():
    """Get the PDF signer singleton instance"""
    return VisualPDFSigner()
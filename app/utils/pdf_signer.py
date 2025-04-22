import os
import uuid
import datetime
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, blue, gray
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from PyPDF2 import PdfReader, PdfWriter


class VisualPDFSigner:
    """
    A simpler, more Docker-compatible PDF signer that adds visual signatures
    without requiring complex cryptographic libraries.
    """

    def __init__(self, logo_path=None, signature_path=None):
        """
        Initialize the visual PDF signer

        Args:
            logo_path: Path to the organization logo
            signature_path: Path to the signature image
        """
        self.logo_path = logo_path or "static/lexanalytics_logo.png"
        self.signature_path = signature_path or "static/signatures/signature.png"

    def sign_pdf(self, pdf_data, reason="Document verification", location="Digital"):
        """
        Add a visual signature to the PDF

        Args:
            pdf_data: PDF document data as bytes
            reason: The reason for signing
            location: The location of signing

        Returns:
            PDF data as bytes with visual signature
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

            return output.getvalue()

        except Exception as e:
            # Log the error but return the original PDF
            print(f"Error adding visual signature to PDF: {str(e)}")
            return pdf_data

    def _add_visual_signature(self, page, reason, location):
        """
        Add a visual signature box to the page

        Args:
            page: The PDF page to add the signature to
            reason: The reason for the signature
            location: The location of the signature

        Returns:
            The modified PDF page
        """
        try:
            # Create a new PDF with just the signature
            signature_pdf = BytesIO()
            c = canvas.Canvas(signature_pdf, pagesize=A4)

            # Get page size
            page_width, page_height = A4

            # Draw signature box at bottom of page
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
            print(f"Error adding visual signature elements: {str(e)}")
            return page
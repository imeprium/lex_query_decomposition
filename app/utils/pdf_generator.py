import os
import uuid
import datetime
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional

# ReportLab imports
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, Flowable
)
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import logging

logger = logging.getLogger("pdf_generator")


# Custom canvas class to handle page numbers properly
class NumberedCanvas(Canvas):
    """
    Custom canvas that adds page numbers to each page
    """

    def __init__(self, *args, **kwargs):
        Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """Add page info to each page (page x of y)"""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            Canvas.showPage(self)
        Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.drawCentredString(
            self._pagesize[0] / 2,
            20,  # 20 points from the bottom of the page
            f"Page {self._pageNumber} of {page_count}"
        )


class Watermark(Flowable):
    """
    Custom Flowable for creating a watermark
    """

    def __init__(self, text="CONFIDENTIAL", angle=45, fontsize=100, color=colors.lightgrey):
        Flowable.__init__(self)
        self.text = text
        self.angle = angle
        self.fontsize = fontsize
        self.color = color

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.translate(A4[0] / 2, A4[1] / 2)
        canvas.rotate(self.angle)
        canvas.setFont("Helvetica-Bold", self.fontsize)
        canvas.setFillColor(self.color)
        canvas.setStrokeColor(self.color)
        canvas.drawCentredString(0, 0, self.text)
        canvas.restoreState()


class PDFGenerator:
    """
    PDF Generator using ReportLab
    """

    def __init__(self):
        # Define styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()

        # Ensure needed directories exist
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        signatures_dir = Path("static/signatures")
        signatures_dir.mkdir(exist_ok=True, parents=True)

        # Paths
        self.logo_path = "static/lexanalytics_logo.png"
        self.signature_path = "static/signatures/signature.png"

    def _setup_styles(self):
        """Setup custom paragraph styles by extending existing ones"""
        # Modify existing styles instead of adding new ones with the same name
        self.styles['Title'].fontSize = 18
        self.styles['Title'].spaceAfter = 12
        self.styles['Title'].alignment = TA_LEFT

        self.styles['Heading2'].fontSize = 14
        self.styles['Heading2'].spaceAfter = 10
        self.styles['Heading2'].spaceBefore = 15

        self.styles['Heading3'].fontSize = 12
        self.styles['Heading3'].spaceAfter = 8
        self.styles['Heading3'].spaceBefore = 10

        self.styles['Normal'].fontSize = 11
        self.styles['Normal'].spaceAfter = 6
        self.styles['Normal'].alignment = TA_JUSTIFY

        # Create new styles with unique names for our custom needs
        self.styles.add(ParagraphStyle(
            name='BodyText-Justified',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))

        self.styles.add(ParagraphStyle(
            name='SourceItem',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=3
        ))

        self.styles.add(ParagraphStyle(
            name='SignatureInfo',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.gray
        ))

    def _create_header_with_logo(self, canvas, doc):
        """
        Add header with logo to each page
        """
        canvas.saveState()

        # Header with logo
        if os.path.exists(self.logo_path):
            try:
                # Draw logo at the top of the page
                canvas.drawImage(
                    self.logo_path,
                    doc.leftMargin,
                    doc.height + doc.topMargin - 1.5 * cm,
                    width=4 * cm,
                    height=1.5 * cm,
                    preserveAspectRatio=True
                )
            except Exception as e:
                logger.error(f"Error drawing logo: {str(e)}")

        canvas.restoreState()

    def generate_pdf(
            self,
            question: str,
            decomposed_questions: List[Dict[str, str]],
            final_answer: str,
            document_metadata: List[Dict[str, Any]],
            include_watermark: bool = True
    ) -> bytes:
        """
        Generate a PDF with legal analysis

        Args:
            question: The original question
            decomposed_questions: List of dicts with 'question' and 'answer' keys
            final_answer: The final synthesized answer
            document_metadata: List of document metadata dicts
            include_watermark: Whether to include watermark

        Returns:
            PDF as bytes
        """
        # Create buffer for the PDF
        buffer = BytesIO()

        try:
            # Create document with debug info
            logger.debug(f"Creating SimpleDocTemplate with question: {question[:50]}")
            pdf_doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                leftMargin=2 * cm,
                rightMargin=2 * cm,
                topMargin=2.5 * cm,
                bottomMargin=2 * cm,
                title=f"Legal Analysis - {question[:50]}"
            )

            # Verify document is created correctly
            logger.debug(f"Document created: {type(pdf_doc)}")

            # List of elements to build the PDF
            elements = []

            # Add watermark flowable if requested (will be placed on each page)
            if include_watermark:
                elements.append(Watermark(
                    text="LEXANALYTICS",
                    angle=-45,
                    fontsize=100,
                    color=colors.Color(0.9, 0.9, 0.9, alpha=0.2)
                ))

            # Title
            elements.append(Paragraph(question, self.styles['Title']))
            elements.append(Spacer(1, 0.5 * cm))

            # Key Legal Questions
            elements.append(Paragraph("Key Legal Questions", self.styles['Heading2']))
            elements.append(Spacer(1, 0.3 * cm))

            # Add each question and answer
            for idx, q_a in enumerate(decomposed_questions, 1):
                question_text = q_a.get('question', '')
                answer_text = q_a.get('answer', 'No answer available')

                # Format question
                elements.append(Paragraph(f"Q{idx}: {question_text}", self.styles['Heading3']))

                # Format answer - handle lists and paragraphs
                paragraphs = answer_text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        if para.strip().startswith('•') or para.strip().startswith('-'):
                            # Handle bullet lists
                            items = []
                            for line in para.strip().split('\n'):
                                if line.strip().startswith('•') or line.strip().startswith('-'):
                                    line = line.strip()[1:].strip()
                                    items.append(ListItem(Paragraph(line, self.styles['Normal'])))
                            elements.append(ListFlowable(items, bulletType='bullet', leftIndent=20))
                        else:
                            elements.append(Paragraph(para, self.styles['BodyText-Justified']))

                # Add source references if they exist in the answer
                if 'document' in answer_text.lower() or 'case:' in answer_text.lower():
                    # Extract source lines
                    for line in answer_text.split('\n'):
                        if ('document' in line.lower() or 'case:' in line.lower()) and len(line) < 100:
                            elements.append(Paragraph(line, self.styles['SourceItem']))

                elements.append(Spacer(1, 0.3 * cm))

            # Final Answer
            elements.append(Paragraph("Final Answer", self.styles['Heading2']))
            elements.append(Spacer(1, 0.3 * cm))

            # Process the final answer - detect sections
            sections = final_answer.split('\n\n')
            current_section = None

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # Check if this is a section header
                if (section.endswith(':') and len(section.split()) <= 5) or (
                        section and all(c.isalpha() or c.isspace() for c in section) and len(section) < 30):
                    # This is a header
                    header_text = section.rstrip(':')
                    elements.append(Paragraph(header_text, self.styles['Heading3']))
                    current_section = header_text
                else:
                    # This is content
                    paragraphs = section.split('\n')
                    for para in paragraphs:
                        if para.strip():
                            elements.append(Paragraph(para, self.styles['BodyText-Justified']))

            # Sources
            if document_metadata:
                elements.append(Paragraph("Sources", self.styles['Heading3']))

                # Group sources by title for uniqueness
                unique_sources = {}
                for doc in document_metadata:
                    source_key = None
                    for field in ["case_title", "article_title", "legislation_title"]:
                        if field in doc:
                            source_key = (field, doc[field])
                            break

                    if source_key and (
                            source_key not in unique_sources or doc.get("score", 0) >
                            unique_sources[source_key].get("score", 0)):
                        unique_sources[source_key] = doc

                # Format sources
                for (field_type, title), doc in unique_sources.items():
                    type_name = field_type.replace("_title", "").capitalize()
                    source_text = f"• <b>{type_name}:</b> {title} (Document ID: {doc.get('document_id', 'Unknown')})"
                    elements.append(Paragraph(source_text, self.styles['SourceItem']))

            # Add signature section
            elements.append(Spacer(1, 1 * cm))
            elements.append(Paragraph("_" * 50, self.styles['Normal']))

            # Generate unique document ID and timestamp
            document_id = str(uuid.uuid4())
            generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            elements.append(Paragraph(
                f"This document was digitally generated on {generation_date} by Lexanalytics AI.",
                self.styles['SignatureInfo']
            ))
            elements.append(Paragraph(f"Document ID: {document_id}", self.styles['SignatureInfo']))

            # Add signature image if it exists
            if os.path.exists(self.signature_path):
                signature = Image(self.signature_path, width=3 * cm, height=1 * cm)
                elements.append(signature)

            # Verify elements are correct
            logger.debug(f"Number of elements to build: {len(elements)}")

            # Build the PDF with our custom canvas class for page numbering
            # and the header function for the logo
            logger.debug("Building PDF document with NumberedCanvas...")
            pdf_doc.build(
                elements,
                onFirstPage=self._create_header_with_logo,
                onLaterPages=self._create_header_with_logo,
                canvasmaker=NumberedCanvas
            )
            logger.debug("PDF document built successfully")

            # Return PDF as bytes
            pdf_bytes = buffer.getvalue()
            return pdf_bytes

        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
            raise
        finally:
            buffer.close()
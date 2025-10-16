"""
PDF generation utilities using ReportLab for creating legal analysis documents.
"""
import os
import uuid
import datetime
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any, Optional
from functools import lru_cache

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


class NumberedCanvas(Canvas):
    """
    Custom canvas that adds page numbers to each page.
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
    Custom Flowable for creating a watermark.
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
    PDF Generator using ReportLab for creating legal analysis documents.
    """

    def __init__(self):
        # Set up styles
        self.styles = getSampleStyleSheet()
        self._setup_styles()

        # Ensure needed directories exist
        self._ensure_directories()

        # Set paths
        self.logo_path = self._get_logo_path()
        self.signature_path = self._get_signature_path()

    def _ensure_directories(self):
        """Ensure required directories exist"""
        for directory in ["static", "static/signatures"]:
            Path(directory).mkdir(exist_ok=True, parents=True)

    def _get_logo_path(self):
        """Get path to logo file, with fallback if not exists"""
        path = "static/lexanalytics_logo.png"
        if not os.path.exists(path):
            logger.warning(f"Logo file not found at {path}")
        return path

    def _get_signature_path(self):
        """Get path to signature file"""
        return "static/signatures/signature.png"

    def _setup_styles(self):
        """Set up custom paragraph styles"""
        # Update existing styles
        self._update_existing_styles()

        # Add custom styles
        self._add_custom_styles()

    def _update_existing_styles(self):
        """Update existing styles with custom attributes"""
        style_updates = {
            'Title': {'fontSize': 18, 'spaceAfter': 12, 'alignment': TA_LEFT},
            'Heading2': {'fontSize': 14, 'spaceAfter': 10, 'spaceBefore': 15},
            'Heading3': {'fontSize': 12, 'spaceAfter': 8, 'spaceBefore': 10},
            'Normal': {'fontSize': 11, 'spaceAfter': 6, 'alignment': TA_JUSTIFY}
        }

        for style_name, attributes in style_updates.items():
            for attr, value in attributes.items():
                setattr(self.styles[style_name], attr, value)

    def _add_custom_styles(self):
        """Add custom styles to the stylesheet"""
        custom_styles = [
            ('BodyText-Justified', self.styles['Normal'], {
                'fontSize': 11,
                'spaceAfter': 6,
                'alignment': TA_JUSTIFY
            }),
            ('SourceItem', self.styles['Normal'], {
                'fontSize': 10,
                'leftIndent': 20,
                'spaceAfter': 3
            }),
            ('SignatureInfo', self.styles['Normal'], {
                'fontSize': 9,
                'textColor': colors.gray
            })
        ]

        for name, parent, attributes in custom_styles:
            style = ParagraphStyle(name=name, parent=parent)
            for attr, value in attributes.items():
                setattr(style, attr, value)
            self.styles.add(style)

    def _create_header_with_logo(self, canvas, doc):
        """Add header with logo to each page"""
        canvas.saveState()

        # Draw logo at the top of the page if it exists
        if os.path.exists(self.logo_path):
            try:
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
        Generate a PDF document with legal analysis

        Args:
            question: Original question
            decomposed_questions: List of dicts with 'question' and 'answer' keys
            final_answer: Final synthesized answer
            document_metadata: List of document metadata dicts
            include_watermark: Whether to include watermark

        Returns:
            PDF as bytes
        """
        buffer = BytesIO()

        try:
            # Create document
            logger.debug(f"Creating PDF for question: {question[:50]}")
            pdf_doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                leftMargin=2 * cm,
                rightMargin=2 * cm,
                topMargin=2.5 * cm,
                bottomMargin=2 * cm,
                title=f"Legal Analysis - {question[:50]}"
            )

            # List of elements to build the PDF
            elements = []

            # Add watermark if requested
            if include_watermark:
                elements.append(Watermark(
                    text="LEXANALYTICS",
                    angle=-45,
                    fontsize=100,
                    color=colors.Color(0.9, 0.9, 0.9, alpha=0.2)
                ))

            # Add content sections
            self._add_title_section(elements, question)
            self._add_questions_section(elements, decomposed_questions)
            self._add_answer_section(elements, final_answer)
            self._add_sources_section(elements, document_metadata)
            self._add_signature_section(elements)

            # Build the PDF with custom canvas for page numbering
            logger.debug("Building PDF document")
            pdf_doc.build(
                elements,
                onFirstPage=self._create_header_with_logo,
                onLaterPages=self._create_header_with_logo,
                canvasmaker=NumberedCanvas
            )

            # Return PDF as bytes
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
            raise
        finally:
            buffer.close()

    def _add_title_section(self, elements, question):
        """Add title section to PDF"""
        elements.append(Paragraph(question, self.styles['Title']))
        elements.append(Spacer(1, 0.5 * cm))

    def _add_questions_section(self, elements, decomposed_questions):
        """Add key legal questions section to PDF"""
        elements.append(Paragraph("Key Legal Questions", self.styles['Heading2']))
        elements.append(Spacer(1, 0.3 * cm))

        for idx, q_a in enumerate(decomposed_questions, 1):
            question_text = q_a.get('question', '')
            answer_text = q_a.get('answer', 'No answer available')

            # Format question
            elements.append(Paragraph(f"Q{idx}: {question_text}", self.styles['Heading3']))

            # Format answer
            self._add_formatted_answer(elements, answer_text)

            elements.append(Spacer(1, 0.3 * cm))

    def _add_answer_section(self, elements, final_answer):
        """Add final answer section to PDF"""
        elements.append(Paragraph("Final Answer", self.styles['Heading2']))
        elements.append(Spacer(1, 0.3 * cm))

        # Process sections
        sections = self._detect_sections(final_answer)

        if sections:
            # Add each detected section
            for header, content in sections:
                elements.append(Paragraph(header, self.styles['Heading3']))
                for para in content.split('\n'):
                    if para.strip():
                        elements.append(Paragraph(para, self.styles['BodyText-Justified']))
        else:
            # No sections detected, format as regular paragraphs
            for para in final_answer.split('\n\n'):
                if para.strip():
                    elements.append(Paragraph(para, self.styles['BodyText-Justified']))

    def _add_sources_section(self, elements, document_metadata):
        """Add sources section to PDF"""
        if not document_metadata:
            return

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
                    source_key not in unique_sources or
                    doc.get("score", 0) > unique_sources[source_key].get("score", 0)):
                unique_sources[source_key] = doc

        # Format sources
        for (field_type, title), doc in unique_sources.items():
            type_name = field_type.replace("_title", "").capitalize()
            source_text = f"• <b>{type_name}:</b> {title} (Document ID: {doc.get('document_id', 'Unknown')})"
            elements.append(Paragraph(source_text, self.styles['SourceItem']))

    def _add_signature_section(self, elements):
        """Add signature section to PDF"""
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

    def _add_formatted_answer(self, elements, answer_text):
        """Add formatted answer text to PDF elements"""
        # Split answer into paragraphs
        paragraphs = answer_text.split('\n\n')

        for para in paragraphs:
            if not para.strip():
                continue

            if para.strip().startswith('•') or para.strip().startswith('-'):
                # Handle bullet lists
                items = []
                for line in para.strip().split('\n'):
                    if line.strip().startswith('•') or line.strip().startswith('-'):
                        line = line.strip()[1:].strip()
                        items.append(ListItem(Paragraph(line, self.styles['Normal'])))

                if items:
                    elements.append(ListFlowable(items, bulletType='bullet', leftIndent=20))
            else:
                elements.append(Paragraph(para, self.styles['BodyText-Justified']))

        # Add source references if they exist in the answer
        if 'document' in answer_text.lower() or 'case:' in answer_text.lower():
            for line in answer_text.split('\n'):
                if ('document' in line.lower() or 'case:' in line.lower()) and len(line) < 100:
                    elements.append(Paragraph(line, self.styles['SourceItem']))

    def _detect_sections(self, text):
        """Detect sections in text based on headings"""
        sections = []
        current_section = None
        current_content = []

        for line in text.split('\n\n'):
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header
            if (line.endswith(':') and len(line.split()) <= 5) or (
                    line and all(c.isalpha() or c.isspace() for c in line) and len(line) < 30):
                # This is a header - save previous section and start new one
                if current_section:
                    sections.append((current_section, '\n'.join(current_content)))

                current_section = line.rstrip(':')
                current_content = []
            else:
                # This is content - add to current section
                current_content.append(line)

        # Add the last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections


# Factory function
@lru_cache(maxsize=1)
def get_pdf_generator():
    """Get the PDF generator singleton instance"""
    return PDFGenerator()
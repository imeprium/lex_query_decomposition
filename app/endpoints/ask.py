import os

from fastapi import APIRouter, Query, HTTPException, Depends, Request
import logging
import time
from fastapi.responses import Response
from app.pipelines.legal_decomposition_pipeline import process_question
from app.models import LegalQueryResponse, DocumentMetadata, Question
from app.utils.formatter import format_as_markdown
from app.utils.sanitizer import sanitize_legal_query
from app.utils.pdf_generator import PDFGenerator
from app.utils.pdf_signer import VisualPDFSigner
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api")
logger = logging.getLogger("api")

# Create PDF generator and signer instances
pdf_generator = PDFGenerator()
pdf_signer = VisualPDFSigner()


@router.get("/ask", response_model=LegalQueryResponse)
async def ask_legal_question(
        sanitized_question: str = Depends(sanitize_legal_query),
        format: str = Query("json", description="Response format: 'json' or 'markdown'")
):
    """
    Process a legal question through the decomposition pipeline

    Args:
        sanitized_question: The sanitized legal question (validated by dependency)
        format: Response format (json or markdown)

    Returns:
        LegalQueryResponse containing the original question, decomposed questions, and final answer
    """
    start_time = time.time()
    logger.info(f"Processing sanitized legal query: {sanitized_question[:100]}...")

    try:
        # Process the question asynchronously using the pipeline
        result = await process_question(sanitized_question)

        # Create response data
        response_data = {
            "original_question": sanitized_question,
            "decomposed_questions": result.get("sub_questions", {}).questions if hasattr(
                result.get("sub_questions", {}), "questions") else [],
            "final_answer": result.get("answer", ""),
            "document_metadata": result.get("document_metadata", [])
        }

        # Return either JSON or markdown based on format parameter
        if format.lower() == "markdown":
            # Create a serializable dictionary from the decomposed_questions objects
            # This prevents the AttributeError with 'Question' object has no attribute 'get'
            serialized_data = {
                "original_question": response_data["original_question"],
                "decomposed_questions": [
                    {"question": q.question, "answer": q.answer}
                    for q in response_data["decomposed_questions"]
                ],
                "final_answer": response_data["final_answer"],
                "document_metadata": response_data["document_metadata"]
            }
            markdown_content = format_as_markdown(serialized_data)
            return Response(content=markdown_content, media_type="text/markdown")
        else:
            return response_data

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)

        # Return error response
        error_response = {
            "original_question": sanitized_question,
            "decomposed_questions": [],
            "final_answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
            "document_metadata": []
        }

        if format.lower() == "markdown":
            markdown_error = format_as_markdown(error_response)
            return Response(content=markdown_error, media_type="text/markdown")
        else:
            return error_response


@router.get("/ask/pdf")
async def download_legal_analysis_pdf(
        request: Request,
        question: str = Query(..., description="The legal question to research"),
        include_watermark: bool = Query(True, description="Include watermark in the PDF"),
        sign_document: bool = Query(False, description="Add visual signature to the document"),
        signature_reason: str = Query("Legal Analysis Document", description="Reason for signature"),
        signature_location: str = Query("Digital", description="Location of signing")
):
    """
    Process a legal question and return the analysis as a PDF document.

    Args:
        request: The FastAPI request object
        question: The legal question to research
        include_watermark: Whether to include a watermark in the PDF
        sign_document: Whether to add a visual signature to the document
        signature_reason: Reason for the signature
        signature_location: Location of signing

    Returns:
        PDF document response
    """
    start_time = time.time()
    logger.info(f"Processing PDF generation for query: {question[:100]}...")

    try:
        # First sanitize the question
        sanitized_question = sanitize_legal_query(question)

        # Process the question asynchronously using the pipeline
        result = await process_question(sanitized_question)

        # Prepare data for PDF generation
        decomposed_questions = []
        if hasattr(result.get("sub_questions", {}), "questions"):
            for q in result.get("sub_questions", {}).questions:
                decomposed_questions.append({
                    "question": q.question,
                    "answer": q.answer if q.answer else "No answer available"
                })

        # Generate PDF
        pdf_bytes = pdf_generator.generate_pdf(
            question=sanitized_question,
            decomposed_questions=decomposed_questions,
            final_answer=result.get("answer", ""),
            document_metadata=result.get("document_metadata", []),
            include_watermark=include_watermark
        )

        # Add visual signature if requested
        if sign_document:
            try:
                # Get the visual signer
                if not hasattr(request.app.state, "pdf_signer"):
                    # Create signer if it doesn't exist
                    from app.utils.pdf_signer import VisualPDFSigner
                    request.app.state.pdf_signer = VisualPDFSigner()

                # Sign the PDF
                pdf_signer = request.app.state.pdf_signer
                signed_pdf = pdf_signer.sign_pdf(
                    pdf_bytes,
                    reason=signature_reason,
                    location=signature_location
                )

                # Use the signed PDF if successful
                if signed_pdf:
                    logger.info("Document visually signed successfully")
                    pdf_bytes = signed_pdf
                else:
                    logger.warning("Document could not be signed, using unsigned version")
            except Exception as e:
                logger.error(f"Error during document signing: {str(e)}")
                # Continue with unsigned document

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"PDF generation completed in {processing_time:.2f} seconds")

        # Set filename based on query
        safe_filename = sanitized_question[:30].replace(" ", "_").lower()
        filename = f"legal_analysis_{safe_filename}.pdf"

        # Return PDF response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Document-Signed": "true" if sign_document else "false"
            }
        )

    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating PDF: {str(e)}"
        )
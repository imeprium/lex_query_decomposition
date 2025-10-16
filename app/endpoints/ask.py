import os
import logging
import time
from typing import Dict, Any, Optional

from fastapi import APIRouter, Query, HTTPException, Depends, Request
from fastapi.responses import Response

from app.pipelines.legal_decomposition_pipeline import process_question
from app.models import LegalQueryResponse, DocumentMetadata, Question
from app.utils.formatter import format_as_markdown
from app.utils.sanitizer import sanitize_legal_query
from app.utils.pdf_generator import get_pdf_generator
from app.utils.pdf_signer import get_pdf_signer
from app.core.async_component import AsyncComponent
from app.auth import (
    require_legal_research_access,
    require_query_decomposition_access,
    require_pdf_generation_access,
    get_user_context
)

router = APIRouter(prefix="/api")
logger = logging.getLogger("api")


class QueryProcessor(AsyncComponent):
    """
    Handles processing of legal queries and formatting of responses.
    Encapsulates the query processing logic to separate it from API routing.
    """

    async def process_query(
            self,
            question: str,
            format: str = "json"
    ) -> Dict[str, Any]:
        """
        Process a legal question through the pipeline

        Args:
            question: The sanitized query
            format: Response format (json or markdown)

        Returns:
            Processed response data
        """
        start_time = time.time()
        logger.info(f"Processing legal query: {question[:100]}...")

        try:
            # Process question through the pipeline
            result = await process_question(question)

            # Format response based on requested format
            response_data = self._prepare_response_data(question, result)

            if format.lower() == "markdown":
                # Create markdown content
                markdown_content = format_as_markdown(self._serialize_response_data(response_data))
                logger.info(f"Query processed in {time.time() - start_time:.2f}s, returning markdown")
                return {"content": markdown_content, "media_type": "text/markdown"}
            else:
                logger.info(f"Query processed in {time.time() - start_time:.2f}s, returning JSON")
                return {"content": response_data, "media_type": "application/json"}

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)

            # Create error response
            error_response = {
                "original_question": question,
                "decomposed_questions": [],
                "final_answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "document_metadata": []
            }

            if format.lower() == "markdown":
                markdown_error = format_as_markdown(error_response)
                return {"content": markdown_error, "media_type": "text/markdown"}
            else:
                return {"content": error_response, "media_type": "application/json"}

    async def generate_pdf(
            self,
            question: str,
            include_watermark: bool = True,
            sign_document: bool = False,
            signature_reason: str = "Legal Analysis Document",
            signature_location: str = "Digital"
    ) -> Dict[str, Any]:
        """
        Process a legal question and generate a PDF document

        Args:
            question: The legal question
            include_watermark: Whether to include watermark
            sign_document: Whether to sign the document
            signature_reason: Reason for signature
            signature_location: Location of signing

        Returns:
            Dictionary with PDF content and headers
        """
        start_time = time.time()
        logger.info(f"Processing PDF generation for query: {question[:100]}...")

        try:
            # Process the question asynchronously
            result = await process_question(question)

            # Prepare decomposed questions data
            decomposed_questions = self._prepare_decomposed_questions(result)

            # Get PDF generator
            pdf_generator = get_pdf_generator()

            # Generate PDF
            pdf_bytes = pdf_generator.generate_pdf(
                question=question,
                decomposed_questions=decomposed_questions,
                final_answer=result.get("answer", ""),
                document_metadata=result.get("document_metadata", []),
                include_watermark=include_watermark
            )

            # Add visual signature if requested
            if sign_document:
                pdf_signer = get_pdf_signer()
                try:
                    signed_pdf = pdf_signer.sign_pdf(
                        pdf_bytes,
                        reason=signature_reason,
                        location=signature_location
                    )

                    # Use signed PDF if successful
                    if signed_pdf:
                        logger.info("Document signed successfully")
                        pdf_bytes = signed_pdf
                    else:
                        logger.warning("Document could not be signed, using unsigned version")
                except Exception as e:
                    logger.error(f"Error during document signing: {str(e)}")

            # Create safe filename based on query
            safe_filename = question[:30].replace(" ", "_").lower()
            filename = f"legal_analysis_{safe_filename}.pdf"

            logger.info(f"PDF generation completed in {time.time() - start_time:.2f}s")

            # Return PDF data and headers
            return {
                "content": pdf_bytes,
                "media_type": "application/pdf",
                "headers": {
                    "Content-Disposition": f"attachment; filename={filename}",
                    "X-Document-Signed": "true" if sign_document else "false"
                }
            }

        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating PDF: {str(e)}"
            )

    def _prepare_response_data(self, question: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare response data from pipeline result

        Args:
            question: Original question
            result: Pipeline result

        Returns:
            Structured response data
        """
        # Get sub_questions safely
        sub_questions_obj = result.get("sub_questions")

        # Extract questions list appropriately based on the object type
        if sub_questions_obj:
            if hasattr(sub_questions_obj, "questions"):
                # It's already a Pydantic model
                decomposed_questions = sub_questions_obj.questions
            elif isinstance(sub_questions_obj, dict) and "questions" in sub_questions_obj:
                # It's a dictionary representation (should now be handled by cache reconstruction)
                # This is a fallback in case reconstruction failed
                try:
                    from app.models import Question
                    questions_data = sub_questions_obj["questions"]
                    decomposed_questions = []
                    for q_data in questions_data:
                        if isinstance(q_data, dict):
                            decomposed_questions.append(
                                Question(
                                    question=q_data.get("question", ""),
                                    answer=q_data.get("answer")
                                )
                            )
                        else:
                            # Already a Question object
                            decomposed_questions.append(q_data)
                except Exception:
                    logger.warning("Failed to convert questions dict to objects", exc_info=True)
                    decomposed_questions = []
            else:
                decomposed_questions = []
        else:
            decomposed_questions = []

        return {
            "original_question": question,
            "decomposed_questions": decomposed_questions,
            "final_answer": result.get("answer", ""),
            "document_metadata": result.get("document_metadata", [])
        }


    def _serialize_response_data(self,  response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Questions objects to serializable dictionaries

        Args:
            response_data: Response data with possible Pydantic objects

        Returns:
            Serializable dictionary
        """
        return {
            "original_question": response_data["original_question"],
            "decomposed_questions": [
                {"question": q.question, "answer": q.answer}
                for q in response_data["decomposed_questions"]
            ],
            "final_answer": response_data["final_answer"],
            "document_metadata": response_data["document_metadata"]
        }

    def _prepare_decomposed_questions(self, result: Dict[str, Any]) -> list:
        """
        Convert decomposed questions to list of dictionaries

        Args:
            result: Pipeline result

        Returns:
            List of question dictionaries
        """
        decomposed_questions = []

        # Get sub_questions safely
        sub_questions_obj = result.get("sub_questions")

        # Extract questions only if it's a proper object with questions attribute
        if sub_questions_obj and hasattr(sub_questions_obj, "questions"):
            for q in sub_questions_obj.questions:
                decomposed_questions.append({
                    "question": q.question,
                    "answer": q.answer if q.answer else "No answer available"
                })

        return decomposed_questions


# Create singleton processor instance
query_processor = QueryProcessor()


@router.get("/ask", response_model=LegalQueryResponse)
async def ask_legal_question(
        sanitized_question: str = Depends(sanitize_legal_query),
        format: str = Query("json", description="Response format: 'json' or 'markdown'"),
        enable_followup: bool = Query(False, description="Enable chat follow-up support"),
        user_context: dict = Depends(get_user_context),
        _: None = Depends(require_legal_research_access),
        __: None = Depends(require_query_decomposition_access)
):
    """
    Process a legal question through the decomposition pipeline

    Args:
        sanitized_question: The sanitized legal question
        format: Response format (json or markdown)
        enable_followup: Whether to enable chat follow-up support

    Returns:
        Processed query response
    """
    if enable_followup:
        # Use enhanced pipeline with chat support
        from app.services.enhanced_pipeline_service import get_enhanced_pipeline_service
        enhanced_service = get_enhanced_pipeline_service()

        response = await enhanced_service.process_with_chat_support(
            question=sanitized_question,
            enable_followup=enable_followup
        )

        # Convert to original response format for backward compatibility
        if format.lower() == "markdown":
            markdown_content = format_as_markdown({
                "original_question": response.original_question,
                "decomposed_questions": [
                    {"question": q.question, "answer": q.answer}
                    for q in response.decomposed_questions
                ],
                "final_answer": response.final_answer,
                "document_metadata": response.document_metadata
            })

            # Add chat information if followup is enabled
            if response.supports_followup and response.conversation_id:
                markdown_content += f"\n\n---\n**Follow-up Questions Enabled**\nConversation ID: `{response.conversation_id}`\nUse: `POST /api/chat/followup/{response.conversation_id}`"

            return Response(
                content=markdown_content,
                media_type="text/markdown"
            )
        else:
            return response
    else:
        # Use original pipeline
        result = await query_processor.process_query(sanitized_question, format)

        if format.lower() == "markdown":
            # For markdown, return as text response
            return Response(
                content=result["content"],
                media_type=result["media_type"]
            )
        else:
            # For JSON, directly return the content dict
            return result["content"]


@router.get("/ask/pdf")
async def download_legal_analysis_pdf(
        question: str = Query(..., description="The legal question to research"),
        include_watermark: bool = Query(True, description="Include watermark in the PDF"),
        sign_document: bool = Query(False, description="Add visual signature to the document"),
        signature_reason: str = Query("Legal Analysis Document", description="Reason for signature"),
        signature_location: str = Query("Digital", description="Location of signing"),
        user_context: dict = Depends(get_user_context),
        _: None = Depends(require_legal_research_access),
        __: None = Depends(require_pdf_generation_access)
):
    """
    Process a legal question and return the analysis as a PDF document

    Args:
        question: The legal question
        include_watermark: Whether to include watermark
        sign_document: Whether to sign the document
        signature_reason: Reason for signature
        signature_location: Location of signing

    Returns:
        PDF document response
    """
    # First sanitize the question
    sanitized_question = sanitize_legal_query(question)

    # Generate PDF
    result = await query_processor.generate_pdf(
        sanitized_question,
        include_watermark,
        sign_document,
        signature_reason,
        signature_location
    )

    return Response(
        content=result["content"],
        media_type=result["media_type"],
        headers=result["headers"]
    )
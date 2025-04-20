from fastapi import APIRouter, Query, HTTPException, Depends, Request
import logging
import time
from fastapi.responses import JSONResponse, Response
from app.pipelines.legal_decomposition_pipeline import process_question
from app.models import LegalQueryResponse, DocumentMetadata, Question
from app.utils.formatter import format_as_markdown
from app.utils.sanitizer import sanitize_legal_query
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api")
logger = logging.getLogger("api")


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
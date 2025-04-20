import logging
import asyncio
from typing import Optional
from pydantic import BaseModel
from haystack import component
from app.models import Questions, Question

logger = logging.getLogger("pipeline")


@component
class DecompositionValidator:
    """
    Component that validates decomposed questions and provides fallback when decomposition fails.
    This addresses the issue where query decomposition might return empty questions.
    """

    @component.output_types(valid_questions=Questions)
    def run(self, questions: Optional[BaseModel] = None, original_question: str = ""):
        """
        Check if questions are valid and provide fallback if decomposition failed

        Args:
            questions: The Questions object from the decomposer (as BaseModel)
            original_question: The original question string as backup

        Returns:
            Dictionary with valid_questions containing either the original
            questions or a fallback question
        """
        # First convert the input to a Questions object if possible
        questions_obj = None

        # Check if input is a valid Questions object or can be converted to one
        if questions is not None:
            if isinstance(questions, Questions):
                questions_obj = questions
            else:
                # Try to convert from a generic BaseModel
                try:
                    # If it's a BaseModel with a 'questions' attribute
                    if hasattr(questions, 'questions') and isinstance(questions.questions, list):
                        questions_obj = Questions(questions=questions.questions)
                    else:
                        logger.warning(f"Could not convert input to Questions: {questions}")
                except Exception as e:
                    logger.error(f"Error converting to Questions: {str(e)}")

        # Check if we have valid questions
        if questions_obj is None or not questions_obj.questions or len(questions_obj.questions) == 0:
            logger.warning(f"Decomposition failed, using original question as fallback: {original_question}")
            # Create a fallback question using the original query
            fallback_questions = Questions(questions=[
                Question(question=original_question, answer=None)
            ])
            return {"valid_questions": fallback_questions}

        # Log valid decomposition
        logger.info(f"Valid decomposition with {len(questions_obj.questions)} sub-questions")
        for i, q in enumerate(questions_obj.questions):
            logger.debug(f"Sub-question {i + 1}: {q.question}")

        return {"valid_questions": questions_obj}

    @component.output_types(valid_questions=Questions)
    async def run_async(self, questions: Optional[BaseModel] = None, original_question: str = ""):
        """
        Asynchronous version: Check if questions are valid and provide fallback if decomposition failed

        Args:
            questions: The Questions object from the decomposer (as BaseModel)
            original_question: The original question string as backup

        Returns:
            Dictionary with valid_questions containing either the original
            questions or a fallback question
        """
        # Since validation is a CPU-bound task not requiring I/O, we can offload to a thread pool
        # This ensures we don't block the event loop for CPU-intensive operations
        return await asyncio.to_thread(self.run, questions, original_question)
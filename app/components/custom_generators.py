import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Type
from pydantic import BaseModel
from haystack.dataclasses import ChatMessage
from haystack import component
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereGenerator
from app.config.settings import COHERE_API_KEY, COHERE_MODEL
from app.models import Questions, Question
from app.core.async_component import AsyncComponent

logger = logging.getLogger("components")


@component
class ExtendedCohereGenerator(AsyncComponent):
    """
    Extended Cohere Generator with structured output support using Cohere's JSON schema functionality.
    Works with Command R and Command R+ models.
    """

    def __init__(
            self,
            model_type: Optional[Type[BaseModel]] = None,
            model: str = COHERE_MODEL,
            api_key: Optional[Union[str, Secret]] = None,
            **kwargs
    ):
        # Process API key
        if api_key is None and COHERE_API_KEY:
            api_key = Secret.from_token(COHERE_API_KEY)

        # Create the base generator
        try:
            self.base_generator = CohereGenerator(
                model=model,
                api_key=api_key,
                **kwargs
            )
            self.model_type = model_type
            self.model_name = model

            # Log initialization
            logger.info(f"Initialized ExtendedCohereGenerator with model {self.model_name}")
            if model_type:
                logger.info(f"Using structured output with schema: {model_type.__name__}")

        except Exception as e:
            logger.error(f"Error initializing CohereGenerator: {str(e)}")
            raise

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]], structured_reply=BaseModel)
    def run(self, prompt: str, **kwargs):
        """
        Run the extended generator with structured output support
        """
        try:
            # Handle structured output for compatible models
            if self._should_use_structured_output():
                return self._run_with_structured_output(prompt)
            else:
                # Standard operation without structured output
                result = self.base_generator.run(prompt)
                return {
                    "replies": result["replies"],
                    "meta": result["meta"],
                    "structured_reply": None
                }
        except Exception as e:
            logger.error(f"Error in generator run: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))

    def _run_with_structured_output(self, prompt: str):
        """
        Run generator with structured output support

        Args:
            prompt: The input prompt

        Returns:
            Dictionary with replies, meta info, and structured output
        """
        try:
            # Create structured prompt
            structured_prompt = self._create_structured_prompt(prompt)

            # Call the base generator
            result = self.base_generator.run(structured_prompt)
            replies = result["replies"]
            meta = result["meta"]

            # Parse the JSON response
            structured_output = self._parse_structured_output(replies[0])

            return {
                "replies": replies,
                "meta": meta,
                "structured_reply": structured_output
            }
        except Exception as e:
            logger.error(f"Error in structured output generation: {str(e)}")
            return self._create_error_response(str(e))

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]], structured_reply=BaseModel)
    async def run_async(self, prompt: str, **kwargs):
        """
        Asynchronous version of run method for better performance in async pipelines
        """
        try:
            # Handle structured output for compatible models
            if self._should_use_structured_output():
                # Check if base_generator has async support
                if hasattr(self.base_generator, 'run_async'):
                    structured_prompt = self._create_structured_prompt(prompt)
                    result = await self.base_generator.run_async(structured_prompt)
                else:
                    # Fall back to sync method in thread pool
                    return await self.to_thread(self._run_with_structured_output, prompt)

                replies = result["replies"]
                meta = result["meta"]
                structured_output = self._parse_structured_output(replies[0])

                return {
                    "replies": replies,
                    "meta": meta,
                    "structured_reply": structured_output
                }
            else:
                # Standard operation - run base generator async if available
                if hasattr(self.base_generator, 'run_async'):
                    result = await self.base_generator.run_async(prompt)
                else:
                    result = await self.to_thread(self.base_generator.run, prompt)

                return {
                    "replies": result["replies"],
                    "meta": result["meta"],
                    "structured_reply": None
                }
        except Exception as e:
            logger.error(f"Error in async generator run: {str(e)}", exc_info=True)
            return self._create_error_response(str(e))

    def _should_use_structured_output(self) -> bool:
        """Check if structured output should be used"""
        structured_models = [
            "command-r", "command-r-plus", "command-r-08-2024", "command-r+-08-2024"
        ]
        return (self.model_type is not None and
                self.model_name in structured_models)

    def _create_structured_prompt(self, prompt: str) -> str:
        """Create a prompt with structured output instructions"""
        schema = self.model_type.model_json_schema()
        return f"""
        {prompt}

        IMPORTANT: Please format your response as a valid JSON object following the schema:
        {json.dumps(schema, indent=2)}

        Return only the JSON object without any additional text.
        """

    def _parse_structured_output(self, text: str) -> BaseModel:
        """Parse JSON response into the provided Pydantic model"""
        try:
            # Try to find JSON in the text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1

            if 0 <= json_start < json_end:
                json_str = text[json_start:json_end]
                logger.debug(f"Extracted JSON: {json_str[:100]}...")
                parsed_data = json.loads(json_str)
                return self.model_type.model_validate(parsed_data)
            else:
                # Try parsing the whole text as JSON
                logger.warning("Could not find JSON delimiters, attempting to parse entire response")
                return self.model_type.model_validate_json(text)
        except Exception as e:
            logger.error(f"Failed to parse structured output: {str(e)}. Response: {text[:200]}...")
            return self._create_empty_model()

    def _create_empty_model(self) -> BaseModel:
        """Create an empty model instance based on model_type"""
        if self.model_type == Questions:
            return Questions(questions=[])
        else:
            return self.model_type()

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        if self.model_type == Questions:
            empty_model = Questions(questions=[])
        elif self.model_type:
            empty_model = self.model_type()
        else:
            empty_model = None

        return {
            "replies": [f"Error: {error_message}"],
            "meta": [{"error": error_message}],
            "structured_reply": empty_model
        }
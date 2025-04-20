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

logger = logging.getLogger("components")


@component
class ExtendedCohereGenerator:
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

        Args:
            prompt: The input prompt for the generator
            **kwargs: Additional keyword arguments (ignored for CohereGenerator compatibility)

        Returns:
            Dictionary with replies, meta info, and structured output
        """
        # If model_type is provided and we have a compatible Cohere model, set up structured output
        if self.model_type and self.model_name in ["command-r", "command-r-plus", "command-r-08-2024",
                                                   "command-r+-08-2024"]:
            logger.debug(f"Using structured output with model {self.model_name}")

            # Create schema for response format
            schema = self.model_type.model_json_schema()

            # Add instruction to return JSON
            structured_prompt = f"""
            {prompt}

            IMPORTANT: Please format your response as a valid JSON object following the schema:
            {json.dumps(schema, indent=2)}

            Return only the JSON object without any additional text.
            """

            # Call the Cohere API - NO generation_kwargs parameter
            try:
                # Call base generator's run method WITHOUT unsupported parameters
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
                # Return empty structured response with valid default values
                if self.model_type == Questions:
                    # For Questions model, initialize with empty questions list
                    empty_model = Questions(questions=[])
                else:
                    # For other models, use empty initialization
                    empty_model = self.model_type()

                return {
                    "replies": ["Error generating structured response"],
                    "meta": [{"error": str(e)}],
                    "structured_reply": empty_model
                }
        else:
            # Standard operation without structured output - NO generation_kwargs
            result = self.base_generator.run(prompt)

            return {
                "replies": result["replies"],
                "meta": result["meta"],
                "structured_reply": None
            }

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]], structured_reply=BaseModel)
    async def run_async(self, prompt: str, **kwargs):
        """
        Asynchronous version of run method for better performance in async pipelines

        Args:
            prompt: The input prompt for the generator
            **kwargs: Additional keyword arguments (ignored for CohereGenerator compatibility)

        Returns:
            Dictionary with replies, meta info, and structured output
        """
        # If model_type is provided and we have a compatible Cohere model, set up structured output
        if self.model_type and self.model_name in ["command-r", "command-r-plus", "command-r-08-2024",
                                                   "command-r+-08-2024"]:
            logger.debug(f"Using structured output with model {self.model_name} in async mode")

            # Create schema for response format
            schema = self.model_type.model_json_schema()

            # Add instruction to return JSON
            structured_prompt = f"""
            {prompt}

            IMPORTANT: Please format your response as a valid JSON object following the schema:
            {json.dumps(schema, indent=2)}

            Return only the JSON object without any additional text.
            """

            # Call the Cohere API asynchronously if possible
            try:
                # Check if base_generator has async support
                if hasattr(self.base_generator, 'run_async'):
                    # Use async method if available
                    result = await self.base_generator.run_async(structured_prompt)
                else:
                    # Fall back to running sync method in a thread pool
                    result = await asyncio.to_thread(self.base_generator.run, structured_prompt)

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
                logger.error(f"Error in async structured output generation: {str(e)}")
                # Return empty structured response with valid default values
                if self.model_type == Questions:
                    # For Questions model, initialize with empty questions list
                    empty_model = Questions(questions=[])
                else:
                    # For other models, use empty initialization
                    empty_model = self.model_type()

                return {
                    "replies": ["Error generating structured response"],
                    "meta": [{"error": str(e)}],
                    "structured_reply": empty_model
                }
        else:
            # Standard operation without structured output
            try:
                # Check if base_generator has async support
                if hasattr(self.base_generator, 'run_async'):
                    # Use async method if available
                    result = await self.base_generator.run_async(prompt)
                else:
                    # Fall back to running sync method in a thread pool
                    result = await asyncio.to_thread(self.base_generator.run, prompt)

                return {
                    "replies": result["replies"],
                    "meta": result["meta"],
                    "structured_reply": None
                }
            except Exception as e:
                logger.error(f"Error in async standard output generation: {str(e)}")
                return {
                    "replies": [f"Error: {str(e)}"],
                    "meta": [{"error": str(e)}],
                    "structured_reply": None
                }

    def _parse_structured_output(self, text: str) -> BaseModel:
        """
        Parse the JSON response from Cohere into the provided Pydantic model
        """
        try:
            # Try to find JSON in the text (in case there's additional text)
            json_start = text.find('{')
            json_end = text.rfind('}') + 1

            if 0 <= json_start < json_end:
                json_str = text[json_start:json_end]
                logger.debug(f"Extracted JSON: {json_str[:100]}...")

                # Parse the JSON
                parsed_data = json.loads(json_str)

                # Convert to Pydantic model
                structured_output = self.model_type.model_validate(parsed_data)
                return structured_output
            else:
                # Try to parse the whole text as JSON
                logger.warning("Could not find JSON delimiters, attempting to parse entire response")
                structured_output = self.model_type.model_validate_json(text)
                return structured_output

        except Exception as e:
            logger.error(f"Failed to parse structured output: {str(e)}. Response was: {text[:200]}...")

            # Return properly initialized empty model
            if self.model_type == Questions:
                # For Questions model, initialize with empty questions list
                return Questions(questions=[])
            else:
                # For other models, use empty initialization
                return self.model_type()
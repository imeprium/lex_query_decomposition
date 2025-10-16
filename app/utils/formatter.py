"""
Formatter utilities for converting structured data to markdown and other formats.
"""
import json
from typing import Dict, Any, List


class MarkdownFormatter:
    """
    Formats structured data as markdown with consistent styling.
    Extracts and formats answers, sources, and questions.
    """

    @staticmethod
    def format_as_markdown(result: Dict[str, Any]) -> str:
        """
        Format query response as JSON with markdown content.

        Args:
            result: Query result dictionary

        Returns:
            JSON string with formatted markdown content
        """
        output = []

        # Empty heading instead of question (privacy/security consideration)
        output.append("# ")
        output.append("")

        # Process sections in preferred order
        sections = [
            MarkdownFormatter._format_answer_section,
            MarkdownFormatter._format_sources_section,
            MarkdownFormatter._format_questions_section
        ]

        # Apply each section formatter
        for formatter in sections:
            section_content = formatter(result)
            if section_content:
                output.extend(section_content)

        # Combine into a single markdown string
        markdown_content = "\n".join(output)

        # Return as JSON with markdown key
        return json.dumps({"markdown": markdown_content})

    @staticmethod
    def _format_answer_section(result: Dict[str, Any]) -> List[str]:
        """Format the answer section"""
        if not result.get("final_answer"):
            return []

        output = ["## Answer", ""]

        # Process answer content
        final_answer = result["final_answer"]
        sections = MarkdownFormatter._extract_sections(final_answer)

        if sections:
            # Format sections with ### headings
            for heading, content in sections:
                output.append(f"### {heading}")
                output.append("")
                output.append(content)
                output.append("")
        else:
            # No sections detected, include entire answer
            output.append(final_answer)
            output.append("")

        return output

    @staticmethod
    def _format_sources_section(result: Dict[str, Any]) -> List[str]:
        """Format the sources section"""
        if not result.get("document_metadata"):
            return []

        output = ["### Sources", ""]

        # Create a dictionary to deduplicate sources
        unique_sources = {}
        for doc in result["document_metadata"]:
            source_key = None

            # Find the title field
            for field in ["case_title", "article_title", "legislation_title"]:
                if field in doc:
                    source_key = (field, doc[field])
                    break

            # Only keep sources with higher scores
            if source_key and (
                    source_key not in unique_sources or
                    doc.get("score", 0) > unique_sources[source_key].get("score", 0)):
                unique_sources[source_key] = doc

        # Output unique sources
        for (field_type, title), doc in unique_sources.items():
            type_name = field_type.replace("_title", "").capitalize()
            output.append(f"- **{type_name}**: {title} (Document ID: {doc.get('document_id', 'Unknown')})")

        output.append("")
        return output

    @staticmethod
    def _format_questions_section(result: Dict[str, Any]) -> List[str]:
        """Format the key legal questions section"""
        decomposed_questions = result.get("decomposed_questions")
        if not decomposed_questions:
            return []

        output = ["## Key Legal Questions", ""]

        for i, q_a in enumerate(decomposed_questions, 1):
            # Handle both dictionary and Pydantic model formats
            if hasattr(q_a, 'question'):
                # It's a Pydantic model
                question = q_a.question
                answer = q_a.answer if q_a.answer is not None else "No answer available"
            else:
                # It's a dictionary
                question = q_a.get('question', '')
                answer = q_a.get('answer', 'No answer available')

            output.append(f"### Q{i}: {question}")
            output.append("")
            output.append(answer)
            output.append("")

        return output

    @staticmethod
    def _extract_sections(text: str) -> List[tuple]:
        """
        Extract sections from text based on headings.

        Args:
            text: Text to extract sections from

        Returns:
            List of (heading, content) tuples
        """
        sections = []
        current_section = []
        current_heading = None

        for line in text.split('\n'):
            # Check if line is a potential heading
            is_heading = (line.endswith(':') and not line.startswith(' ')) or (
                    line and all(c.isalpha() or c.isspace() for c in line) and len(line) < 30
            )

            if is_heading:
                # Save previous section if it exists
                if current_heading and current_section:
                    sections.append((current_heading, '\n'.join(current_section)))

                # Start new section
                current_heading = line.strip(':')
                current_section = []
            else:
                current_section.append(line)

        # Add the last section
        if current_heading and current_section:
            sections.append((current_heading, '\n'.join(current_section)))

        return sections


# Export the function with the original name for backward compatibility
def format_as_markdown(result: Dict[str, Any]) -> str:
    """
    Format query response as markdown.
    Backward-compatible function that uses the MarkdownFormatter class.

    Args:
        result: Result dictionary

    Returns:
        JSON string with markdown content
    """
    return MarkdownFormatter.format_as_markdown(result)
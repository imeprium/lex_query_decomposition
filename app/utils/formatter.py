from typing import Dict, Any, List
import json


def format_as_markdown(result: Dict[str, Any]) -> str:
    """
    Format the query response as valid JSON with a markdown key.
    Omits the original question and changes "Decomposed Questions" to "Key Legal Questions"
    Now in order: Empty heading, Answer, Sources, Key Legal Questions

    Args:
        result: The query result dictionary

    Returns:
        Valid JSON string with the formatted markdown content
    """
    output = ["# ", ""]

    # Add empty level 1 heading instead of the question

    # Add final answer with sections (moved to be second)
    if "final_answer" in result and result["final_answer"]:
        output.append("## Answer")
        output.append("")

        final_answer = result["final_answer"]

        # Process sections (Introduction, Analysis, etc.)
        sections = []
        current_section = []
        current_heading = None

        for line in final_answer.split('\n'):
            if (line.endswith(':') and not line.startswith(' ')) or (
                    line and all(c.isalpha() or c.isspace() for c in line) and len(line) < 30):
                if current_heading and current_section:
                    sections.append((current_heading, '\n'.join(current_section)))
                current_heading = line.strip(':')
                current_section = []
            else:
                current_section.append(line)

        # Add the last section
        if current_heading and current_section:
            sections.append((current_heading, '\n'.join(current_section)))

        # Format sections with ### headings
        for heading, content in sections:
            output.append(f"### {heading}")
            output.append("")
            output.append(content)
            output.append("")

    # Add document sources (third position)
    if "document_metadata" in result and result["document_metadata"]:
        output.append("### Sources")
        output.append("")

        # Group sources by title to ensure uniqueness
        unique_sources = {}
        for doc in result["document_metadata"]:
            source_key = None
            for field in ["case_title", "article_title", "legislation_title"]:
                if field in doc:
                    source_key = (field, doc[field])
                    break

            if source_key and (
                    source_key not in unique_sources or doc.get("score", 0) > unique_sources[source_key].get("score",
                                                                                                             0)):
                unique_sources[source_key] = doc

        # Output unique sources
        for (field_type, title), doc in unique_sources.items():
            type_name = field_type.replace("_title", "").capitalize()
            output.append(f"- **{type_name}**: {title} (Document ID: {doc.get('document_id', 'Unknown')})")

        output.append("")  # Add blank line after sources

    # MOVED TO BOTTOM: Add key legal questions section (instead of decomposed questions)
    if "decomposed_questions" in result and result["decomposed_questions"]:
        output.append("## Key Legal Questions")
        output.append("")

        for i, q_a in enumerate(result["decomposed_questions"], 1):
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

    # Combine all lines to create the Markdown content
    markdown_content = "\n".join(output)

    # Create a dictionary with Markdown key and then convert to JSON
    result_dict = {"markdown": markdown_content}

    # Return the JSON string
    return json.dumps(result_dict)
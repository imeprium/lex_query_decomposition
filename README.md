# Lexanalytics Legal Query Decomposition System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [API Endpoints](#api-endpoints)
5. [PDF Generation](#pdf-generation)
6. [Troubleshooting](#troubleshooting)
7. [Developer Guide](#developer-guide)

## System Overview

The Legal Query Decomposition System is a sophisticated API service that processes complex legal questions by breaking them down into simpler sub-questions, retrieving relevant documents, and generating comprehensive answers. The system uses a decomposition-based retrieval-augmented generation (RAG) approach to provide accurate, context-aware responses to legal inquiries.

### Architecture

The system consists of several interconnected components:

1. **Query Decomposition**: Breaks down complex legal questions into sub-questions
2. **Document Retrieval**: Uses hybrid search (dense + sparse embeddings) to find relevant documents
3. **Answer Generation**: Creates answers to sub-questions based on retrieved context
4. **Answer Synthesis**: Combines sub-answers into a comprehensive final response
5. **PDF Generation**: Provides downloadable, professionally formatted reports

### Key Features

- Query decomposition for better retrieval
- Hybrid search with dense and sparse embeddings
- Reranking of search results
- Formatted PDF reports with optional watermarks and signatures
- Caching for improved performance
- Multiple output formats (JSON, Markdown, PDF)

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- Access to Qdrant vector database (can be local or remote)
- Document corpus indexed in Qdrant

### Standard Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd legal_query_decomposition
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p static/signatures templates
   ```

5. Start the application:
   ```bash
   python -m app.main
   ```

### Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t legal_query_decomposition .
   ```

2. Run the container:
   ```bash
   docker run -p 9005:9005 -e QDRANT_URL=<your_qdrant_url> -e QDRANT_API_KEY=<your_api_key> legal_query_decomposition
   ```

## Configuration

Configuration is managed through environment variables, which can be set directly or through a `.env` file.

### Core Settings

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `APP_PORT` | 9005 | Port the API server will run on |
| `APP_HOST` | 0.0.0.0 | Host address to bind the server |
| `COHERE_API_KEY` | (required) | API key for Cohere |
| `COHERE_MODEL` | command-r-plus | Cohere model to use |

### Document Store Settings

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `QDRANT_URL` | (required) | URL for Qdrant vector database |
| `QDRANT_API_KEY` | (required) | API key for Qdrant |
| `QDRANT_COLLECTION_NAME` | LegalDocs | Collection name in Qdrant |

### Model Settings

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `DENSE_EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 | Model for dense embeddings |
| `SPARSE_EMBEDDING_MODEL` | prithivida/Splade_PP_en_v1 | Model for sparse embeddings |
| `RANKER_MODEL` | Xenova/ms-marco-MiniLM-L-6-v2 | Model for reranking |

### Cache Settings

| Environment Variable | Default | Description |
|----------------------|---------|-------------|
| `REDIS_CACHE_ENABLED` | True | Enable/disable Redis caching |
| `REDIS_CACHE_TTL` | 3600 | Cache TTL in seconds |
| `UPSTASH_REDIS_REST_URL` | (required if cache enabled) | URL for Redis |
| `UPSTASH_REDIS_REST_TOKEN` | (required if cache enabled) | Token for Redis |

## API Endpoints

The system exposes two main endpoints:

### 1. Ask Legal Question

```
GET /api/ask
```

Process a legal question and return the analysis in JSON or markdown format.

#### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| question | string | Yes | - | The legal question to research |
| format | string | No | json | Response format: 'json' or 'markdown' |

#### Example Request

```bash
curl -X GET "http://localhost:9005/api/ask?question=What%20are%20the%20elements%20of%20rape?&format=json"
```

#### JSON Response Format

```json
{
  "original_question": "What are the elements of rape?",
  "decomposed_questions": [
    {
      "question": "What are the essential elements that must be proven for rape under Nigerian law?",
      "answer": "The essential elements of rape under Nigerian law are that 'the accused had sexual intercourse with the woman in question' and 'the act was done in circumstances following under any one of the five paragraphs in Section 282(1) of the Penal Code.'"
    },
    // Additional questions...
  ],
  "final_answer": "Based on the provided documents, the elements of rape under Nigerian law are...",
  "document_metadata": [
    {
      "id": "doc-1",
      "score": 0.92,
      "case_title": "Eyong Idam v. Federal Republic of Nigeria",
      "document_id": "d8deff2e-0442-40eb-b442-e1260ee7d55c"
    },
    // Additional documents...
  ]
}
```

#### Markdown Response Format

When requesting the markdown format (`format=markdown`), the response will be:

```
{
markdown:
"# 

## Key Legal Questions

### Q1: What are the essential elements that must be proven for rape under Nigerian law?

The essential elements of rape under Nigerian law are that 'the accused had sexual intercourse with the woman in question' and 'the act was done in circumstances following under any one of the five paragraphs in Section 282(1) of the Penal Code.'

...more content...
"
}
```

### 2. Download Legal Analysis PDF

```
GET /api/ask/pdf
```

Process a legal question and return the analysis as a professionally formatted PDF document.

#### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| question | string | Yes | - | The legal question to research |
| include_watermark | boolean | No | true | Include watermark in the PDF |
| sign_document | boolean | No | false | Add visual signature to the document |
| signature_reason | string | No | "Legal Analysis Document" | Reason for signature |
| signature_location | string | No | "Digital" | Location of signing |

#### Example Request

```bash
curl -X GET "http://localhost:9005/api/ask/pdf?question=What%20are%20the%20elements%20of%20rape?&include_watermark=true&sign_document=true" --output legal_analysis.pdf
```

#### Response

The response is a PDF file with Content-Type `application/pdf` and appropriate headers for downloading. The PDF includes:

- Original question as the title
- All decomposed questions and their answers
- Final synthesized answer
- Sources of information
- Optional watermark
- Optional signature block
- Page numbers

## PDF Generation

The system generates professionally formatted PDFs using ReportLab. These PDFs can be enhanced with several features:

### Watermarks

Adding `include_watermark=true` to the PDF endpoint adds a subtle "LEXANALYTICS" watermark diagonally across the pages. This helps identify the source of the document and adds a level of branding.

### Visual Signatures

Adding `sign_document=true` to the PDF endpoint adds a visual signature block to the last page of the document, including:

- Document ID (UUID)
- Timestamp
- Reason for signing
- Location information
- "DIGITALLY VERIFIED" stamp

### Logo

You can add your organization logo to the PDFs by placing an image file at:
```
static/lexanalytics_logo.png
```

This logo will appear in the header of each page.

### Signature Image

You can add a visual signature image by placing a file at:
```
static/signatures/signature.png
```

This image will appear in the signature block when `sign_document=true`.

## Troubleshooting

### Common Issues

#### Connection to Qdrant Fails

**Symptoms**: Server fails to start with error messages about Qdrant connection.

**Solutions**:
- Verify that the Qdrant URL and API key are correct
- Ensure that the Qdrant server is running and accessible
- Check if the collection exists in Qdrant

#### PDF Generation Errors

**Symptoms**: PDF endpoint returns errors or invalid PDFs.

**Solutions**:
- Ensure the `static` and `static/signatures` directories exist
- Verify that ReportLab and PyPDF2 are installed correctly
- Check that the logo file exists at the correct path

#### Rate Limiting with Cohere

**Symptoms**: Responses fail with messages about rate limits.

**Solutions**:
- Check your Cohere API usage
- Consider upgrading your Cohere plan
- Implement additional rate limiting in your application

### Logs

The application uses structured logging. Logs are written to:
- Console (stdout)
- File: `logs/legal_decomposition.log`

You can adjust log levels in `app/config/logging.py`.

## Developer Guide

### Extending the System

#### Adding New Models

1. Create appropriate model class in `app/models.py`
2. Update the pipeline in `app/pipelines/legal_decomposition_pipeline.py`

#### Custom Prompt Templates

Modify the templates in `app/prompts/` to customize:
- Query decomposition logic
- Answer generation style
- Final synthesis approach

#### Custom PDF Styling

To modify PDF styling, edit the `PDFGenerator` class in `app/utils/pdf_generator.py`. You can:
- Change fonts and colors
- Modify page layouts
- Adjust sizing and spacing

### Code Structure

```
app/
├── api.py                      # Main API endpoints
├── components/                 # Haystack components
│   ├── custom_generators.py    # Extended Cohere generator
│   ├── embedders.py            # Dense and sparse embedders
│   ├── retrievers.py           # Hybrid retrieval components
│   └── decomposition_validator.py # Validation components
├── config/                     # Configuration
│   ├── logging.py              # Logging setup
│   └── settings.py             # Environment variables
├── document_store/             # Document store connections
│   └── store.py                # Qdrant document store
├── endpoints/                  # API endpoints
│   ├── __init__.py             # Router initialization
│   └── ask.py                  # Question answering endpoints
├── models.py                   # Pydantic models for data structures
├── pipelines/                  # Query processing pipelines
│   └── legal_decomposition_pipeline.py # Main RAG pipeline
├── prompts/                    # Prompt templates
│   ├── answering.py            # Answer generation prompts
│   ├── decomposition.py        # Query decomposition prompts
│   └── reasoning.py            # Final reasoning prompts
├── utils/                      # Utility functions
│   ├── cache.py                # Redis caching
│   ├── formatter.py            # Output formatting
│   ├── pdf_generator.py        # PDF generation
│   ├── pdf_signer.py           # PDF signing
│   └── sanitizer.py            # Input validation
├── main.py                     # Application entrypoint
└── __init__.py                 # Package initialization
```
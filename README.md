# Week 4 – Session 7: Retrieval-Augmented Generation (RAG) Pipeline

## Project Overview

This project implements a basic Retrieval-Augmented Generation (RAG) pipeline using Python, Jina AI embeddings, and the Groq API.

The pipeline retrieves relevant information from a company handbook and uses a Large Language Model (LLM) to generate answers based only on the retrieved context.

If the required information is not available in the retrieved context, the system responds with:

`Information not found`

## Project Structure

```text
week-04-rag/
│
├── main.py
├── .env
├── requirements.txt
├── README.md
│
└── DATA/
    └── knowledge.txt
```

## Technologies Used

- Python
- Jina AI Embeddings API
- Groq API
- Requests
- Python-dotenv
- Cosine Similarity

## Installation and Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root directory.

Add your API keys:

```text
GROQ_API_KEY=your_groq_api_key
JINA_API_KEY=your_jina_api_key
```

Keep your API keys private. Do not upload the `.env` file to GitHub.

### 4. Run the application

```powershell
python main.py
```

## How the RAG Pipeline Works

1. **Document Loading:** Reads the company handbook from `DATA/knowledge.txt`.
2. **Text Chunking:** Splits the document into chunks of approximately 800 characters with 100 characters of overlap.
3. **Embedding Generation:** Uses Jina AI to convert document chunks into numerical embeddings.
4. **Similarity Search:** Uses cosine similarity to compare the question embedding with document embeddings.
5. **Retrieval:** Selects the top 3 most relevant chunks. In this test, only 2 chunks were available.
6. **Answer Generation:** Sends the retrieved context and question to Groq.
7. **Missing Information Handling:** Returns `Information not found` when the answer is absent from the retrieved context.

## Test Results

The pipeline was tested using three different queries.

### Test 1: Direct Question

**Question:**

How many paid vacation days do full-time employees receive?

**Retrieved similarity scores:**

- Chunk 1: 0.693
- Chunk 2: 0.590

**Actual Answer:**

15 paid vacation days per calendar year.

**Result:** PASSED

### Test 2: Missing Information

**Question:**

What is the company's parental leave policy?

**Retrieved similarity scores:**

- Chunk 1: 0.638
- Chunk 2: 0.564

**Actual Answer:**

`Information not found`

**Result:** PASSED

The handbook does not contain a parental leave policy, and the system correctly identified that the information was unavailable.

### Test 3: Vague Question

**Question:**

What are the time-off arrangements?

**Retrieved similarity scores:**

- Chunk 1: 0.614
- Chunk 2: 0.504

**Actual Answer:**

**Time-off arrangements**

- **Vacation:** Full-time employees receive 15 paid vacation days per calendar year. Requests must be sent to a manager at least two weeks in advance. Unused vacation days do not roll over.
- **Sick Leave:** Employees receive 8 paid sick days annually. A medical note is required after more than three consecutive sick days.

**Result:** PASSED

## Exit Ticket

**Question: Why should you debug retrieval failure (inspecting the retrieved chunks) before investigating generation failure when a RAG pipeline produces an incorrect or "not found" answer?**

**Answer:**

Retrieval should be debugged first because the generation model depends on the context supplied by the retrieval system.

If the retrieved chunks are irrelevant, incomplete, or do not contain the required information, the language model cannot reliably generate a correct, context-grounded answer.

Inspecting the retrieved chunks helps identify whether the problem comes from text chunking, embedding generation, similarity scoring, or retrieval.

Once retrieval is confirmed to be working correctly, generation can then be investigated separately.

## Conclusion

The Session 7 RAG pipeline was successfully implemented and tested.

All three test queries produced the expected results:

- Direct factual questions were answered correctly.
- Missing information was handled using `Information not found`.
- Vague questions retrieved relevant information and generated useful answers.

The project demonstrates how document retrieval and LLM generation can work together to provide context-grounded answers.
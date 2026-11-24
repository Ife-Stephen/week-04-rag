import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# 1. PROJECT CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

GROQ_KEY = os.getenv("GROQ_API_KEY")
JINA_KEY = os.getenv("JINA_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JINA_URL = "https://api.jina.ai/v1/embeddings"

GROQ_MODEL = "openai/gpt-oss-20b"
JINA_MODEL = "jina-embeddings-v4"

# Knowledge-base location
KNOWLEDGE_BASE_PATH = BASE_DIR / "DATA" / "knowledge.txt"

TOP_K = 3
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# 2. TEXT CHUNKING

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    chunks = []
    start = 0

    while start < len(text):
        piece = text[start:start + size].strip()

        if piece:
            chunks.append(piece)

        if start + size >= len(text):
            break

        start += size - overlap

    return chunks


# 3. GENERATE EMBEDDINGS

def embed(texts, task, api_key):

    if not api_key:
        raise ValueError("Missing JINA_API_KEY in your .env file.")

    if not texts:
        return []

    response = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": JINA_MODEL,
            "task": task,
            "input": texts
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json().get("data", [])

    if len(data) != len(texts):
        raise ValueError("Jina returned an unexpected number of embeddings.")

    # Preserve the original input order
    data.sort(key=lambda item: item.get("index", 0))

    return [item["embedding"] for item in data]


# 4. COSINE SIMILARITY

def cosine(a, b):

    if len(a) != len(b):
        raise ValueError("Embedding dimensions do not match.")

    dot = sum(x * y for x, y in zip(a, b))

    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


# 5. RETRIEVE RELEVANT CHUNKS

def retrieve(question, chunks, embeddings, top_k=TOP_K):

    if not chunks or not embeddings:
        return []

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings do not match.")

    query_embedding = embed(
        [question],
        "retrieval.query",
        JINA_KEY
    )[0]

    ranked = []

    for chunk, vector in zip(chunks, embeddings):
        score = cosine(query_embedding, vector)
        ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return ranked[:top_k]


# 6. GENERATE ANSWER USING GROQ

def answer(question, ranked):

    if not ranked:
        return "Information not found"

    context = "\n\n".join(
        chunk for _, chunk in ranked
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    'Answer using only the provided context. '
                    'If the answer is absent from the context, '
                    'say exactly "Information not found". '
                    'Do not use outside knowledge. '
                    'Do not guess or invent information.'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        "temperature": 0
    }

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()


# 7. MAIN PROGRAM

def main():

    print("=" * 50)
    print("       WEEK 4 - RAG PIPELINE")
    print("=" * 50)

    # Check API keys
    if not GROQ_KEY or not JINA_KEY:
        print("\nERROR: Missing API keys.")
        print("Check that your .env file contains:")
        print("GROQ_API_KEY=your_groq_key")
        print("JINA_API_KEY=your_jina_key")
        return

    try:

        # Check whether knowledge base exists
        if not KNOWLEDGE_BASE_PATH.is_file():
            print("\nERROR: Knowledge-base file not found.")
            print(f"Expected location: {KNOWLEDGE_BASE_PATH}")
            return

        # Load knowledge base
        text = KNOWLEDGE_BASE_PATH.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            print("\nERROR: Knowledge-base file is empty.")
            return

        # Split text into chunks
        chunks = chunk_text(text)

        print(f"\nIndexed {len(chunks)} chunks.")
        print(f"Source file: {KNOWLEDGE_BASE_PATH}")

        # Generate document embeddings
        embeddings = embed(
            chunks,
            "retrieval.passage",
            JINA_KEY
        )

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts do not match.")

    
        # 8. TEST QUERIES

        tests = [
            "How many paid vacation days do full-time employees receive?",
            "What is the company's parental leave policy?",
            "What are the time-off arrangements?"
        ]

        for question in tests:

            print("\n" + "=" * 50)
            print("QUESTION:", question)

            ranked = retrieve(
                question,
                chunks,
                embeddings,
                TOP_K
            )

            print("\nRETRIEVED CHUNKS:")

            if not ranked:
                print("No chunks retrieved.")

            for score, chunk in ranked:
                print(f"\n[{score:.3f}] {chunk[:220]}...")

            # Generate final answer
            result = answer(question, ranked)

            print("\nANSWER:")
            print(result)

        print("\n" + "=" * 50)
        print("RAG PIPELINE COMPLETED SUCCESSFULLY.")
        print("=" * 50)

    except requests.exceptions.HTTPError as e:

        print("\nAPI HTTP ERROR:", e)

        if e.response is not None:
            print("Details:", e.response.text)

    except requests.exceptions.Timeout:
        print("\nERROR: API request timed out. Try again.")

    except requests.exceptions.RequestException as e:
        print("\nNETWORK/API ERROR:", e)

    except (OSError, ValueError, KeyError, TypeError) as e:
        print("\nPROGRAM ERROR:", e)


# 9. RUN PROGRAM

if __name__ == "__main__":
    main()

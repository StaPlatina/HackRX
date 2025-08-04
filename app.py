from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import logging
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import PyPDF2
import io
import json
import uuid
import os
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Q&A API",
    description="Upload documents, embed them in Pinecone, and answer questions using LLM",
    version="1.0.0"
)

# Note: Render handles HTTPS automatically, no need for HTTPS redirect middleware
# Only add trusted host middleware if you want to restrict domains
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
if ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# Security
security = HTTPBearer()

# Configuration - Replace with your actual API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VALID_BEARER_TOKEN = os.getenv("VALID_BEARER_TOKEN")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone index configuration
INDEX_NAME = "document-qa-index"
DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Pydantic models
class DocumentRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QuestionAnswer(BaseModel):
    answers: List[str]

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != VALID_BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    return credentials.credentials

# Initialize Pinecone index
async def initialize_pinecone_index():
    """Initialize Pinecone index if it doesn't exist"""
    try:
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Created Pinecone index: {INDEX_NAME}")
        else:
            logger.info(f"Pinecone index {INDEX_NAME} already exists")
            
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize vector database")

# Document processing functions
async def download_document(url: str) -> bytes:
    """Download document from URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence end
        if end < len(text):
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            break_point = max(last_period, last_newline)
            
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

async def embed_and_store_chunks(chunks: List[str], document_url: str, index) -> None:
    """Embed text chunks and store in Pinecone"""
    try:
        vectors = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embedding_model.encode(chunk).tolist()
            
            # Create vector with metadata
            vector_id = f"{uuid.uuid4()}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "document_url": str(document_url),
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        # Upsert vectors to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            
        logger.info(f"Successfully stored {len(vectors)} chunks in Pinecone")
        
    except Exception as e:
        logger.error(f"Failed to embed and store chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to store document embeddings")

async def retrieve_relevant_chunks(questions: List[str], index, top_k: int = 5) -> List[str]:
    """Retrieve relevant chunks for given questions"""
    try:
        relevant_chunks = []
        
        for question in questions:
            # Embed the question
            question_embedding = embedding_model.encode(question).tolist()
            
            # Query Pinecone
            results = index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract relevant text chunks
            for match in results.matches:
                if match.score > 0.7:  # Similarity threshold
                    chunk_text = match.metadata.get('text', '')
                    if chunk_text and chunk_text not in relevant_chunks:
                        relevant_chunks.append(chunk_text)
        
        return relevant_chunks[:20]  # Limit to top 20 most relevant chunks
        
    except Exception as e:
        logger.error(f"Failed to retrieve relevant chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant information")

async def generate_answers(questions: List[str], relevant_chunks: List[str]) -> List[str]:
    """Generate answers using Groq LLM"""
    try:
        # Prepare context
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt
        prompt = f"""Based on the following document context, please answer the questions accurately and concisely. If the answer cannot be found in the context, state that the information is not available in the provided document.

Context:
{context}

Questions and Answers:
"""
        
        for i, question in enumerate(questions, 1):
            prompt += f"\n{i}. {question}\n"
        
        # Call Groq API
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama3-8b-8192",  # or "mixtral-8x7b-32768" for larger context
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers questions based on provided document context. Be precise and concise in your responses. Answer each question with a clear, direct response."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.1,
            top_p=0.9
        )
        
        # Parse the response
        full_response = response.choices[0].message.content.strip()
        
        # Extract individual answers
        answers = []
        lines = full_response.split('\n')
        current_answer = ""
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, len(questions) + 1)):
                if current_answer:
                    answers.append(current_answer.strip())
                current_answer = line.split('.', 1)[1].strip() if '.' in line else line
            elif current_answer and line:
                current_answer += " " + line
        
        if current_answer:
            answers.append(current_answer.strip())
        
        # Ensure we have the right number of answers
        while len(answers) < len(questions):
            answers.append("Information not available in the provided document.")
        
        return answers[:len(questions)]
        
    except Exception as e:
        logger.error(f"Failed to generate answers: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answers")

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await initialize_pinecone_index()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Document Q&A API is running"}

@app.post("/hackrx/run", response_model=QuestionAnswer)
async def process_document_and_answer_questions(
    request: DocumentRequest,
    token: str = Security(verify_token)
):
    """
    Main endpoint to process document and answer questions
    """
    try:
        logger.info(f"Processing request for document: {request.documents}")
        
        # Initialize Pinecone index
        index = await initialize_pinecone_index()
        
        # Step 1: Download document
        logger.info("Downloading document...")
        document_content = await download_document(str(request.documents))
        
        # Step 2: Extract text from PDF
        logger.info("Extracting text from document...")
        text = extract_text_from_pdf(document_content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        # Step 3: Chunk the text
        logger.info("Chunking document text...")
        chunks = chunk_text(text)
        
        # Step 4: Embed and store in Pinecone
        logger.info("Embedding and storing chunks...")
        await embed_and_store_chunks(chunks, request.documents, index)
        
        # Step 5: Retrieve relevant chunks for questions
        logger.info("Retrieving relevant chunks...")
        relevant_chunks = await retrieve_relevant_chunks(request.questions, index)
        
        if not relevant_chunks:
            logger.warning("No relevant chunks found for the questions")
            answers = ["Information not available in the provided document." for _ in request.questions]
        else:
            # Step 6: Generate answers using LLM
            logger.info("Generating answers...")
            answers = await generate_answers(request.questions, relevant_chunks)
        
        logger.info("Request processed successfully")
        return QuestionAnswer(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check Pinecone connection
        pc.list_indexes()
        
        return {
            "status": "healthy",
            "services": {
                "pinecone": "connected",
                "groq": "configured",
                "embedding_model": "loaded"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render sets this automatically)
    PORT = int(os.getenv("PORT", "8000"))
    
    # Always use HTTP locally - Render handles HTTPS at the load balancer level
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        # Remove any SSL configuration for Render
        # Render handles HTTPS termination at the load balancer
    )

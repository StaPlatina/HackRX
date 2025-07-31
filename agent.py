import os
import hashlib
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import uuid
import re
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, HttpUrl
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import aiohttp
import redis
from functools import lru_cache
import tiktoken
import time
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent RAG System", version="4.0.0")

# Add GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=500)

# Global session for HTTP connections
session: Optional[aiohttp.ClientSession] = None

# Agent Types
class AgentType(Enum):
    RETRIEVAL_AGENT = "retrieval"
    ANALYSIS_AGENT = "analysis"
    FORMATTING_AGENT = "formatting"
    ORCHESTRATOR_AGENT = "orchestrator"

# Configuration
class Config:
    # Vector DB Config
    QDRANT_URL = "https://2554cb57-0957-4521-95bc-38baa9df1b45.us-west-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sQAgEJa_vDzoXOBCAO2lTjHwiaoqQ9o_KDv6SG67uzg"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Document Processing
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    MIN_CHUNK_LENGTH = 100
    
    # Agent Config
    AWS_REGION = "us-east-1"
    DEEPSEEK_MODEL_ID = "us.deepseek.r1-v1:0"
    MAX_AGENT_TOKENS = 4000
    
    # Retrieval Config
    TOP_K_RETRIEVAL = 15
    TOP_K_FOR_AGENT = 10
    
    # System Config
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    CACHE_EXPIRY = 86400
    TIMEOUT = 60
    USE_GPU = torch.cuda.is_available()

config = Config()

# Thread pools
thread_pool = ThreadPoolExecutor(max_workers=4)
agent_pool = ThreadPoolExecutor(max_workers=3)

# Initialize services
@lru_cache(maxsize=1)
def get_embedding_model():
    logger.info("Initializing embedding model...")
    device = "cuda" if config.USE_GPU else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    return model

@lru_cache(maxsize=1)
def get_bedrock_client():
    logger.info("Initializing AWS Bedrock client...")
    return boto3.client('bedrock-runtime', region_name=config.AWS_REGION)

@lru_cache(maxsize=1)
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

# Initialize services
embedding_model = get_embedding_model()
bedrock_runtime = get_bedrock_client()
tokenizer = get_tokenizer()

# Redis client
try:
    redis_pool = redis.ConnectionPool(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        max_connections=20,
        decode_responses=True
    )
    redis_client = redis.Redis(connection_pool=redis_pool)
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# Qdrant client
try:
    qdrant_client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        prefer_grpc=True,
        timeout=config.TIMEOUT
    )
    qdrant_client.get_collections()
    logger.info("Qdrant connected successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {e}")
    raise

# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    global session
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=100),
        timeout=aiohttp.ClientTimeout(total=config.TIMEOUT)
    )
    logger.info("Initialized aiohttp session")

@app.on_event("shutdown")
async def shutdown_event():
    global session
    if session:
        await session.close()

# Request/Response Models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
async def verify_token(authorization: str = Header(...)):
    expected_token = "Bearer 9fb2eed12d2b4def6242c0e16708fe60fa1d99fd7fa2c6323d1913cd4303b446"
    if authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return True

# Document Processing
class DocumentProcessor:
    @staticmethod
    def generate_collection_name(url: str) -> str:
        """Generate collection name from URL"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"doc_{url_hash}"
    
    @staticmethod
    async def download_document(url: str) -> bytes:
        """Download document"""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download: {str(e)}")
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> str:
        """Extract text from PDF"""
        try:
            import io
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    full_text += f"\n\n[Page {page_num}]\n{text}"
            
            return full_text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_docx(content: bytes) -> str:
        """Extract text from DOCX"""
        try:
            import io
            doc = docx.Document(io.BytesIO(content))
            
            full_text = ""
            for para in doc.paragraphs:
                if para.style.name.startswith('Heading'):
                    full_text += f"\n\n### {para.text} ###\n"
                elif para.text.strip():
                    full_text += para.text + "\n"
            
            return full_text.strip()
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process DOCX: {str(e)}")
    
    @staticmethod
    def smart_chunk_text(text: str) -> List[Dict[str, Any]]:
        """Create smart chunks for vector storage"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > config.CHUNK_SIZE and current_chunk:
                page_match = re.search(r'\[Page (\d+)\]', current_chunk)
                page_num = int(page_match.group(1)) if page_match else 0
                
                clean_chunk = re.sub(r'\[Page \d+\]', '', current_chunk).strip()
                
                if len(clean_chunk) >= config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": clean_chunk,
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "tokens": current_tokens
                    })
                    chunk_id += 1
                
                # Start new chunk with overlap
                if config.CHUNK_OVERLAP > 0:
                    overlap_words = current_chunk.split()[-config.CHUNK_OVERLAP:]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    current_tokens = len(tokenizer.encode(current_chunk))
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add last chunk
        if current_chunk.strip() and len(current_chunk) >= config.MIN_CHUNK_LENGTH:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": chunk_id,
                "page_number": 0,
                "tokens": current_tokens
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

# Vector Storage
class VectorStorage:
    @staticmethod
    def collection_exists(collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = qdrant_client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except:
            return False
    
    @staticmethod
    def create_collection(collection_name: str):
        """Create collection"""
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    @staticmethod
    async def store_embeddings(collection_name: str, chunks: List[Dict[str, Any]], document_url: str):
        """Store chunk embeddings"""
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            thread_pool,
            lambda: embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()
        )
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk.get("chunk_id", i),
                    "page_number": chunk.get("page_number", 0),
                    "document_url": document_url
                }
            )
            points.append(point)
        
        # Store in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)
        
        logger.info(f"Stored {len(points)} embeddings")

# Multi-Agent System
class MultiAgentSystem:
    
    @staticmethod
    async def retrieval_agent(query: str, collection_name: str) -> List[Dict[str, Any]]:
        """Agent 1: Intelligent retrieval from vector database"""
        try:
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                thread_pool,
                lambda: embedding_model.encode([query], show_progress_bar=False)[0].tolist()
            )
            
            # Search for relevant chunks
            search_result = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=config.TOP_K_RETRIEVAL,
                score_threshold=0.0
            ).points
            
            # Convert to chunk format
            chunks = []
            for point in search_result:
                chunks.append({
                    "text": point.payload.get("text", ""),
                    "score": float(point.score),
                    "page_number": point.payload.get("page_number", 0),
                    "chunk_id": point.payload.get("chunk_id", 0)
                })
            
            # Sort by score
            chunks.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top chunks
            return chunks[:config.TOP_K_FOR_AGENT]
            
        except Exception as e:
            logger.error(f"Retrieval agent error: {e}")
            return []
    
    @staticmethod
    async def analysis_agent(chunks: List[Dict[str, Any]], question: str) -> str:
        """Agent 2: Analyze chunks and extract answer"""
        if not chunks:
            return "No relevant information found in the document."
        
        # Prepare context
        context = "\n\n".join([
            f"[Excerpt {i+1}, Page {chunk.get('page_number', 'N/A')}]:\n{chunk['text']}"
            for i, chunk in enumerate(chunks)
        ])
        
        prompt = f"""<｜begin_of_sentence｜><｜User｜>You are an Analysis Agent. Analyze the following document excerpts and provide a comprehensive answer to the question.

Document Excerpts:
{context}

Question: {question}

Instructions:
1. Extract ALL relevant information from the excerpts
2. If specific numbers, dates, or limits are mentioned, include them
3. If multiple conditions or scenarios exist, list them all
4. Be thorough but factual - only use information from the excerpts
5. If the information is not in the excerpts, state that clearly

Provide your analysis:<｜Assistant｜>"""
        
        try:
            response = await MultiAgentSystem._query_llm(prompt, agent_type=AgentType.ANALYSIS_AGENT)
            return MultiAgentSystem._clean_response(response)
        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            return "Unable to analyze the information."
    
    @staticmethod
    async def formatting_agent(raw_answer: str, question: str) -> str:
        """Agent 3: Format and make answer concise"""
        prompt = f"""<｜begin_of_sentence｜><｜User｜>You are a Formatting Agent. Your job is to take an analytical answer and make it concise and well-formatted.

Original Question: {question}

Raw Answer: {raw_answer}

Instructions:
1. Make the answer concise but complete
2. Keep all important facts, numbers, and conditions
3. Remove redundancy and verbose explanations
4. Use clear, direct language
5. Ensure the answer directly addresses the question
6. If lists are appropriate, use them for clarity
7. Do not add information not present in the raw answer

Provide the formatted answer:<｜Assistant｜>"""
        
        try:
            response = await MultiAgentSystem._query_llm(prompt, agent_type=AgentType.FORMATTING_AGENT)
            formatted = MultiAgentSystem._clean_response(response)
            
            # Ensure proper ending
            if formatted and not formatted.endswith(('.', '!', '?')):
                formatted += '.'
            
            return formatted
        except Exception as e:
            logger.error(f"Formatting agent error: {e}")
            return raw_answer  # Return unformatted if formatting fails
    
    @staticmethod
    async def orchestrator_agent(questions: List[str], collection_name: str) -> List[str]:
        """Agent 4: Orchestrate the entire Q&A process"""
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Step 1: Retrieval Agent
                chunks = await MultiAgentSystem.retrieval_agent(question, collection_name)
                logger.info(f"Retrieved {len(chunks)} relevant chunks")
                
                # Step 2: Analysis Agent
                raw_answer = await MultiAgentSystem.analysis_agent(chunks, question)
                logger.info(f"Analysis complete: {len(raw_answer)} chars")
                
                # Step 3: Formatting Agent
                formatted_answer = await MultiAgentSystem.formatting_agent(raw_answer, question)
                logger.info(f"Formatting complete: {len(formatted_answer)} chars")
                
                answers.append(formatted_answer)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append("Unable to process this question.")
        
        return answers
    
    @staticmethod
    async def _query_llm(prompt: str, agent_type: AgentType, max_tokens: int = 1500) -> str:
        """Query LLM for specific agent"""
        try:
            request_body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1 if agent_type == AgentType.FORMATTING_AGENT else 0.3,
                "top_p": 0.9,
                "stop": ["<｜User｜>", "<｜begin_of_sentence｜>"]
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                agent_pool,
                lambda: bedrock_runtime.invoke_model(
                    modelId=config.DEEPSEEK_MODEL_ID,
                    body=json.dumps(request_body)
                )
            )
            
            response_body = json.loads(response['body'].read())
            choices = response_body.get('choices', [])
            
            if choices:
                return choices[0].get('text', '').strip()
            return ""
            
        except Exception as e:
            logger.error(f"LLM query error for {agent_type.value}: {e}")
            raise
    
    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean LLM response"""
        if not response:
            return "Unable to generate response."
        
        # Remove thinking tags
        thinking_patterns = [
            r'<think>.*?</think>',
            r'<｜think｜>.*?<｜/think｜>',
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean whitespace
        response = ' '.join(response.split())
        
        return response.strip()

# Cache Service
class CacheService:
    @staticmethod
    def get_answer_cache(collection_name: str, question: str) -> Optional[str]:
        """Get cached answer"""
        if not redis_client:
            return None
        try:
            cache_key = f"ma_answer:{collection_name}:{hashlib.md5(question.encode()).hexdigest()}"
            return redis_client.get(cache_key)
        except:
            return None
    
    @staticmethod
    def set_answer_cache(collection_name: str, question: str, answer: str):
        """Cache answer"""
        if not redis_client:
            return
        try:
            cache_key = f"ma_answer:{collection_name}:{hashlib.md5(question.encode()).hexdigest()}"
            redis_client.setex(cache_key, config.CACHE_EXPIRY, answer)
        except:
            pass
    
    @staticmethod
    def get_collection_cache(collection_name: str) -> Optional[bool]:
        """Check if collection is processed"""
        if not redis_client:
            return None
        try:
            return redis_client.get(f"ma_collection:{collection_name}")
        except:
            return None
    
    @staticmethod
    def set_collection_cache(collection_name: str):
        """Mark collection as processed"""
        if not redis_client:
            return
        try:
            redis_client.setex(f"ma_collection:{collection_name}", config.CACHE_EXPIRY, "1")
        except:
            pass

# Main API Endpoint
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    authorized: bool = Depends(verify_token)
):
    """Process document and answer questions using multi-agent system"""
    start_time = time.time()
    
    try:
        document_url = str(request.documents)
        questions = request.questions
        collection_name = DocumentProcessor.generate_collection_name(document_url)
        
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        
        # Check if document is already processed
        collection_exists = VectorStorage.collection_exists(collection_name)
        is_cached = CacheService.get_collection_cache(collection_name)
        
        if not collection_exists or not is_cached:
            logger.info("Processing new document...")
            
            # Create collection if needed
            if not collection_exists:
                VectorStorage.create_collection(collection_name)
            
            # Download and process document
            content = await DocumentProcessor.download_document(document_url)
            
            # Extract text
            if document_url.lower().endswith('.pdf'):
                full_text = DocumentProcessor.extract_text_from_pdf(content)
            elif document_url.lower().endswith('.docx'):
                full_text = DocumentProcessor.extract_text_from_docx(content)
            else:
                full_text = DocumentProcessor.extract_text_from_pdf(content)
            
            # Create chunks
            chunks = DocumentProcessor.smart_chunk_text(full_text)
            
            # Store in vector database
            await VectorStorage.store_embeddings(collection_name, chunks, document_url)
            
            # Mark as processed
            CacheService.set_collection_cache(collection_name)
        else:
            logger.info("Using existing document embeddings")
        
        # Process questions with multi-agent system
        answers = []
        
        for i, question in enumerate(questions):
            # Check cache first
            cached_answer = CacheService.get_answer_cache(collection_name, question)
            if cached_answer:
                logger.info(f"Using cached answer for question {i+1}")
                answers.append(cached_answer)
                continue
            
            logger.info(f"Processing question {i+1}: {question[:50]}...")
            
            # Use multi-agent system
            agent_answers = await MultiAgentSystem.orchestrator_agent([question], collection_name)
            answer = agent_answers[0] if agent_answers else "Unable to process question."
            
            # Cache the answer
            CacheService.set_answer_cache(collection_name, question, answer)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    qdrant_status = "connected"
    try:
        qdrant_client.get_collections()
    except:
        qdrant_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "qdrant": qdrant_status,
            "redis": "connected" if redis_client else "disconnected",
            "bedrock": "available"
        },
        "agents": [
            "retrieval_agent",
            "analysis_agent", 
            "formatting_agent",
            "orchestrator_agent"
        ]
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent RAG System",
        "version": "4.0.0",
        "docs": "/docs",
        "architecture": {
            "storage": "Qdrant Vector Database",
            "agents": {
                "1_retrieval": "Intelligently retrieves relevant chunks from vector DB",
                "2_analysis": "Analyzes chunks and extracts comprehensive answers",
                "3_formatting": "Formats answers to be concise and well-structured",
                "4_orchestrator": "Coordinates the entire multi-agent workflow"
            },
            "features": [
                "Vector-based semantic search",
                "Multi-agent processing pipeline",
                "Intelligent answer formatting",
                "Redis caching at multiple levels",
                "Support for PDF and DOCX"
            ]
        }
    }

# Debug endpoints
@app.post("/debug/test-agents")
async def test_agents(
    collection_name: str,
    question: str,
    authorized: bool = Depends(verify_token)
):
    """Test individual agents"""
    try:
        # Test retrieval
        chunks = await MultiAgentSystem.retrieval_agent(question, collection_name)
        
        # Test analysis
        raw_answer = await MultiAgentSystem.analysis_agent(chunks, question) if chunks else "No chunks found"
        
        # Test formatting
        formatted = await MultiAgentSystem.formatting_agent(raw_answer, question)
        
        return {
            "question": question,
            "retrieval": {
                "chunks_found": len(chunks),
                "top_scores": [c['score'] for c in chunks[:3]]
            },
            "analysis": {
                "raw_answer_length": len(raw_answer),
                "raw_answer_preview": raw_answer[:200] + "..."
            },
            "formatting": {
                "formatted_length": len(formatted),
                "formatted_answer": formatted
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_level="info"
    )

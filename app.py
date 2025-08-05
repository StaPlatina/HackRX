import os
import json
import asyncio
import hashlib
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Union, TypedDict, Annotated
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import dataclass
import uuid
import re
import io

import aiohttp
import numpy as np
import PyPDF2
import docx
import boto3
import tiktoken
from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, HttpUrl, Field, validator
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import ResponseHandlingException
from botocore.exceptions import ClientError, BotoCoreError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
DOCUMENT_PROCESSING_TIME = Histogram('document_processing_seconds', 'Document processing time')
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')
ERROR_COUNT = Counter('rag_errors_total', 'Total errors', ['error_type'])

@dataclass
class ServiceStatus:
    embedding_model: bool = False
    bedrock: bool = False
    qdrant: bool = False
    last_check: datetime = datetime.utcnow()

# Robust Configuration with validation
class Config:
    # API Keys - Required
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") 
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL", "https://bb8c74df-d846-4381-8492-ac268bc83109.eu-west-2-0.aws.cloud.qdrant.io:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j40xGhFknLMQw79QUVUjCSizFgfgbl174pbo6z4rm8I")
    
    # Local Embedding Model
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Processing parameters - Optimized for local embedding
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K = int(os.getenv("TOP_K", "12"))  # Increased for better coverage
    FINAL_K = int(os.getenv("FINAL_K", "6"))  # More context chunks
    MIN_SCORE = float(os.getenv("MIN_SCORE", "0.5"))  # Lower threshold
    
    # Model settings - Fixed model ID format
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "us.deepseek.r1-v1:0")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))  # Increased for detailed answers
    
    # System settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
    MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", "50")) * 1024 * 1024
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))

    # Parallel processing settings
    MAX_CONCURRENT_QUESTIONS = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "5"))  # Max parallel questions
    ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
    
    # Authentication
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "Bearer 9fb2eed12d2b4def6242c0e16708fe60fa1d99fd7fa2c6323d1913cd4303b446")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")

config = Config()
config.validate()

# Global service status
service_status = ServiceStatus()

# Enhanced Models with validation
class QueryRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the document to process")
    questions: List[str] = Field(..., min_items=1, max_items=20, description="Questions to answer")
    
    @validator('questions')
    def validate_questions(cls, v):
        for q in v:
            if len(q.strip()) < 5:
                raise ValueError("Questions must be at least 5 characters long")
            if len(q) > 500:
                raise ValueError("Questions must be less than 500 characters")
        return [q.strip() for q in v]

class QueryResponse(BaseModel):
    answers: List[str]

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    uptime: float
    version: str
    timestamp: datetime

# Custom Exceptions
class DocumentProcessingError(Exception):
    pass

class EmbeddingError(Exception):
    pass

class VectorSearchError(Exception):
    pass

class LLMError(Exception):
    pass

# LangGraph State Definition
class WorkflowState(TypedDict):
    question: str
    document_url: str
    collection_name: str
    question_analysis: Dict[str, Any]
    search_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    refined_chunks: List[Dict[str, Any]]
    answer: str
    confidence: float
    workflow_steps: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int

# Rate limiting - In-memory store
rate_limit_store = {}

def clean_rate_limit_store():
    """Clean old entries from rate limit store"""
    now = time.time()
    for client_id in list(rate_limit_store.keys()):
        rate_limit_store[client_id] = [
            call_time for call_time in rate_limit_store[client_id] 
            if now - call_time < 60
        ]
        if not rate_limit_store[client_id]:
            del rate_limit_store[client_id]

def check_rate_limit(client_id: str, max_calls: int = 100) -> bool:
    """Check if client has exceeded rate limit"""
    clean_rate_limit_store()
    
    current_calls = len(rate_limit_store.get(client_id, []))
    if current_calls >= max_calls:
        return False
    
    rate_limit_store.setdefault(client_id, []).append(time.time())
    return True

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
    
    def call(self, func):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func()
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            
            raise e

# Global variables for services
qdrant_client: Optional[AsyncQdrantClient] = None
bedrock_client = None
session: Optional[aiohttp.ClientSession] = None
tokenizer = None
embedding_model: Optional[SentenceTransformer] = None
start_time = time.time()

# In-memory cache for metadata (replacing Redis)
in_memory_cache = {}
CACHE_EXPIRY_SECONDS = 86400  # 24 hours

def get_cache_key(prefix: str, identifier: str) -> str:
    """Generate cache key with namespace"""
    key_hash = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    return f"rag:v5:{prefix}:{key_hash}"

def is_cache_expired(timestamp: float) -> bool:
    """Check if cache entry is expired"""
    return time.time() - timestamp > CACHE_EXPIRY_SECONDS

def get_from_cache(key: str) -> Optional[Dict]:
    """Get from in-memory cache"""
    if key in in_memory_cache:
        entry = in_memory_cache[key]
        if not is_cache_expired(entry['timestamp']):
            return entry['data']
        else:
            del in_memory_cache[key]
    return None

def set_in_cache(key: str, data: Dict):
    """Set in in-memory cache"""
    in_memory_cache[key] = {
        'data': data,
        'timestamp': time.time()
    }

def clean_expired_cache():
    """Clean expired entries from cache"""
    current_time = time.time()
    expired_keys = []
    for key, entry in in_memory_cache.items():
        if current_time - entry['timestamp'] > CACHE_EXPIRY_SECONDS:
            expired_keys.append(key)
    
    for key in expired_keys:
        del in_memory_cache[key]

# Enhanced Document Processing
class RobustDocumentProcessor:
    @staticmethod
    def get_collection_name(url: str) -> str:
        """Generate deterministic collection name"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"doc_{url_hash}"
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def download_document(url: str) -> bytes:
        """Download document with retry logic and validation"""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > config.MAX_DOCUMENT_SIZE:
                    raise DocumentProcessingError(f"Document too large: {content_length} bytes")
                
                content = await response.read()
                
                if len(content) > config.MAX_DOCUMENT_SIZE:
                    raise DocumentProcessingError(f"Document too large: {len(content)} bytes")
                
                return content
                
        except aiohttp.ClientError as e:
            ERROR_COUNT.labels(error_type="download_failed").inc()
            raise DocumentProcessingError(f"Failed to download document: {str(e)}")
    
    @staticmethod
    def extract_text(content: bytes, url: str) -> str:
        """Extract text with enhanced error handling"""
        try:
            if '.pdf' in url.lower():
                return RobustDocumentProcessor._extract_pdf_robust(content)
            elif '.docx' in url.lower():
                return RobustDocumentProcessor._extract_docx_robust(content)
            elif any(ext in url.lower() for ext in ['.txt', '.md']):
                return content.decode('utf-8', errors='replace')
            else:
                try:
                    return RobustDocumentProcessor._extract_pdf_robust(content)
                except:
                    return content.decode('utf-8', errors='replace')
        except Exception as e:
            ERROR_COUNT.labels(error_type="text_extraction_failed").inc()
            raise DocumentProcessingError(f"Text extraction failed: {str(e)}")
    
    @staticmethod
    def _extract_pdf_robust(content: bytes) -> str:
        """Robust PDF extraction"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_parts = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text().strip()
                    if page_text:
                        text_parts.append(f"[Page {page_num}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue
            
            if not text_parts:
                raise DocumentProcessingError("No text could be extracted from PDF")
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise DocumentProcessingError(f"PDF processing failed: {str(e)}")
    
    @staticmethod
    def _extract_docx_robust(content: bytes) -> str:
        """Robust DOCX extraction"""
        try:
            doc = docx.Document(io.BytesIO(content))
            paragraphs = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    if para.style.name.startswith('Heading'):
                        paragraphs.append(f"\n## {text}\n")
                    else:
                        paragraphs.append(text)
            
            if not paragraphs:
                raise DocumentProcessingError("No text found in DOCX document")
            
            return "\n".join(paragraphs)
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise DocumentProcessingError(f"DOCX processing failed: {str(e)}")
    
    @staticmethod
    def create_intelligent_chunks(text: str) -> List[Dict[str, Any]]:
        """Create optimized chunks for local embedding model"""
        if not text or len(text.strip()) < 100:
            raise DocumentProcessingError("Document text too short or empty")
        
        chunks = []
        sections = re.split(r'(\[Page \d+\]|#{1,3}\s+.*)', text)
        current_context = {"page": 0, "section": ""}
        
        for section in sections:
            if not section.strip():
                continue
            
            page_match = re.match(r'\[Page (\d+)\]', section)
            if page_match:
                current_context["page"] = int(page_match.group(1))
                continue
            
            heading_match = re.match(r'#{1,3}\s+(.*)', section)
            if heading_match:
                current_context["section"] = heading_match.group(1).strip()
                continue
            
            sentences = re.split(r'(?<=[.!?])\s+', section)
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                token_count = len(tokenizer.encode(test_chunk))
                
                if token_count > config.CHUNK_SIZE and current_chunk:
                    if len(current_chunk.strip()) > 100:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "page": current_context["page"],
                            "section": current_context["section"],
                            "length": len(current_chunk),
                            "tokens": token_count,
                            "chunk_id": len(chunks)
                        })
                    
                    overlap_sentences = current_chunk.split('. ')[-1:] if '. ' in current_chunk else []
                    current_chunk = '. '.join(overlap_sentences) + ". " + sentence if overlap_sentences else sentence
                else:
                    current_chunk = test_chunk
            
            if current_chunk.strip() and len(current_chunk.strip()) > 100:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": current_context["page"],
                    "section": current_context["section"],
                    "length": len(current_chunk),
                    "tokens": len(tokenizer.encode(current_chunk)),
                    "chunk_id": len(chunks)
                })
        
        chunks = [chunk for chunk in chunks if chunk["tokens"] > 50]
        
        if not chunks:
            raise DocumentProcessingError("No valid chunks could be created")
        
        logger.info(f"Created {len(chunks)} optimized chunks (avg tokens: {sum(c['tokens'] for c in chunks) // len(chunks)})")
        return chunks

# Enhanced Embedding Service with local SentenceTransformer
class RobustEmbeddingService:
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((Exception,))
    )  
    async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local SentenceTransformer model"""
        if not service_status.embedding_model:
            raise EmbeddingError("Local embedding model unavailable")
        
        try:
            logger.info(f"Processing {len(texts)} texts for embedding generation")
            
            # Prepare texts (truncate if too long for the model)
            processed_texts = []
            for text in texts:
                if len(text) > 8000:  # Conservative limit for sentence transformers
                    text = text[:7900] + "..."
                processed_texts.append(text)
            
            def _generate_batch():
                # Use the local embedding model
                embeddings = embedding_model.encode(processed_texts, 
                                                  convert_to_numpy=True,
                                                  show_progress_bar=False,
                                                  batch_size=config.BATCH_SIZE)
                return embeddings.tolist()
            
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, _generate_batch)
            
            logger.info(f"Generated {len(embeddings)} embeddings successfully")
            return embeddings
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="embedding_failed").inc()
            logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")
    
    @staticmethod
    async def generate_embeddings(texts: List[str]) -> List[List[float]]:
        """Wrapper for backward compatibility"""
        return await RobustEmbeddingService.generate_embeddings_batch(texts)
    
    @staticmethod
    async def generate_query_embedding(query: str) -> List[float]:
        """Generate query-optimized embedding using local model"""
        try:
            def _generate():
                # Use the local embedding model for query
                embedding = embedding_model.encode([query], 
                                                 convert_to_numpy=True,
                                                 show_progress_bar=False)
                return embedding[0].tolist()
            
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, _generate)
            return embedding
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="query_embedding_failed").inc()
            raise EmbeddingError(f"Query embedding failed: {str(e)}")

# Enhanced Vector Store with connection pooling
class RobustVectorStore:
    @staticmethod
    async def collection_exists(collection_name: str) -> bool:
        """Check collection existence with error handling"""
        if not service_status.qdrant:
            return False
        
        try:
            collections = await qdrant_client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            ERROR_COUNT.labels(error_type="collection_check_failed").inc()
            logger.warning(f"Collection check failed: {e}")
            return False
    
    @staticmethod
    async def create_collection(collection_name: str):
        """Create collection with validation - adjusted for local embeddings dimension"""
        if not service_status.qdrant:
            raise VectorSearchError("Qdrant service unavailable")
        
        try:
            # all-MiniLM-L6-v2 produces 384-dimensional embeddings
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Changed from 768 to 384 for all-MiniLM-L6-v2
                    distance=Distance.COSINE,
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 100
                    }
                )
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            ERROR_COUNT.labels(error_type="collection_creation_failed").inc()
            raise VectorSearchError(f"Failed to create collection: {str(e)}")
    
    @staticmethod
    async def store_chunks(collection_name: str, chunks: List[Dict[str, Any]], url: str):
        """Store chunks with optimized batch processing"""
        if not chunks:
            raise VectorSearchError("No chunks to store")
        
        try:
            chunk_texts = [chunk["text"] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
            
            embeddings = await RobustEmbeddingService.generate_embeddings_batch(chunk_texts)
            
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "page": chunk.get("page", 0),
                        "section": chunk.get("section", ""),
                        "chunk_id": chunk["chunk_id"],
                        "length": chunk["length"],
                        "tokens": chunk["tokens"],
                        "url": url,
                        "created_at": datetime.utcnow().isoformat()
                    }
                ))
            
            batch_size = 25
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                await qdrant_client.upsert(collection_name=collection_name, points=batch)
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully stored {len(points)} chunks in {collection_name}")
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="chunk_storage_failed").inc()
            raise VectorSearchError(f"Failed to store chunks: {str(e)}")
    
    @staticmethod
    async def enhanced_search(collection_name: str, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Enhanced search with configurable limits"""
        if not service_status.qdrant:
            raise VectorSearchError("Qdrant service unavailable")
        
        try:
            limit = limit or config.TOP_K
            query_embedding = await RobustEmbeddingService.generate_query_embedding(query)
            
            search_results = await qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=config.MIN_SCORE
            )
            
            if not search_results.points:
                return []
            
            query_terms = set(query.lower().split())
            ranked_results = []
            
            for point in search_results.points:
                text_terms = set(point.payload["text"].lower().split())
                
                keyword_overlap = len(query_terms.intersection(text_terms))
                keyword_density = keyword_overlap / len(text_terms) if text_terms else 0
                length_penalty = 1 - (point.payload["length"] - 200) / 2000 if point.payload["length"] > 200 else 1
                
                combined_score = (
                    point.score * 0.7 +
                    keyword_density * 0.2 +
                    length_penalty * 0.1
                )
                
                ranked_results.append({
                    "text": point.payload["text"],
                    "page": point.payload.get("page", 0),
                    "section": point.payload.get("section", ""),
                    "score": combined_score,
                    "semantic_score": point.score,
                    "keyword_overlap": keyword_overlap,
                    "chunk_id": point.payload["chunk_id"],
                    "confidence": min(combined_score, 1.0)
                })
            
            ranked_results.sort(key=lambda x: x["score"], reverse=True)
            return ranked_results
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="vector_search_failed").inc()
            raise VectorSearchError(f"Vector search failed: {str(e)}")

# Simple Cache Service - Using in-memory storage instead of Redis
class RobustCacheService:
    @staticmethod
    def get_cached_document_metadata(url: str) -> Optional[Dict]:
        """Get cached document metadata from in-memory cache"""
        try:
            key = get_cache_key("doc_meta", url)
            result = get_from_cache(key)
            if result:
                logger.info(f"Cache hit for document metadata: {result.get('filename', url[:50])}...")
                return result
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None
    
    @staticmethod
    def cache_document_metadata(url: str, filename: str, size: int, chunks_count: int):
        """Cache document metadata in memory"""
        try:
            key = get_cache_key("doc_meta", url)
            metadata = {
                "filename": filename,
                "size": size,
                "chunks_count": chunks_count,
                "processed_at": datetime.utcnow().isoformat()
            }
            set_in_cache(key, metadata)
            logger.info(f"Document metadata cached: {filename}")
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    @staticmethod
    def get_collection_info(collection_name: str) -> Optional[Dict]:
        """Get cached collection metadata from in-memory cache"""
        try:
            key = get_cache_key("collection", collection_name)
            return get_from_cache(key)
        except Exception:
            return None
    
    @staticmethod
    def cache_collection_info(collection_name: str, info: Dict):
        """Cache collection metadata in memory"""
        try:
            key = get_cache_key("collection", collection_name)
            set_in_cache(key, info)
        except Exception as e:
            logger.warning(f"Collection info cache failed: {e}")

# Enhanced LLM Service with multiple model support
class RobustLLMService:
    @staticmethod
    async def get_available_models():
        """Get list of available models in the region"""
        try:
            models_client = boto3.client(
                'bedrock',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            response = models_client.list_foundation_models()
            available_models = [model['modelId'] for model in response['modelSummaries']]
            logger.info(f"Available models: {available_models}")
            return available_models
        except Exception as e:
            logger.warning(f"Could not list models: {e}")
            return []
    
    @staticmethod
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ClientError, BotoCoreError))
    )
    async def query_llm(prompt: str, max_tokens: int = None) -> str:
        """Query LLM with robust error handling and multiple model support"""
        if not service_status.bedrock:
            raise LLMError("Bedrock service unavailable")
        
        max_tokens = max_tokens or config.MAX_TOKENS
        
        models_to_try = [
            config.DEEPSEEK_MODEL,
        ]
        
        for model_id in models_to_try:
            try:
                logger.info(f"Trying model: {model_id}")
                
                if "meta.llama" in model_id:
                    body = json.dumps({
                        "prompt": prompt,
                        "max_gen_len": max_tokens,
                        "temperature": 0.1,
                        "top_p": 0.9
                    })
                elif "ai21" in model_id:
                    body = json.dumps({
                        "prompt": prompt,
                        "maxTokens": max_tokens,
                        "temperature": 0.1,
                        "topP": 0.9
                    })
                elif "amazon.titan" in model_id:
                    body = json.dumps({
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": max_tokens,
                            "temperature": 0.1,
                            "topP": 0.9
                        }
                    })
                else:
                    body = json.dumps({
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.1,
                        "top_p": 0.9
                    })
                
                response = bedrock_client.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(response.get('body').read())
                
                if 'content' in response_body and response_body['content']:
                    result = response_body['content'][0]['text'].strip()
                elif 'completion' in response_body:
                    result = response_body['completion'].strip()
                elif 'generation' in response_body:
                    result = response_body['generation'].strip()
                elif 'completions' in response_body:
                    result = response_body['completions'][0]['data']['text'].strip()
                elif 'results' in response_body:
                    result = response_body['results'][0]['outputText'].strip()
                elif 'choices' in response_body and response_body['choices']:
                    result = response_body['choices'][0]['message']['content'].strip()
                else:
                    logger.warning(f"Unknown response format for {model_id}: {response_body}")
                    continue
                
                logger.info(f"Successfully used model: {model_id}")
                return result
                
            except Exception as e:
                logger.warning(f"Model {model_id} failed: {str(e)}")
                continue
        
        ERROR_COUNT.labels(error_type="llm_query_failed").inc()
        raise LLMError("All available models failed to respond")

# LangGraph Workflow Implementation
class IntelligentRAGWorkflow:
    def __init__(self):
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("analyze_question", self.analyze_question)
        workflow.add_node("generate_search_queries", self.generate_search_queries)
        workflow.add_node("retrieve_chunks", self.retrieve_chunks)
        workflow.add_node("refine_context", self.refine_context)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("evaluate_answer", self.evaluate_answer)
        
        # Set entry point
        workflow.set_entry_point("analyze_question")
        
        # Add edges
        workflow.add_edge("analyze_question", "generate_search_queries")
        workflow.add_edge("generate_search_queries", "retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "refine_context")
        workflow.add_edge("refine_context", "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_answer")
        
        # Add conditional edge for iteration
        workflow.add_conditional_edges(
            "evaluate_answer",
            self.should_continue,
            {
                "continue": "generate_search_queries",
                "end": END
            }
        )
        
        return workflow.compile()
    
    async def analyze_question(self, state: WorkflowState) -> WorkflowState:
        """Analyze the question to understand intent and key concepts"""
        question = state["question"]
        
        analysis_prompt = f"""
        Analyze this question to understand what the user is looking for:
        
        Question: {question}
        
        Provide a JSON response with:
        1. "intent": The main intent (definition, explanation, specific_fact, comparison, etc.)
        2. "key_terms": List of important keywords and phrases
        3. "domain": The subject domain (insurance, policy, medical, legal, etc.)
        4. "specificity": Level of specificity needed (high, medium, low)
        5. "answer_type": Expected answer format (short_fact, detailed_explanation, list, etc.)
        
        JSON Response:
        """
        
        try:
            analysis_text = await RobustLLMService.query_llm(analysis_prompt, max_tokens=300)
            # Extract JSON from response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                analysis = json.loads(analysis_text[json_start:json_end])
            else:
                # Fallback analysis
                analysis = {
                    "intent": "explanation",
                    "key_terms": question.split(),
                    "domain": "general",
                    "specificity": "medium",
                    "answer_type": "detailed_explanation"
                }
        except Exception as e:
            logger.warning(f"Question analysis failed: {e}")
            analysis = {
                "intent": "explanation",
                "key_terms": question.split(),
                "domain": "general",
                "specificity": "medium",
                "answer_type": "detailed_explanation"
            }
        
        state["question_analysis"] = analysis
        state["workflow_steps"].append({
            "step": "analyze_question",
            "result": analysis,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def generate_search_queries(self, state: WorkflowState) -> WorkflowState:
        """Generate multiple search queries based on question analysis"""
        question = state["question"]
        analysis = state["question_analysis"]
        
        query_prompt = f"""
        Based on this question analysis, generate 3-5 different search queries to find comprehensive information:
        
        Original Question: {question}
        Intent: {analysis.get('intent', 'explanation')}
        Key Terms: {analysis.get('key_terms', [])}
        Domain: {analysis.get('domain', 'general')}
        
        Generate search queries that cover:
        1. Direct keywords from the question
        2. Synonyms and related terms
        3. Broader context queries
        4. Specific detail queries
        
        Provide only the search queries, one per line:
        """
        
        try:
            queries_text = await RobustLLMService.query_llm(query_prompt, max_tokens=200)
            search_queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
            
            # Always include the original question
            if question not in search_queries:
                search_queries.insert(0, question)
                
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            search_queries = [question]
        
        state["search_queries"] = search_queries[:5]  # Limit to 5 queries
        state["workflow_steps"].append({
            "step": "generate_search_queries",
            "result": {"queries": search_queries},
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def retrieve_chunks(self, state: WorkflowState) -> WorkflowState:
        """Retrieve relevant chunks using multiple search queries"""
        collection_name = state["collection_name"]
        search_queries = state["search_queries"]
        
        all_chunks = []
        chunk_ids_seen = set()
        
        for query in search_queries:
            try:
                chunks = await RobustVectorStore.enhanced_search(
                    collection_name, query, limit=config.TOP_K
                )
                
                # Add unique chunks
                for chunk in chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id not in chunk_ids_seen:
                        chunk["search_query"] = query
                        all_chunks.append(chunk)
                        chunk_ids_seen.add(chunk_id)
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue
        
        # Sort by score and take top chunks
        all_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        retrieved_chunks = all_chunks[:config.TOP_K * 2]  # Get more chunks for refinement
        
        state["retrieved_chunks"] = retrieved_chunks
        state["workflow_steps"].append({
            "step": "retrieve_chunks",
            "result": {
                "total_chunks": len(retrieved_chunks),
                "queries_used": len(search_queries)
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def refine_context(self, state: WorkflowState) -> WorkflowState:
        """Refine and filter chunks based on question relevance"""
        question = state["question"]
        analysis = state["question_analysis"]
        retrieved_chunks = state["retrieved_chunks"]
        
        if not retrieved_chunks:
            state["refined_chunks"] = []
            return state
        
        # Score chunks based on question analysis
        scored_chunks = []
        key_terms = set(term.lower() for term in analysis.get("key_terms", []))
        question_terms = set(question.lower().split())
        
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"].lower()
            chunk_terms = set(chunk_text.split())
            
            # Calculate relevance scores
            key_term_matches = len(key_terms.intersection(chunk_terms))
            question_term_matches = len(question_terms.intersection(chunk_terms))
            
            # Boost score based on content relevance
            relevance_boost = 0
            if key_term_matches > 0:
                relevance_boost += key_term_matches * 0.1
            if question_term_matches > 0:
                relevance_boost += question_term_matches * 0.05
            
            # Apply boost to existing score
            boosted_score = chunk.get("score", 0) + relevance_boost
            chunk["refined_score"] = boosted_score
            scored_chunks.append(chunk)
        
        # Sort by refined score and take top chunks
        scored_chunks.sort(key=lambda x: x.get("refined_score", 0), reverse=True)
        refined_chunks = scored_chunks[:config.FINAL_K]
        
        state["refined_chunks"] = refined_chunks
        state["workflow_steps"].append({
            "step": "refine_context",
            "result": {
                "refined_chunks_count": len(refined_chunks),
                "avg_score": sum(c.get("refined_score", 0) for c in refined_chunks) / len(refined_chunks) if refined_chunks else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def generate_answer(self, state: WorkflowState) -> WorkflowState:
        """Generate comprehensive answer using refined context"""
        question = state["question"]
        document_url = state["document_url"]
        analysis = state["question_analysis"]
        refined_chunks = state["refined_chunks"]
        
        if not refined_chunks:
            state["answer"] = "I couldn't find sufficient relevant information in the document to answer this question accurately."
            state["confidence"] = 0.0
            return state
        
        # Build enhanced context
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(refined_chunks, 1):
            page_info = f" (Page {chunk['page']})" if chunk.get('page', 0) > 0 else ""
            section_info = f" - {chunk['section']}" if chunk.get('section') else ""
            score_info = f" [Relevance: {chunk.get('refined_score', 0):.2f}]"
            
            context_parts.append(f"[Context {i}]{page_info}{section_info}{score_info}:\n{chunk['text']}")
            sources.append({
                "chunk_id": chunk.get('chunk_id', i-1),
                "page": chunk.get('page', 0),
                "section": chunk.get('section', ''),
                "confidence": chunk.get('refined_score', 0.0),
                "search_query": chunk.get('search_query', ''),
                "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        context_text = "\n\n".join(context_parts)
        
        # Enhanced prompt based on question analysis
        answer_type = analysis.get("answer_type", "detailed_explanation")
        specificity = analysis.get("specificity", "medium")
        prompt = f"""
        You are an expert document analyst. Provide a PRECISE, CONCISE answer to the question based on the provided context.
        
        DOCUMENT: {document_url}
        QUESTION TYPE: {analysis.get('intent', 'explanation')}
        REQUIRED SPECIFICITY: {specificity}
        
        CONTEXT:
        {context_text}
        
        QUESTION: {question}
        
        INSTRUCTIONS:
        1. Provide a DIRECT, PRECISE answer - get straight to the point
        2. Use EXACT numbers, timeframes, and specific details from the context
        3. Keep the answer CONCISE - avoid unnecessary elaboration
        4. For yes/no questions, start with "Yes" or "No" clearly
        5. For specific facts (periods, amounts, limits), state them directly
        6. Include page references only when specifically relevant
        7. Avoid filler phrases like "according to the document" or "based on the context"
        8. Maximum 2-3 sentences unless complex explanation is absolutely necessary
        
        ANSWER (be precise and concise):
        """        
        
        try:
            answer = await RobustLLMService.query_llm(prompt, max_tokens=config.MAX_TOKENS)
            
            # Calculate confidence based on context quality and coverage
            avg_chunk_score = sum(chunk.get('refined_score', 0) for chunk in refined_chunks) / len(refined_chunks)
            coverage_score = min(len(refined_chunks) / config.FINAL_K, 1.0)
            
            # Adjust confidence based on answer quality indicators
            confidence_adjustments = 0
            answer_lower = answer.lower()
            
            if "according to the document" in answer_lower or "based on the context" in answer_lower:
                confidence_adjustments += 0.1
            if "page" in answer_lower or "section" in answer_lower:
                confidence_adjustments += 0.05
            if "i don't know" in answer_lower or "not mentioned" in answer_lower:
                confidence_adjustments -= 0.3
            if "insufficient information" in answer_lower:
                confidence_adjustments -= 0.2
            
            final_confidence = max(0.0, min(1.0, avg_chunk_score * 0.7 + coverage_score * 0.2 + confidence_adjustments))
            
            state["answer"] = answer
            state["confidence"] = final_confidence
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            state["answer"] = f"Error generating answer: {str(e)}"
            state["confidence"] = 0.0
        
        state["workflow_steps"].append({
            "step": "generate_answer",
            "result": {
                "answer_length": len(state["answer"]),
                "confidence": state["confidence"],
                "sources_used": len(sources)
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    async def evaluate_answer(self, state: WorkflowState) -> WorkflowState:
        """Evaluate answer quality and decide if iteration is needed"""
        answer = state["answer"]
        confidence = state["confidence"]
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 2)
        
        # Simple evaluation criteria
        should_iterate = (
            confidence < 0.6 and  # Low confidence
            iteration_count < max_iterations and  # Haven't hit max iterations
            len(answer) < 100  # Very short answer
        )
        
        state["iteration_count"] = iteration_count + 1
        state["workflow_steps"].append({
            "step": "evaluate_answer",
            "result": {
                "should_iterate": should_iterate,
                "confidence": confidence,
                "iteration": iteration_count + 1
            },
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return state
    
    def should_continue(self, state: WorkflowState) -> str:
        """Decide whether to continue iterating or end"""
        answer = state["answer"]
        confidence = state["confidence"]
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 2)
        
        should_iterate = (
            confidence < 0.6 and
            iteration_count < max_iterations and
            len(answer) < 100
        )
        
        return "continue" if should_iterate else "end"
    
    async def run_workflow(self, question: str, document_url: str, collection_name: str) -> Dict[str, Any]:
        """Run the complete workflow"""
        initial_state = WorkflowState(
            question=question,
            document_url=document_url,
            collection_name=collection_name,
            question_analysis={},
            search_queries=[],
            retrieved_chunks=[],
            refined_chunks=[],
            answer="",
            confidence=0.0,
            workflow_steps=[],
            iteration_count=0,
            max_iterations=2
        )
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            return {
                "answer": final_state["answer"],
                "confidence": final_state["confidence"],
                "sources": [
                    {
                        "chunk_id": chunk.get("chunk_id", 0),
                        "page": chunk.get("page", 0),
                        "section": chunk.get("section", ""),
                        "confidence": chunk.get("refined_score", 0.0),
                        "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                    }
                    for chunk in final_state["refined_chunks"]
                ],
                "context_chunks_used": len(final_state["refined_chunks"]),
                "workflow_steps": final_state["workflow_steps"]
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "answer": f"Error in workflow execution: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "context_chunks_used": 0,
                "workflow_steps": []
            }

# Enhanced RAG System with LangGraph Integration
class EnterpriseRAGWithWorkflow:
    def __init__(self):
        self.workflow = IntelligentRAGWorkflow()
    
    @staticmethod
    async def process_document_pipeline(url: str) -> str:
        """Optimized document processing pipeline"""
        start_time = time.time()
        
        try:
            filename = url.split('/')[-1].split('?')[0]
            logger.info(f"Processing document: {filename}")
            
            cached_metadata = RobustCacheService.get_cached_document_metadata(url)
            
            with DOCUMENT_PROCESSING_TIME.time():
                content = await RobustDocumentProcessor.download_document(url)
                logger.info(f"Downloaded {len(content)} bytes")
                
                text = RobustDocumentProcessor.extract_text(content, url)
                logger.info(f"Extracted {len(text)} characters")
                
                if len(text.strip()) < 100:
                    raise DocumentProcessingError("Document text too short after extraction")
                
                processing_time = time.time() - start_time
                logger.info(f"Document processing completed in {processing_time:.2f}s")
                
                return text
                
        except Exception as e:
            ERROR_COUNT.labels(error_type="document_processing_failed").inc()
            logger.error(f"Document processing failed for {url}: {e}")
            raise DocumentProcessingError(f"Document processing failed: {str(e)}")
    
    @staticmethod
    async def setup_vector_collection(url: str, text: str) -> str:
        """Setup vector collection with intelligent chunking"""
        collection_name = RobustDocumentProcessor.get_collection_name(url)
        filename = url.split('/')[-1].split('?')[0]
        
        try:
            collection_info = RobustCacheService.get_collection_info(collection_name)
            if collection_info and await RobustVectorStore.collection_exists(collection_name):
                logger.info(f"Using existing collection: {collection_name}")
                return collection_name
            
            chunks = RobustDocumentProcessor.create_intelligent_chunks(text)
            logger.info(f"Created {len(chunks)} chunks for vectorization")
            
            if not await RobustVectorStore.collection_exists(collection_name):
                await RobustVectorStore.create_collection(collection_name)
            
            await RobustVectorStore.store_chunks(collection_name, chunks, url)
            
            collection_info = {
                "url": url,
                "chunks_count": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                "total_tokens": sum(chunk["tokens"] for chunk in chunks)
            }
            RobustCacheService.cache_collection_info(collection_name, collection_info)
            
            RobustCacheService.cache_document_metadata(
                url, filename, len(text), len(chunks)
            )
            
            logger.info(f"Vector collection setup completed: {collection_name}")
            return collection_name
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="vector_setup_failed").inc()
            logger.error(f"Vector collection setup failed: {e}")
            raise VectorSearchError(f"Vector collection setup failed: {str(e)}")
    
    async def answer_questions_with_workflow(self, collection_name: str, questions: List[str], document_url: str) -> Dict[str, Any]:
        """Answer multiple questions using the enhanced workflow"""
        results = {
            "answers": [],
            "confidence_scores": [],
            "total_chunks_used": 0,
            "processing_details": [],
            "workflow_details": []
        }
        
        for question in questions:
            try:
                logger.info(f"Processing question with workflow: {question[:100]}...")
                
                # Run the LangGraph workflow
                workflow_result = await self.workflow.run_workflow(
                    question=question,
                    document_url=document_url,
                    collection_name=collection_name
                )
                
                results["answers"].append(workflow_result["answer"])
                results["confidence_scores"].append(workflow_result["confidence"])
                results["total_chunks_used"] += workflow_result.get("context_chunks_used", 0)
                
                results["processing_details"].append({
                    "question": question,
                    "chunks_found": workflow_result.get("context_chunks_used", 0),
                    "confidence": workflow_result["confidence"],
                    "sources": workflow_result.get("sources", [])
                })
                
                results["workflow_details"].append({
                    "question": question,
                    "workflow_steps": workflow_result.get("workflow_steps", [])
                })
                
            except Exception as e:
                ERROR_COUNT.labels(error_type="question_processing_failed").inc()
                logger.error(f"Question processing failed: {e}")
                
                results["answers"].append(f"Error processing question: {str(e)}")
                results["confidence_scores"].append(0.0)
                results["processing_details"].append({
                    "question": question,
                    "error": str(e),
                    "chunks_found": 0,
                    "confidence": 0.0
                })
                results["workflow_details"].append({
                    "question": question,
                    "workflow_steps": [{"error": str(e)}]
                })
        
        return results

    async def process_single_question_with_workflow(
        self, 
        question: str, 
        collection_name: str, 
        document_url: str, 
        question_index: int
    ) -> tuple[int, Dict[str, Any]]:
        """Process a single question and return its index with the result"""
        try:
            logger.info(f"Processing question {question_index + 1}: {question[:100]}...")
            
            # Run the LangGraph workflow
            workflow_result = await self.workflow.run_workflow(
                question=question,
                document_url=document_url,
                collection_name=collection_name
            )
            
            result = {
                "answer": workflow_result["answer"],
                "confidence": workflow_result["confidence"],
                "context_chunks_used": workflow_result.get("context_chunks_used", 0),
                "sources": workflow_result.get("sources", []),
                "workflow_steps": workflow_result.get("workflow_steps", []),
                "question": question,
                "processing_success": True,
                "error": None
            }
            
            logger.info(f"✅ Question {question_index + 1} processed successfully (confidence: {workflow_result['confidence']:.2f})")
            return question_index, result
            
        except Exception as e:
            ERROR_COUNT.labels(error_type="question_processing_failed").inc()
            logger.error(f"❌ Question {question_index + 1} processing failed: {e}")
            
            result = {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "context_chunks_used": 0,
                "sources": [],
                "workflow_steps": [{"error": str(e)}],
                "question": question,
                "processing_success": False,
                "error": str(e)
            }
            
            return question_index, result

    async def answer_questions_with_workflow_parallel(
        self, 
        collection_name: str, 
        questions: List[str], 
        document_url: str,
        max_concurrent_questions: int = 5  # Configurable concurrency limit
    ) -> Dict[str, Any]:
        """Answer multiple questions in parallel using the enhanced workflow"""
        
        # Create semaphore to limit concurrent questions processing
        semaphore = asyncio.Semaphore(max_concurrent_questions)
        
        async def process_with_semaphore(question: str, index: int) -> tuple[int, Dict[str, Any]]:
            """Process question with concurrency control"""
            async with semaphore:
                return await self.process_single_question_with_workflow(
                    question, collection_name, document_url, index
                )
        
        # Start processing all questions concurrently
        logger.info(f"🚀 Starting parallel processing of {len(questions)} questions (max concurrent: {max_concurrent_questions})")
        start_time = time.time()
        
        # Create tasks for all questions
        tasks = [
            process_with_semaphore(question, index) 
            for index, question in enumerate(questions)
        ]
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ Parallel processing completed in {processing_time:.2f}s")
        
        # Initialize results structure
        results = {
            "answers": [""] * len(questions),  # Pre-allocate with correct size
            "confidence_scores": [0.0] * len(questions),
            "total_chunks_used": 0,
            "processing_details": [None] * len(questions),
            "workflow_details": [None] * len(questions),
            "parallel_processing_stats": {
                "total_questions": len(questions),
                "processing_time_seconds": processing_time,
                "max_concurrent": max_concurrent_questions,
                "successful_questions": 0,
                "failed_questions": 0
            }
        }
        
        # Process results and maintain order
        for result in completed_results:
            if isinstance(result, Exception):
                # Handle exception case
                logger.error(f"Task resulted in exception: {result}")
                # Find the first empty slot for this error
                for i in range(len(questions)):
                    if results["answers"][i] == "":
                        results["answers"][i] = f"Error processing question: {str(result)}"
                        results["confidence_scores"][i] = 0.0
                        results["processing_details"][i] = {
                            "question": questions[i] if i < len(questions) else "Unknown",
                            "error": str(result),
                            "chunks_found": 0,
                            "confidence": 0.0
                        }
                        results["workflow_details"][i] = {
                            "question": questions[i] if i < len(questions) else "Unknown",
                            "workflow_steps": [{"error": str(result)}]
                        }
                        results["parallel_processing_stats"]["failed_questions"] += 1
                        break
                continue
            
            # Unpack the result
            question_index, question_result = result
            
            # Ensure the index is valid
            if 0 <= question_index < len(questions):
                # Place result in correct position
                results["answers"][question_index] = question_result["answer"]
                results["confidence_scores"][question_index] = question_result["confidence"]
                results["total_chunks_used"] += question_result.get("context_chunks_used", 0)
                
                results["processing_details"][question_index] = {
                    "question": question_result["question"],
                    "chunks_found": question_result.get("context_chunks_used", 0),
                    "confidence": question_result["confidence"],
                    "sources": question_result.get("sources", []),
                    "error": question_result.get("error")
                }
                
                results["workflow_details"][question_index] = {
                    "question": question_result["question"],
                    "workflow_steps": question_result.get("workflow_steps", [])
                }
                
                if question_result["processing_success"]:
                    results["parallel_processing_stats"]["successful_questions"] += 1
                else:
                    results["parallel_processing_stats"]["failed_questions"] += 1
            else:
                logger.warning(f"Invalid question index received: {question_index}")
                results["parallel_processing_stats"]["failed_questions"] += 1
        
        # Calculate additional stats
        if results["confidence_scores"]:
            avg_confidence = sum(results["confidence_scores"]) / len(results["confidence_scores"])
            results["parallel_processing_stats"]["average_confidence"] = avg_confidence
        
        results["parallel_processing_stats"]["questions_per_second"] = len(questions) / processing_time if processing_time > 0 else 0
        
        logger.info(f"📊 Parallel processing stats: {results['parallel_processing_stats']['successful_questions']}/{len(questions)} successful, avg confidence: {results['parallel_processing_stats'].get('average_confidence', 0):.2f}")
        
        return results

    # Keep the original sequential method as backup
    async def answer_questions_with_workflow_sequential(self, collection_name: str, questions: List[str], document_url: str) -> Dict[str, Any]:
        """Original sequential processing method (kept as backup)"""
        results = {
            "answers": [],
            "confidence_scores": [],
            "total_chunks_used": 0,
            "processing_details": [],
            "workflow_details": []
        }
        
        for question in questions:
            try:
                logger.info(f"Processing question with workflow: {question[:100]}...")
                
                # Run the LangGraph workflow
                workflow_result = await self.workflow.run_workflow(
                    question=question,
                    document_url=document_url,
                    collection_name=collection_name
                )
                
                results["answers"].append(workflow_result["answer"])
                results["confidence_scores"].append(workflow_result["confidence"])
                results["total_chunks_used"] += workflow_result.get("context_chunks_used", 0)
                
                results["processing_details"].append({
                    "question": question,
                    "chunks_found": workflow_result.get("context_chunks_used", 0),
                    "confidence": workflow_result["confidence"],
                    "sources": workflow_result.get("sources", [])
                })
                
                results["workflow_details"].append({
                    "question": question,
                    "workflow_steps": workflow_result.get("workflow_steps", [])
                })
                
            except Exception as e:
                ERROR_COUNT.labels(error_type="question_processing_failed").inc()
                logger.error(f"Question processing failed: {e}")
                
                results["answers"].append(f"Error processing question: {str(e)}")
                results["confidence_scores"].append(0.0)
                results["processing_details"].append({
                    "question": question,
                    "error": str(e),
                    "chunks_found": 0,
                    "confidence": 0.0
                })
                results["workflow_details"].append({
                    "question": question,
                    "workflow_steps": [{"error": str(e)}]
                })
        
        return results

# Service initialization with health checks
async def initialize_services():
    global qdrant_client, bedrock_client, session, tokenizer, embedding_model, service_status
    
    logger.info("Initializing services...")
    
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("✅ Tokenizer initialized")
    except Exception as e:
        logger.error(f"❌ Tokenizer initialization failed: {e}")
        raise
    
    # Initialize local embedding model
    try:
        logger.info(f"Loading local embedding model: {config.EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        # Test the model
        test_embedding = embedding_model.encode(["test"], convert_to_numpy=True)
        logger.info(f"✅ Local embedding model initialized (dimension: {test_embedding.shape[1]})")
        service_status.embedding_model = True
    except Exception as e:
        logger.error(f"❌ Local embedding model initialization failed: {e}")
        ERROR_COUNT.labels(error_type="embedding_model_init").inc()
    
    # Initialize Qdrant
    try:
        qdrant_client = AsyncQdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=config.TIMEOUT
        )
        await qdrant_client.get_collections()
        service_status.qdrant = True
        logger.info("✅ Qdrant connected with gRPC")
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        ERROR_COUNT.labels(error_type="qdrant_init").inc()
    
    # Initialize Bedrock
    try:
        # Create Bedrock Runtime client
        global bedrock_client
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

        # Create Bedrock Models client (optional, just for verifying model access)
        bedrock_models_client = boto3.client(
            'bedrock',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

        # Check model availability (optional but helpful)
        models_response = bedrock_models_client.list_foundation_models()
        available_models = [m["modelId"] for m in models_response["modelSummaries"]]

        logger.info(f"✅ Found {len(available_models)} Bedrock models.")
        if config.DEEPSEEK_MODEL not in available_models:
            logger.warning(f"⚠️ DeepSeek model '{config.DEEPSEEK_MODEL}' not found in Bedrock model list.")

        # Optional: test DeepSeek with dummy prompt
        test_prompt = "<｜begin_of_sentence｜><｜User｜>Hello｜Assistant｜>"
        request_body = {
            "prompt": test_prompt,
            "max_tokens": 5,
            "temperature": 0.5,
            "top_p": 0.9
        }

        response = bedrock_client.invoke_model(
            modelId=config.DEEPSEEK_MODEL,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response['body'].read())
        logger.info(f"✅ DeepSeek test response: {result}")

        # Mark Bedrock as available
        service_status.bedrock = True
        logger.info("✅ Bedrock client initialized and DeepSeek model is usable.")

    except (ClientError, BotoCoreError) as e:
        logger.exception("❌ Bedrock initialization failed:")
        service_status.bedrock = False
        ERROR_COUNT.labels(error_type="bedrock_init").inc()
    
    # Initialize HTTP session
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=config.TIMEOUT),
        headers={"User-Agent": "Enterprise-RAG-System/5.0.0"}
    )
    
    service_status.last_check = datetime.utcnow()
    logger.info("🚀 All services initialized")

async def cleanup_services():
    global qdrant_client, session
    
    if session:
        await session.close()
    
    if qdrant_client:
        await qdrant_client.close()
    
    logger.info("Services cleaned up")

# Initialize services with health checks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_services()
    yield
    # Shutdown
    await cleanup_services()

app = FastAPI(
    title="Enterprise RAG System with Local Embeddings",
    version="5.0.0",
    description="Production-ready RAG system with local SentenceTransformer embeddings and intelligent workflow",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication
async def verify_token(authorization: str = Header(...)):
    if not authorization or authorization != config.AUTH_TOKEN:
        ERROR_COUNT.labels(error_type="auth_failed").inc()
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return True

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with service status"""
    uptime = time.time() - start_time
    
    current_time = datetime.utcnow()
    if (current_time - service_status.last_check).seconds > 300:
        service_status.last_check = current_time
        
        try:
            if qdrant_client:
                await qdrant_client.get_collections()
                service_status.qdrant = True
        except:
            service_status.qdrant = False
    
# Continuing from where the code was cut off...

    overall_status = "healthy" if all([
        service_status.embedding_model,
        service_status.qdrant,
        service_status.bedrock
    ]) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services={
            "embedding_model": service_status.embedding_model,
            "qdrant": service_status.qdrant,
            "bedrock": service_status.bedrock,
            "session": session is not None
        },
        uptime=uptime,
        version="5.0.0",
        timestamp=current_time
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Updated query endpoint to use parallel processing
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_token),
    client_request: Request = None
):
    """Main RAG query endpoint with parallel question processing"""
    
    start_time = time.time()
    client_id = client_request.client.host if client_request else "unknown"
    
    # Rate limiting
    if not check_rate_limit(client_id, config.RATE_LIMIT_PER_MINUTE):
        ERROR_COUNT.labels(error_type="rate_limit_exceeded").inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Track active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        REQUEST_COUNT.labels(method="POST", endpoint="/query", status="processing").inc()
        
        logger.info(f"Processing query for {len(request.questions)} questions from document: {request.documents}")
        
        # Initialize RAG system with workflow
        rag_system = EnterpriseRAGWithWorkflow()
        
        # Step 1: Process document
        try:
            document_text = await rag_system.process_document_pipeline(str(request.documents))
        except DocumentProcessingError as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/query", status="document_error").inc()
            raise HTTPException(status_code=422, detail=f"Document processing failed: {str(e)}")
        
        # Step 2: Setup vector collection
        try:
            collection_name = await rag_system.setup_vector_collection(str(request.documents), document_text)
        except VectorSearchError as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/query", status="vector_error").inc()
            raise HTTPException(status_code=500, detail=f"Vector processing failed: {str(e)}")
        
        # Step 3: Answer questions using parallel or sequential processing
        try:
            if config.ENABLE_PARALLEL_PROCESSING and len(request.questions) > 1:
                logger.info(f"Using parallel processing for {len(request.questions)} questions")
                results = await rag_system.answer_questions_with_workflow_parallel(
                    collection_name, 
                    request.questions, 
                    str(request.documents),
                    max_concurrent_questions=config.MAX_CONCURRENT_QUESTIONS
                )
            else:
                logger.info(f"Using sequential processing for {len(request.questions)} questions")
                results = await rag_system.answer_questions_with_workflow_sequential(
                    collection_name, request.questions, str(request.documents)
                )
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/query", status="query_error").inc()
            raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")
        
        # Cleanup expired cache entries in background
        background_tasks.add_task(clean_expired_cache)
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.observe(processing_time)
        REQUEST_COUNT.labels(method="POST", endpoint="/query", status="success").inc()
        
        avg_confidence = sum(results['confidence_scores']) / len(results['confidence_scores']) if results['confidence_scores'] else 0
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        logger.info(f"Average confidence: {avg_confidence:.2f}")
        
        # Log parallel processing stats if available
        if 'parallel_processing_stats' in results:
            stats = results['parallel_processing_stats']
            logger.info(f"Parallel processing: {stats['successful_questions']}/{stats['total_questions']} successful, {stats['questions_per_second']:.1f} questions/sec")
        
        return QueryResponse(answers=results["answers"])
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type="unexpected_error").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/query", status="error").inc()
        logger.error(f"Unexpected error in query processing: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")
    
    finally:
        ACTIVE_CONNECTIONS.dec()

@app.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_token),
    client_request: Request = None
):
    """Alias endpoint for /hackrx/run with parallel processing"""
    return await process_query(request, background_tasks, _, client_request)


@app.get("/collections")
async def list_collections(_: bool = Depends(verify_token)):
    """List all vector collections"""
    try:
        if not service_status.qdrant:
            raise HTTPException(status_code=503, detail="Qdrant service unavailable")
        
        collections = await qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "vectors_count": col.vectors_count,
                    "segments_count": col.segments_count,
                    "status": col.status
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        ERROR_COUNT.labels(error_type="collection_list_failed").inc()
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve collections")

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, _: bool = Depends(verify_token)):
    """Delete a vector collection"""
    try:
        if not service_status.qdrant:
            raise HTTPException(status_code=503, detail="Qdrant service unavailable")
        
        await qdrant_client.delete_collection(collection_name)
        
        # Clear cache entries
        cache_key = get_cache_key("collection", collection_name)
        if cache_key in in_memory_cache:
            del in_memory_cache[cache_key]
        
        logger.info(f"Deleted collection: {collection_name}")
        return {"message": f"Collection {collection_name} deleted successfully"}
        
    except Exception as e:
        ERROR_COUNT.labels(error_type="collection_delete_failed").inc()
        logger.error(f"Failed to delete collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@app.get("/cache/stats")
async def cache_stats(_: bool = Depends(verify_token)):
    """Get cache statistics"""
    total_entries = len(in_memory_cache)
    current_time = time.time()
    
    expired_count = 0
    cache_size = 0
    
    for key, entry in in_memory_cache.items():
        if current_time - entry['timestamp'] > CACHE_EXPIRY_SECONDS:
            expired_count += 1
        cache_size += len(str(entry['data']))
    
    return {
        "total_entries": total_entries,
        "expired_entries": expired_count,
        "active_entries": total_entries - expired_count,
        "cache_size_bytes": cache_size,
        "cache_hit_ratio": "N/A (not tracked in this implementation)"
    }

@app.post("/cache/clear")
async def clear_cache(_: bool = Depends(verify_token)):
    """Clear all cache entries"""
    global in_memory_cache
    entries_cleared = len(in_memory_cache)
    in_memory_cache.clear()
    
    logger.info(f"Cache cleared: {entries_cleared} entries removed")
    return {"message": f"Cache cleared successfully. {entries_cleared} entries removed."}

@app.get("/service/status")
async def service_status_endpoint(_: bool = Depends(verify_token)):
    """Detailed service status"""
    try:
        # Test services
        qdrant_status = False
        qdrant_error = None
        try:
            await qdrant_client.get_collections()
            qdrant_status = True
        except Exception as e:
            qdrant_error = str(e)
        
        bedrock_status = False
        bedrock_error = None
        try:
            # Test with a simple list models call
            bedrock_models_client = boto3.client(
                'bedrock',
                region_name=config.AWS_REGION,
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
            )
            bedrock_models_client.list_foundation_models()
            bedrock_status = True
        except Exception as e:
            bedrock_error = str(e)
        
        embedding_status = service_status.embedding_model
        embedding_error = None
        if not embedding_status:
            embedding_error = "Local embedding model not loaded"
        
        return {
            "services": {
                "qdrant": {
                    "status": qdrant_status,
                    "url": config.QDRANT_URL,
                    "error": qdrant_error
                },
                "bedrock": {
                    "status": bedrock_status,
                    "region": config.AWS_REGION,
                    "model": config.DEEPSEEK_MODEL,
                    "error": bedrock_error
                },
                "embedding_model": {
                    "status": embedding_status,
                    "model_name": config.EMBEDDING_MODEL_NAME,
                    "error": embedding_error
                },
                "session": {
                    "status": session is not None,
                    "error": None if session else "HTTP session not initialized"
                }
            },
            "configuration": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k": config.TOP_K,
                "final_k": config.FINAL_K,
                "min_score": config.MIN_SCORE,
                "max_tokens": config.MAX_TOKENS,
                "batch_size": config.BATCH_SIZE,
                "rate_limit": config.RATE_LIMIT_PER_MINUTE
            },
            "memory_usage": {
                "cache_entries": len(in_memory_cache),
                "rate_limit_entries": len(rate_limit_store)
            },
            "last_check": service_status.last_check.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service status check failed: {str(e)}")

# Error handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_error_handler(request: Request, exc: DocumentProcessingError):
    ERROR_COUNT.labels(error_type="document_processing").inc()
    return HTTPException(status_code=422, detail=str(exc))

@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError):
    ERROR_COUNT.labels(error_type="embedding_error").inc()
    return HTTPException(status_code=503, detail=f"Embedding service error: {str(exc)}")

@app.exception_handler(VectorSearchError)
async def vector_search_error_handler(request: Request, exc: VectorSearchError):
    ERROR_COUNT.labels(error_type="vector_search").inc()
    return HTTPException(status_code=503, detail=f"Vector search error: {str(exc)}")

@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    ERROR_COUNT.labels(error_type="llm_error").inc()
    return HTTPException(status_code=503, detail=f"LLM service error: {str(exc)}")

# Background task for periodic cleanup
async def periodic_cleanup():
    """Periodic cleanup of cache and rate limit store"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            clean_expired_cache()
            clean_rate_limit_store()
            logger.info("Periodic cleanup completed")
        except Exception as e:
            logger.error(f"Periodic cleanup failed: {e}")

# Add the background task to startup
@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(periodic_cleanup())

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Enterprise RAG System with Local Embeddings")
    logger.info(f"Embedding Model: {config.EMBEDDING_MODEL_NAME}")
    logger.info(f"Vector Store: {config.QDRANT_URL}")
    logger.info(f"LLM Model: {config.DEEPSEEK_MODEL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False,
        workers=1  # Single worker to avoid issues with global state
    )

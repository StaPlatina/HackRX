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
from pydantic import BaseModel
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
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
    # Pinecone Config
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Document Processing
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    MIN_CHUNK_LENGTH = 100
    
    # Agent Config - Updated for Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Fast and cost-effective
    MAX_AGENT_TOKENS = 4000
    
    # Retrieval Config
    TOP_K_RETRIEVAL = 15
    TOP_K_FOR_AGENT = 10
    
    # System Config
    REDIS_HOST = os.getenv("REDIS_HOST", "")  # Empty by default
    REDIS_PORT = int(os.getenv("REDIS_PORT", 0)) if os.getenv("REDIS_PORT") else 0  # 0 by default
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
def get_gemini_client():
    logger.info("Initializing Gemini client...")
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    return model

@lru_cache(maxsize=1)
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")

@lru_cache(maxsize=1)
def get_pinecone_client():
    logger.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return pc

# Initialize services
embedding_model = get_embedding_model()
gemini_model = get_gemini_client()
tokenizer = get_tokenizer()
pinecone_client = get_pinecone_client()

# Redis client - Optional for caching
redis_client = None
if config.REDIS_HOST and config.REDIS_PORT:
    try:
        redis_pool = redis.ConnectionPool(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            max_connections=20,
            decode_responses=True
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        logger.info("Redis connected successfully - caching enabled")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} - running without cache")
        redis_client = None
else:
    logger.info("Redis not configured - running without cache")

# Test Pinecone connection and check available regions
try:
    indexes = pinecone_client.list_indexes()
    logger.info("Pinecone connected successfully")
    
    # Log available regions for debugging
    try:
        # This will help us see what regions are actually available
        test_spec = ServerlessSpec(cloud="aws", region="us-east-1")
        logger.info("Pinecone AWS regions appear to be available")
    except Exception as region_check:
        logger.warning(f"AWS regions might not be available: {region_check}")
        try:
            test_spec = ServerlessSpec(cloud="gcp", region="us-central1-gcp") 
            logger.info("Pinecone GCP regions appear to be available")
        except Exception as gcp_check:
            logger.warning(f"GCP regions check: {gcp_check}")
            
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
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

# Request/Response Models - Updated for compatibility
class QueryRequest(BaseModel):
    documents: str  # Changed from HttpUrl to str for compatibility
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
    def generate_index_name(url: str) -> str:
        """Generate Pinecone index name from URL"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        # Pinecone index names must be lowercase and alphanumeric with hyphens
        return f"doc-{url_hash}"
    
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
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        logger.info(f"Chunking text of length: {len(text)} characters")
        
        # Clean the text first
        text = text.replace('\x00', '')  # Remove null bytes
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split into sentences more aggressively
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.info(f"Split into {len(sentences)} sentences")
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_tokens = len(tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > config.CHUNK_SIZE and current_chunk:
                # Extract page number if present
                page_match = re.search(r'\[Page (\d+)\]', current_chunk)
                page_num = int(page_match.group(1)) if page_match else 0
                
                # Clean chunk text
                clean_chunk = re.sub(r'\[Page \d+\]', '', current_chunk).strip()
                clean_chunk = re.sub(r'\s+', ' ', clean_chunk)  # Normalize whitespace
                
                if len(clean_chunk) >= config.MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": clean_chunk,
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "tokens": current_tokens
                    })
                    chunk_id += 1
                    logger.debug(f"Created chunk {chunk_id}: {len(clean_chunk)} chars, page {page_num}")
                
                # Start new chunk with overlap
                if config.CHUNK_OVERLAP > 0 and len(current_chunk.split()) > config.CHUNK_OVERLAP:
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
        if current_chunk.strip():
            page_match = re.search(r'\[Page (\d+)\]', current_chunk)
            page_num = int(page_match.group(1)) if page_match else 0
            clean_chunk = re.sub(r'\[Page \d+\]', '', current_chunk).strip()
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            
            if len(clean_chunk) >= config.MIN_CHUNK_LENGTH:
                chunks.append({
                    "text": clean_chunk,
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "tokens": current_tokens
                })
        
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Log sample chunks for debugging
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Sample chunk {i+1}: {chunk['text'][:100]}... (page {chunk['page_number']})")
        
        return chunks

# Vector Storage with Pinecone
class VectorStorage:
    @staticmethod
    def index_exists(index_name: str) -> bool:
        """Check if Pinecone index exists"""
        try:
            indexes = pinecone_client.list_indexes()
            return any(idx.name == index_name for idx in indexes)
        except:
            return False
    
    @staticmethod
    def create_index(index_name: str):
        """Create Pinecone index"""
        try:
            # Try different region formats based on Pinecone's current offerings
            regions_to_try = [
                "us-east-1",
                "us-west-2", 
                "eu-west-1",
                "us-central1-gcp",
                "us-east1-gcp"
            ]
            
            for region in regions_to_try:
                try:
                    logger.info(f"Attempting to create index in region: {region}")
                    pinecone_client.create_index(
                        name=index_name,
                        dimension=config.EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws" if "us-east" in region or "us-west" in region or "eu-west" in region else "gcp",
                            region=region
                        )
                    )
                    
                    # Wait for index to be ready
                    import time
                    max_retries = 60
                    for _ in range(max_retries):
                        try:
                            status = pinecone_client.describe_index(index_name).status
                            if status.get('ready', False):
                                break
                        except:
                            pass
                        time.sleep(1)
                    
                    logger.info(f"Successfully created Pinecone index: {index_name} in region: {region}")
                    return
                    
                except Exception as region_error:
                    logger.warning(f"Failed to create index in region {region}: {region_error}")
                    continue
            
            # If all regions fail, raise the last error
            raise Exception("Failed to create index in any available region")
            
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            raise
    
    @staticmethod
    async def store_embeddings(index_name: str, chunks: List[Dict[str, Any]], document_url: str):
        """Store chunk embeddings in Pinecone"""
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            thread_pool,
            lambda: embedding_model.encode(chunk_texts, show_progress_bar=False).tolist()
        )
        
        # Get index
        index = pinecone_client.Index(index_name)
        
        # Prepare vectors for upsert
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{hashlib.sha256(chunk['text'].encode()).hexdigest()[:16]}_{i}"
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_id": chunk.get("chunk_id", i),
                    "page_number": chunk.get("page_number", 0),
                    "document_url": document_url
                }
            })
        
        # Upsert in batches (Pinecone has a limit of 100 vectors per batch)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        logger.info(f"Stored {len(vectors)} embeddings in Pinecone index: {index_name}")

# Multi-Agent System
class MultiAgentSystem:
    
    @staticmethod
    async def retrieval_agent(query: str, index_name: str) -> List[Dict[str, Any]]:
        """Agent 1: Intelligent retrieval from Pinecone vector database"""
        try:
            logger.info(f"Retrieval agent processing query: {query[:100]}...")
            
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                thread_pool,
                lambda: embedding_model.encode([query], show_progress_bar=False)[0].tolist()
            )
            
            logger.info(f"Generated embedding vector with dimension: {len(query_embedding)}")
            
            # Get index and search
            index = pinecone_client.Index(index_name)
            
            # First, check if index has any vectors
            try:
                index_stats = index.describe_index_stats()
                logger.info(f"Index stats: {index_stats}")
                
                if index_stats.total_vector_count == 0:
                    logger.warning("Index is empty - no vectors found!")
                    return []
                    
            except Exception as stats_error:
                logger.warning(f"Could not get index stats: {stats_error}")
            
            # Perform the search
            search_result = index.query(
                vector=query_embedding,
                top_k=config.TOP_K_RETRIEVAL,
                include_metadata=True,
                include_values=False  # Don't need the vector values back
            )
            
            logger.info(f"Pinecone returned {len(search_result.matches)} matches")
            
            # Debug: Log top matches
            for i, match in enumerate(search_result.matches[:3]):
                logger.info(f"Match {i+1}: Score={match.score:.4f}, Text preview: {match.metadata.get('text', '')[:100]}...")
            
            # Convert to chunk format with better filtering
            chunks = []
            for match in search_result.matches:
                text = match.metadata.get("text", "").strip()
                score = float(match.score)
                
                # Filter out empty or very short chunks, and very low scores
                if text and len(text) > 20 and score > 0.1:  # Lowered threshold
                    chunks.append({
                        "text": text,
                        "score": score,
                        "page_number": match.metadata.get("page_number", 0),
                        "chunk_id": match.metadata.get("chunk_id", 0)
                    })
            
            logger.info(f"Filtered to {len(chunks)} relevant chunks")
            
            # Sort by score (Pinecone returns highest scores first for cosine similarity)
            chunks.sort(key=lambda x: x['score'], reverse=True)
            
            # Return top chunks
            final_chunks = chunks[:config.TOP_K_FOR_AGENT]
            logger.info(f"Returning {len(final_chunks)} chunks to analysis agent")
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Retrieval agent error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    @staticmethod
    async def analysis_agent(chunks: List[Dict[str, Any]], question: str) -> str:
        """Agent 2: Analyze chunks and extract answer"""
        logger.info(f"Analysis agent received {len(chunks)} chunks for question: {question[:50]}...")
        
        if not chunks:
            return "No relevant information found in the document."
        
        # Log chunk info for debugging
        total_text_length = sum(len(chunk['text']) for chunk in chunks)
        logger.info(f"Total text length in chunks: {total_text_length} characters")
        
        # Prepare context with better formatting
        context_parts = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text'].strip()
            if chunk_text:
                context_parts.append(f"[Excerpt {i+1}, Page {chunk.get('page_number', 'N/A')}]:\n{chunk_text}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Prepared context length: {len(context)} characters")
        
        # More direct prompt for Gemini
        prompt = f"""You are an expert document analyst. Analyze the following document excerpts and provide a comprehensive answer to the question.

Document Excerpts:
{context}

Question: {question}

Instructions:
- Read through ALL the excerpts carefully
- Extract ANY information that relates to the question
- Include specific numbers, dates, timeframes, and conditions mentioned
- Be thorough and detailed in your response
- If you find the answer in the excerpts, provide it clearly
- If multiple conditions exist, list them all
- Only use information from the provided excerpts

Answer:"""
        
        try:
            logger.info("Sending prompt to LLM...")
            response = await MultiAgentSystem._query_llm(prompt, agent_type=AgentType.ANALYSIS_AGENT, max_tokens=2000)
            
            cleaned_response = MultiAgentSystem._clean_response(response)
            logger.info(f"Analysis agent response length: {len(cleaned_response)} characters")
            logger.info(f"Analysis preview: {cleaned_response[:200]}...")
            
            return cleaned_response
        except Exception as e:
            logger.error(f"Analysis agent error: {e}")
            import traceback
            logger.error(f"Analysis agent traceback: {traceback.format_exc()}")
            return f"Unable to analyze the information due to error: {str(e)}"
    
    @staticmethod
    async def formatting_agent(raw_answer: str, question: str) -> str:
        """Agent 3: Format and make answer concise"""
        prompt = f"""You are a Formatting Agent. Your job is to take an analytical answer and make it concise and well-formatted.

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

Provide the formatted answer:"""
        
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
    async def orchestrator_agent(questions: List[str], index_name: str) -> List[str]:
        """Agent 4: Orchestrate the entire Q&A process"""
        answers = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:50]}...")
            
            try:
                # Step 1: Retrieval Agent
                chunks = await MultiAgentSystem.retrieval_agent(question, index_name)
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
        """Query Gemini LLM for specific agent"""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.1 if agent_type == AgentType.FORMATTING_AGENT else 0.3,
                top_p=0.9,
            )
            
            # Use executor to run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                agent_pool,
                lambda: gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini")
                return ""
            
        except Exception as e:
            logger.error(f"Gemini query error for {agent_type.value}: {e}")
            # Handle potential quota/rate limit errors
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                await asyncio.sleep(1)  # Brief delay for rate limits
            raise
    
    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean Gemini response"""
        if not response:
            return "Unable to generate response."
        
        # Remove any markdown formatting that might interfere
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)  # Remove bold
        response = re.sub(r'\*(.*?)\*', r'\1', response)      # Remove italic
        
        # Clean whitespace
        response = ' '.join(response.split())
        
        return response.strip()

# Cache Service
class CacheService:
    @staticmethod
    def get_answer_cache(index_name: str, question: str) -> Optional[str]:
        """Get cached answer"""
        if not redis_client:
            return None
        try:
            cache_key = f"ma_answer:{index_name}:{hashlib.md5(question.encode()).hexdigest()}"
            return redis_client.get(cache_key)
        except:
            return None
    
    @staticmethod
    def set_answer_cache(index_name: str, question: str, answer: str):
        """Cache answer"""
        if not redis_client:
            return
        try:
            cache_key = f"ma_answer:{index_name}:{hashlib.md5(question.encode()).hexdigest()}"
            redis_client.setex(cache_key, config.CACHE_EXPIRY, answer)
        except:
            pass
    
    @staticmethod
    def get_index_cache(index_name: str) -> Optional[bool]:
        """Check if index is processed"""
        if not redis_client:
            return None
        try:
            return redis_client.get(f"ma_index:{index_name}")
        except:
            return None
    
    @staticmethod
    def set_index_cache(index_name: str):
        """Mark index as processed"""
        if not redis_client:
            return
        try:
            redis_client.setex(f"ma_index:{index_name}", config.CACHE_EXPIRY, "1")
        except:
            pass

# Main API Endpoint - Updated path for compatibility
@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    authorized: bool = Depends(verify_token)
):
    """Process document and answer questions using multi-agent system"""
    start_time = time.time()
    
    try:
        document_url = request.documents  # Now it's a string
        questions = request.questions
        index_name = DocumentProcessor.generate_index_name(document_url)
        
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        logger.info(f"Using Pinecone index: {index_name}")
        
        # Check if document is already processed
        index_exists = VectorStorage.index_exists(index_name)
        is_cached = CacheService.get_index_cache(index_name)
        
        if not index_exists or not is_cached:
            logger.info("Processing new document...")
            
            # Create index if needed
            if not index_exists:
                VectorStorage.create_index(index_name)
            
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
            
            # Store in Pinecone
            await VectorStorage.store_embeddings(index_name, chunks, document_url)
            
            # Mark as processed
            CacheService.set_index_cache(index_name)
        else:
            logger.info("Using existing document embeddings from Pinecone")
        
        # Process questions with multi-agent system
        answers = []
        
        for i, question in enumerate(questions):
            # Check cache first
            cached_answer = CacheService.get_answer_cache(index_name, question)
            if cached_answer:
                logger.info(f"Using cached answer for question {i+1}")
                answers.append(cached_answer)
                continue
            
            logger.info(f"Processing question {i+1}: {question[:50]}...")
            
            # Use multi-agent system
            agent_answers = await MultiAgentSystem.orchestrator_agent([question], index_name)
            answer = agent_answers[0] if agent_answers else "Unable to process question."
            
            # Cache the answer
            CacheService.set_answer_cache(index_name, question, answer)
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
    pinecone_status = "connected"
    try:
        pinecone_client.list_indexes()
    except:
        pinecone_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "pinecone": pinecone_status,
            "redis": "connected" if redis_client else "disconnected",
            "gemini": "available" if config.GEMINI_API_KEY else "not configured"
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
            "storage": "Pinecone Vector Database",
            "agents": {
                "1_retrieval": "Intelligently retrieves relevant chunks from Pinecone",
                "2_analysis": "Analyzes chunks and extracts comprehensive answers",
                "3_formatting": "Formats answers to be concise and well-structured",
                "4_orchestrator": "Coordinates the entire multi-agent workflow"
            },
            "features": [
                "Pinecone vector-based semantic search",
                "Multi-agent processing pipeline",
                "Intelligent answer formatting",
                "Redis caching at multiple levels",
                "Support for PDF and DOCX"
            ]
        }
    }

# Debug endpoints
@app.get("/debug/pinecone-regions")
async def check_pinecone_regions(authorized: bool = Depends(verify_token)):
    """Check available Pinecone regions"""
    regions_info = {
        "aws_regions": [],
        "gcp_regions": [],
        "errors": []
    }
    
    # Test AWS regions
    aws_regions = ["us-east-1", "us-west-2", "eu-west-1"]
    for region in aws_regions:
        try:
            spec = ServerlessSpec(cloud="aws", region=region)
            regions_info["aws_regions"].append(region)
        except Exception as e:
            regions_info["errors"].append(f"AWS {region}: {str(e)}")
    
    # Test GCP regions  
    gcp_regions = ["us-central1-gcp", "us-east1-gcp", "us-west1-gcp"]
    for region in gcp_regions:
        try:
            spec = ServerlessSpec(cloud="gcp", region=region)
            regions_info["gcp_regions"].append(region)
        except Exception as e:
            regions_info["errors"].append(f"GCP {region}: {str(e)}")
    
    return regions_info

@app.post("/debug/test-retrieval")
async def test_retrieval(
    index_name: str,
    query: str,
    authorized: bool = Depends(verify_token)
):
    """Test retrieval from Pinecone directly"""
    try:
        # Check if index exists
        if not VectorStorage.index_exists(index_name):
            return {"error": "Index does not exist", "index_name": index_name}
        
        # Get index stats
        index = pinecone_client.Index(index_name)
        try:
            stats = index.describe_index_stats()
        except Exception as e:
            stats = f"Could not get stats: {e}"
        
        # Test retrieval
        chunks = await MultiAgentSystem.retrieval_agent(query, index_name)
        
        return {
            "index_name": index_name,
            "query": query,
            "index_stats": stats,
            "chunks_retrieved": len(chunks),
            "chunks": [
                {
                    "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "score": chunk["score"],
                    "page_number": chunk["page_number"],
                    "text_length": len(chunk["text"])
                }
                for chunk in chunks[:5]  # Show first 5 chunks
            ]
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/debug/test-agents")
async def test_agents(
    index_name: str,
    question: str,
    authorized: bool = Depends(verify_token)
):
    """Test individual agents"""
    try:
        # Test retrieval
        chunks = await MultiAgentSystem.retrieval_agent(question, index_name)
        
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

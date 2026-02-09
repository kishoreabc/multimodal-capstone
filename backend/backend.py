from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import chromadb
import torch
import clip
from PIL import Image
from sentence_transformers import CrossEncoder
import io
import logging
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# FastAPI App
# -----------------------

app = FastAPI(
    title="Multimodal Search API",
    description="API for text and image search with reranking capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Global Variables
# -----------------------

device = None
clip_model = None
preprocess = None
rerank_model = None
chroma_client = None
collection = None
openai_client = None


# -----------------------
# Pydantic Models
# -----------------------

class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Text query for search")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    use_reranking: bool = Field(True, description="Whether to use reranking")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")

class ImageSearchRequest(BaseModel):
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    use_reranking: bool = Field(False, description="Whether to use reranking (requires query text)")
    query_text: Optional[str] = Field(None, description="Optional text query for reranking")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters")

class SearchResult(BaseModel):
    document: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    distance: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int

class OCRSearchResponse(BaseModel):
    extracted_text: str
    results: List[SearchResult]
    total: int


# -----------------------
# Startup Event
# -----------------------

@app.on_event("startup")
async def startup_event():
    """Load models and database on startup"""
    global device, clip_model, preprocess, rerank_model, chroma_client, collection, openai_client
    
    logger.info("Loading models and database...")
    
    # Load environment variables
    load_dotenv()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    logger.info("CLIP model loaded")
    
    # Load Reranking Model
    rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("Reranking model loaded")
    
    # Initialize OpenAI client for Gemini API
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://apidev.navigatelabsai.com")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. OCR functionality may not work.")
    else:
        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("OpenAI/Gemini client initialized for OCR")
    
    # Load Chroma DB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("jewellary")
    logger.info("ChromaDB collection loaded")
    
    logger.info("Startup complete!")

# -----------------------
# Mount Static Files
# -----------------------

# Serve images from raw_images directory
app.mount("/images", StaticFiles(directory="./raw_images"), name="images")

# -----------------------
# Utility Functions
# -----------------------

def generate_query_embedding_text(text: str):
    """Generate CLIP embedding for text query"""
    tokens = clip.tokenize([text]).to(device)
    
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb.cpu().numpy()[0]


def generate_query_embedding_image(image: Image.Image):
    """Generate CLIP embedding for image query"""
    img = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = clip_model.encode_image(img)
    
    emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb.cpu().numpy()[0]


def chroma_search(query_vector, top_k: int = 20, filters: Optional[Dict] = None):
    """Search ChromaDB with query vector"""
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=filters
    )
    
    return results


def rerank_results(query_text: str, search_results: Dict, top_k: int = 5):
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        query_text: The original query text
        search_results: Results from chroma_search
        top_k: Number of top results to return after reranking
    
    Returns:
        Reranked results in the same format as chroma_search
    """
    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]
    distances = search_results["distances"][0] if "distances" in search_results else None
    
    # Create query-document pairs for reranking
    pairs = [[query_text, doc] for doc in documents]
    
    # Get reranking scores
    scores = rerank_model.predict(pairs)
    
    # Sort by scores (higher is better)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=False)[:top_k]
    
    # Reorder results
    reranked_results = {
        "documents": [[documents[i] for i in sorted_indices]],
        "metadatas": [[metadatas[i] for i in sorted_indices]],
        "scores": [[float(scores[i]) for i in sorted_indices]]
    }
    
    if distances:
        reranked_results["distances"] = [[distances[i] for i in sorted_indices]]
    
    return reranked_results


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from image using Gemini API via OpenAI-compatible interface.
    This method is particularly effective for handwritten text recognition.
    
    Args:
        image: PIL Image object
    
    Returns:
        Extracted text string
    """
    try:
        if not openai_client:
            raise HTTPException(
                status_code=500, 
                detail="OpenAI client not initialized. Please check API key configuration."
            )
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Call Gemini API to extract text
        response = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. Return only the extracted text without any additional explanation or formatting. If the text is handwritten, do your best to interpret it accurately."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
        )
        
        # Extract text from response
        text = response.choices[0].message.content.strip()
        print(f"Extracted text via Gemini API: {text}", flush=True)
        logger.info(f"Extracted text via Gemini API: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


# -----------------------
# API Endpoints
# -----------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multimodal Search API",
        "version": "1.0.0",
        "endpoints": {
            "text_search": "/api/search/text",
            "image_search": "/api/search/image",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([clip_model is not None, rerank_model is not None, collection is not None])
    }


@app.post("/api/search/text", response_model=SearchResponse)
async def search_text(request: TextSearchRequest):
    """
    Search using text query with optional reranking
    """
    try:
        logger.info(f"Text search request: {request.query}")
        
        # Generate embedding
        query_vector = generate_query_embedding_text(request.query)
        
        # Determine how many initial results to fetch
        initial_top_k = request.top_k * 2 if request.use_reranking else request.top_k
        
        # Search ChromaDB
        results = chroma_search(query_vector, top_k=initial_top_k, filters=request.filters)
        
        # Rerank if requested
        if request.use_reranking:
            results = rerank_results(request.query, results, top_k=request.top_k)
        
        # Format response
        search_results = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        scores = results.get("scores", [None] * len(documents))[0] if "scores" in results else [None] * len(documents)
        distances = results.get("distances", [None] * len(documents))[0] if "distances" in results else [None] * len(documents)
        
        for doc, meta, score, distance in zip(documents, metadatas, scores, distances):
            search_results.append(SearchResult(
                document=doc,
                metadata=meta,
                score=score,
                distance=distance
            ))
        
        return SearchResponse(results=search_results, total=len(search_results))
        
    except Exception as e:
        logger.error(f"Error in text search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/search/image", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    use_reranking: bool = Form(False),
    query_text: Optional[str] = Form(None)
):
    """
    Search using image upload with optional reranking
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Generate embedding
        query_vector = generate_query_embedding_image(image)
        
        # Determine how many initial results to fetch
        # Fetch more results initially if reranking is enabled
        initial_top_k = max(top_k * 2, 20) if use_reranking  else top_k
        
        # Search ChromaDB
        results = chroma_search(query_vector, top_k=initial_top_k)
        
        # Rerank if requested
        if use_reranking:
            # Use query_text if provided, otherwise use metadata from first result as fallback
            if query_text:
                rerank_query = query_text
            else:
                # Generate a query from the first result's metadata
                first_metadata = results["metadatas"][0][0] if results["metadatas"][0] else {}
                metadata_parts = []
                for key in ['material', 'category', 'color', 'stone', 'style']:
                    if key in first_metadata and first_metadata[key]:
                        metadata_parts.append(str(first_metadata[key]))
                
                if metadata_parts:
                    rerank_query = " ".join(metadata_parts)
                else:
                    # Fallback to generic query
                    rerank_query = "jewelry item"
                
                logger.info(f"Reranking image search results with generated query: {rerank_query}")
            
            results = rerank_results(rerank_query, results, top_k=top_k)
        else:
            # If no reranking, limit to top_k results
            results["documents"] = [results["documents"][0][:top_k]]
            results["metadatas"] = [results["metadatas"][0][:top_k]]
            if "distances" in results:
                results["distances"] = [results["distances"][0][:top_k]]
        
        # Format response
        search_results = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        scores = results.get("scores", [None] * len(documents))[0] if "scores" in results else [None] * len(documents)
        distances = results.get("distances", [None] * len(documents))[0] if "distances" in results else [None] * len(documents)
        
        # Debug logging
        logger.info(f"Results keys: {results.keys()}")
        logger.info(f"Has scores: {'scores' in results}, Has distances: {'distances' in results}")
        logger.info(f"Scores type: {type(scores)}, Distances type: {type(distances)}")
        logger.info(f"First score: {scores[0] if scores else None}, First distance: {distances[0] if distances else None}")
        
        for doc, meta, score, distance in zip(documents, metadatas, scores, distances):
            search_results.append(SearchResult(
                document=doc,
                metadata=meta,
                score=score,
                distance=distance
            ))
            logger.debug(f"Result: score={score}, distance={distance}")
        
        logger.info(f"Returning {len(search_results)} results")
        return SearchResponse(results=search_results, total=len(search_results))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/search/ocr", response_model=OCRSearchResponse)
async def search_ocr(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    use_reranking: bool = Form(True)
):
    """
    Extract text from image using OCR and search using the extracted text
    """
    try:
        logger.info(f"OCR search request: {file.filename}, top_k={top_k}, use_reranking={use_reranking}")
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Extract text from image using OCR
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text:
            return OCRSearchResponse(
                extracted_text="",
                results=[],
                total=0
            )
        
        # Use extracted text to search
        logger.info(f"Searching with extracted text: {extracted_text}")
        query_vector = generate_query_embedding_text(extracted_text)
        
        # Determine how many initial results to fetch
        # Fetch more results initially if reranking is enabled
        initial_top_k = max(top_k * 2, 20) if use_reranking else top_k
        
        # Search ChromaDB
        results = chroma_search(query_vector, top_k=initial_top_k)
        
        # Rerank if requested
        if use_reranking and extracted_text:
            logger.info(f"Reranking OCR search results, returning top {top_k}")
            results = rerank_results(extracted_text, results, top_k=top_k)
        else:
            # If no reranking, limit to top_k results
            results["documents"] = [results["documents"][0][:top_k]]
            results["metadatas"] = [results["metadatas"][0][:top_k]]
            if "distances" in results:
                results["distances"] = [results["distances"][0][:top_k]]
        
        # Format results
        search_results = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        scores = results.get("scores", [None] * len(documents))[0] if "scores" in results else [None] * len(documents)
        distances = results.get("distances", [None] * len(documents))[0] if "distances" in results else [None] * len(documents)
        
        for doc, meta, score, distance in zip(documents, metadatas, scores, distances):
            search_results.append(SearchResult(
                document=doc,
                metadata=meta,
                score=score,
                distance=distance
            ))
        
        logger.info(f"Returning {len(search_results)} OCR search results")
        return OCRSearchResponse(
            extracted_text=extracted_text,
            results=search_results,
            total=len(search_results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OCR search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR search failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

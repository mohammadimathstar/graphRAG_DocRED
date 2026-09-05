
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from src.rag.rags import RAGBase
from src.evaluation.judge import run_online_judgment
from src.db.manager import init_db
from src.db.ingest import update_trace_feedback, update_trace_judgment
import os
from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    """
    # --- STARTUP LOGIC ---
    print("Initializing database tables...")
    init_db(drop_if_exists=False) 
    print("Database ready!")

    # --- initialize RAGBase ---
    try:
        print("Initializing RAG System...")
        openai_client = OpenAI()
        SYSTEM_PROMPT = """You are an expert knowledge assistant... (your prompt)"""
        
        # Attach the RAG system to the app state so routes can access it
        app.state.rag = RAGBase(
            llm_client=openai_client,
            instructions=SYSTEM_PROMPT,
            model="gpt-5.6-luna"
        )
        print("RAG System ready! App is ready to serve requests.")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        app.state.rag = None
    
    yield # The app runs and handles requests here

    # --- SHUTDOWN LOGIC ---
    print("Shutting down...")

# 2. Initialize FastAPI with the lifespan handler
app = FastAPI(
    title="GraphRAG Production API",
    description="API for querying the Knowledge Graph using Hybrid Agentic RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Serve Static Files (CSS/JS)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# 4. Define Request and Response Schemas
class QueryRequest(BaseModel):
    question: str
    session_id: str = None  # Optional, for tracking multi-turn conversations

class QueryResponse(BaseModel):
    answer: str
    trace_id: str = None
    
class FeedbackRequest(BaseModel):
    trace_id: str
    thumbs_up: bool
    user_feedback: str = None


# 5. Define API Endpoints
#@app.get("/")
#def health_check():
 #   """Simple health check to verify the API is running."""
  #  return {"status": "healthy", "rag_initialized": app.state.rag is not None}

@app.get("/")
def get_ui():
    """Serves the index.html file from the static folder"""
    html_path = static_dir / "index.html"
    return FileResponse(html_path)


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, background_tasks: BackgroundTasks):
    """Handles a user question and returns a RAG-generated answer."""

    # Access the RAG system from the app state
    rag_system = app.state.rag
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    
    try:
        # This calls your RAGBase class, which handles retrieval, generation, and logging!
        response = rag_system.rag(
            query=request.question, 
            session_id=request.session_id
        )
        answer = response["answer"]
        trace_id = response["trace_id"]
        
        # 2. Add Judge to Background Task (User doesn't wait for this!)
        if trace_id:
            background_tasks.add_task(
                run_online_judgment,
                trace_id=trace_id,
                question=request.question,
                answer=answer
            )        
        return QueryResponse(answer=answer, trace_id=trace_id)        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Receives user feedback and updates the database."""
    try:
        update_trace_feedback(
            trace_id=request.trace_id,
            thumbs_up=request.thumbs_up,
            user_feedback=request.user_feedback
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save feedback")



# To run this: 
# uvicorn app:app --reload --port 8000

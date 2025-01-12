from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ArxivEmbedderConfig(BaseModel):
    """Configuration for the embedding model"""
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    embedding_dim: int = 1536

class ArxivRetrieverConfig(BaseModel):
    """Configuration for vector retrieval"""
    type: str = "vector"
    field: str = "embedding"
    k: int = 5

class ArxivStorageConfig(BaseModel):
    """Main configuration for arXiv knowledge base storage"""
    storage_type: str = "db"
    path: str
    schema: Dict[str, Any]
    embedder: ArxivEmbedderConfig
    retriever: ArxivRetrieverConfig
    options: Optional[Dict[str, Any]] = None

class SystemPromptSchema(BaseModel):
    """Schema for LLM system prompts"""
    role: str = "You are a helpful research assistant."
    persona: Optional[Dict[str, Any]] = None

class ArxivPaper(BaseModel):
    """Schema for arXiv paper data"""
    title: str
    summary: str

class InputSchema(BaseModel):
    """Schema for agent input"""
    tool_name: str
    tool_input_data: Any
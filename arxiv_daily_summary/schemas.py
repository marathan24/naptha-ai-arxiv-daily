from pydantic import BaseModel, field_validator
from typing import Dict, Any, List, Optional
from naptha_sdk.schemas import KBConfig  # Naptha's base KBConfig

class ArxivEmbedderConfig(BaseModel):
    model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    embedding_dim: int = 1536

class ArxivRetrieverConfig(BaseModel):
    type: str = "vector"
    field: str = "embedding"
    k: int = 5

class ArxivKBConfig(KBConfig):
    """
    Extends Naptha's KBConfig and adds typed fields
    for `embedder` and `retriever`.
    """
    embedder: Optional[ArxivEmbedderConfig] = None
    retriever: Optional[ArxivRetrieverConfig] = None

    class Config:
        # Let Pydantic accept extra fields from JSON
        extra = "allow"

# You already have these:
class SystemPromptSchema(BaseModel):
    role: str = "You are a helpful research assistant."
    persona: Optional[Dict[str, Any]] = None

class ArxivPaper(BaseModel):
    title: str
    summary: str

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: Any
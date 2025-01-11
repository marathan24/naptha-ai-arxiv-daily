from pydantic import BaseModel
from typing import Dict, Optional, Union, Any

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    role: str = "You are a helpful research assistant."
    persona: Optional[Union[Dict[str, Any], str]] = None

class ArxivPaper(BaseModel):
    title: str
    summary: str

class InputSchema(BaseModel):
    tool_name: str
    tool_input_data: Union[Dict[str, Any], str]

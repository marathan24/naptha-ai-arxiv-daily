import openai
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class ArxivEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            return [item["embedding"] for item in response["data"]]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

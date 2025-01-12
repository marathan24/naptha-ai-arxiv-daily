from openai import OpenAI
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

class ArxivEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data[0]]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

"""
OpenAI embedding backend.
Uses text-embedding-3-small for high-quality multilingual embeddings.
"""

import os
import time
from typing import Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseEmbedder, register_embedder
import config


# Load environment variables
load_dotenv()


@register_embedder("openai")
class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding backend using text-embedding-3-small.
    
    Features:
    - Multilingual support (works with Turkish, English, etc.)
    - Automatic batching for large datasets
    - L2 normalization for cosine similarity
    - Rate limit handling with retries
    """
    
    def __init__(
        self,
        model: str = config.OPENAI_MODEL,
        batch_size: int = config.OPENAI_BATCH_SIZE,
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model: OpenAI model name (default: text-embedding-3-small)
            batch_size: Number of texts per API call (default: 100)
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.batch_size = batch_size
        
        # Get API key from env if not provided
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in .env file or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)
        self._dimension = config.OPENAI_EMBEDDING_DIM
    
    @property
    def name(self) -> str:
        return f"openai_{self.model}"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Embed texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            show_progress: Whether to print progress (for large batches)
            
        Returns:
            np.ndarray of shape (len(texts), dimension), L2-normalized
        """
        if not texts:
            return np.array([]).reshape(0, self.dimension)
        
        all_embeddings = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            if show_progress and n_batches > 1:
                print(f"Embedding batch {batch_num}/{n_batches}...")
            
            # Clean texts (remove empty strings, limit length)
            cleaned_batch = [self._clean_text(t) for t in batch]
            
            # Call API with retry logic
            embeddings = self._embed_batch_with_retry(cleaned_batch)
            all_embeddings.extend(embeddings)
        
        # Stack and normalize
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        return self.normalize(embeddings_array)
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            np.ndarray of shape (dimension,), L2-normalized
        """
        result = self.embed([text], show_progress=False)
        return result[0]
    
    def _embed_batch_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> list[list[float]]:
        """
        Embed a batch with retry logic for rate limits.
        
        Args:
            texts: Batch of texts to embed
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds (doubled each retry)
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                # Sort by index to ensure order matches input
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_msg or "limit" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limited. Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                
                # Re-raise other errors or final retry failure
                raise
        
        raise RuntimeError(f"Failed to embed batch after {max_retries} attempts")
    
    def _clean_text(self, text: str, max_tokens: int = 8000) -> str:
        """
        Clean and truncate text for embedding.
        
        Args:
            text: Raw text
            max_tokens: Approximate max tokens (using char estimate)
            
        Returns:
            Cleaned text
        """
        if not text:
            return " "  # Empty strings cause API errors
        
        # Basic cleaning
        text = text.strip()
        
        # Rough truncation (1 token ≈ 4 chars for English)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text or " "

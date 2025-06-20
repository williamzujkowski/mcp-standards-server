"""
Tokenizer module for accurate token counting
@nist-controls: SI-10, SI-12
@evidence: Accurate token measurement for content optimization
@oscal-component: standards-engine
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenizerType(str, Enum):
    """Available tokenizer implementations"""
    TIKTOKEN = "tiktoken"
    TRANSFORMERS = "transformers"
    APPROXIMATE = "approximate"


class BaseTokenizer(ABC):
    """Base class for tokenizer implementations"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        pass
    
    @abstractmethod
    def split_to_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into chunks of specified token size"""
        pass


class TiktokenTokenizer(BaseTokenizer):
    """
    OpenAI tiktoken-based tokenizer
    @nist-controls: SI-10
    @evidence: Industry-standard token counting
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        try:
            import tiktoken
            self.tiktoken = tiktoken
            self.encoding = tiktoken.encoding_for_model(model)
            self.model = model
            logger.info(f"Initialized tiktoken tokenizer for model: {model}")
        except ImportError:
            logger.error("tiktoken not installed. Install with: pip install tiktoken")
            raise ImportError("tiktoken is required for TiktokenTokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}")
            # Fall back to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.model = "cl100k_base"
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        return self.encoding.decode(tokens)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.decode(truncated_tokens)
        
        # Clean up potential incomplete characters at the end
        # tiktoken might produce incomplete UTF-8 sequences
        return self._clean_truncated_text(truncated_text)
    
    def split_to_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into chunks of specified token size with optional overlap"""
        tokens = self.encode(text)
        chunks = []
        
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size  # No overlap if invalid
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + chunk_size]
            if chunk_tokens:  # Skip empty chunks
                chunk_text = self.decode(chunk_tokens)
                chunks.append(self._clean_truncated_text(chunk_text))
        
        return chunks
    
    def _clean_truncated_text(self, text: str) -> str:
        """Clean up truncated text to ensure valid UTF-8 and complete words"""
        # Remove potential incomplete UTF-8 sequences
        try:
            # Encode and decode to ensure valid UTF-8
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            pass
        
        # Try to truncate at word boundary
        if text and not text[-1].isspace():
            # Find last complete word
            last_space = text.rfind(' ')
            if last_space > len(text) * 0.8:  # Only truncate if we're not losing too much
                text = text[:last_space]
        
        return text.strip()


class TransformersTokenizer(BaseTokenizer):
    """
    HuggingFace transformers-based tokenizer
    @nist-controls: SI-10
    @evidence: ML model compatible token counting
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_name = model_name
            logger.info(f"Initialized transformers tokenizer for model: {model_name}")
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise ImportError("transformers is required for TransformersTokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize transformers tokenizer: {e}")
            raise
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        # Use tokenizer's built-in truncation
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_tokens,
            truncation=True
        )
        return self.decode(encoded)
    
    def split_to_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into chunks of specified token size"""
        # Use tokenizer's built-in functionality
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        
        for i in range(0, len(encoded), step):
            chunk_tokens = encoded[i:i + chunk_size]
            if chunk_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text.strip())
        
        return chunks


class ApproximateTokenizer(BaseTokenizer):
    """
    Simple approximation-based tokenizer (fallback)
    @nist-controls: SI-10
    @evidence: Basic token estimation when proper tokenizers unavailable
    """
    
    def __init__(self, tokens_per_word: float = 1.3, chars_per_token: float = 4.0):
        self.tokens_per_word = tokens_per_word
        self.chars_per_token = chars_per_token
        logger.info("Initialized approximate tokenizer (fallback mode)")
    
    def encode(self, text: str) -> List[int]:
        """Approximate encoding - returns dummy token IDs"""
        # This is a simple approximation - not real tokens
        words = text.split()
        approx_token_count = int(len(words) * self.tokens_per_word)
        return list(range(approx_token_count))
    
    def decode(self, tokens: List[int]) -> str:
        """Cannot decode approximate tokens"""
        raise NotImplementedError("Approximate tokenizer cannot decode tokens")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using approximation"""
        # Use word-based approximation as primary method
        words = len(text.split())
        word_estimate = int(words * self.tokens_per_word)
        
        # Use character-based as secondary check
        char_estimate = int(len(text) / self.chars_per_token)
        
        # Return average of both estimates
        return (word_estimate + char_estimate) // 2
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately fit token limit"""
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Calculate approximate character limit
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.95)  # 95% to ensure under limit
        
        # Truncate at word boundary
        if target_chars >= len(text):
            return text
        
        truncated = text[:target_chars]
        last_space = truncated.rfind(' ')
        if last_space > target_chars * 0.8:
            truncated = truncated[:last_space]
        
        return truncated.strip()
    
    def split_to_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> List[str]:
        """Split text into approximate chunks"""
        # Estimate characters per chunk
        chars_per_chunk = int(chunk_size * self.chars_per_token)
        chunks = []
        
        # Adjust for overlap
        step = chars_per_chunk - int(overlap * self.chars_per_token)
        if step <= 0:
            step = chars_per_chunk
        
        # Split by characters, trying to preserve word boundaries
        for i in range(0, len(text), step):
            chunk = text[i:i + chars_per_chunk]
            
            # Adjust chunk boundaries to word boundaries
            if i > 0 and chunk and not chunk[0].isspace():
                # Find previous space
                space_before = text.rfind(' ', i - 10, i)
                if space_before > 0:
                    chunk = text[space_before:i + chars_per_chunk]
            
            if chunk and i + chars_per_chunk < len(text) and not text[i + chars_per_chunk - 1].isspace():
                # Find next space
                space_after = chunk.rfind(' ')
                if space_after > len(chunk) * 0.8:
                    chunk = chunk[:space_after]
            
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


class TokenizerFactory:
    """
    Factory for creating tokenizer instances
    @nist-controls: SI-10, AC-4
    @evidence: Centralized tokenizer management
    """
    
    _instances: Dict[str, BaseTokenizer] = {}
    
    @classmethod
    def get_tokenizer(
        cls,
        tokenizer_type: TokenizerType = TokenizerType.TIKTOKEN,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseTokenizer:
        """Get or create a tokenizer instance"""
        
        # Create cache key
        cache_key = f"{tokenizer_type}:{model or 'default'}"
        
        # Return cached instance if available
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance
        try:
            if tokenizer_type == TokenizerType.TIKTOKEN:
                tokenizer = TiktokenTokenizer(model or "gpt-3.5-turbo")
            elif tokenizer_type == TokenizerType.TRANSFORMERS:
                tokenizer = TransformersTokenizer(model or "bert-base-uncased")
            elif tokenizer_type == TokenizerType.APPROXIMATE:
                tokenizer = ApproximateTokenizer(**kwargs)
            else:
                raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
            
            cls._instances[cache_key] = tokenizer
            return tokenizer
            
        except ImportError as e:
            logger.warning(f"Failed to create {tokenizer_type} tokenizer: {e}")
            logger.warning("Falling back to approximate tokenizer")
            # Fall back to approximate tokenizer
            tokenizer = ApproximateTokenizer(**kwargs)
            cls._instances[cache_key] = tokenizer
            return tokenizer
    
    @classmethod
    def create_with_fallback(
        cls,
        preferred_types: List[TokenizerType] = None,
        model: Optional[str] = None
    ) -> BaseTokenizer:
        """Create tokenizer with fallback options"""
        if preferred_types is None:
            preferred_types = [
                TokenizerType.TIKTOKEN,
                TokenizerType.TRANSFORMERS,
                TokenizerType.APPROXIMATE
            ]
        
        for tokenizer_type in preferred_types:
            try:
                return cls.get_tokenizer(tokenizer_type, model)
            except Exception as e:
                logger.debug(f"Failed to create {tokenizer_type}: {e}")
                continue
        
        # Final fallback
        return cls.get_tokenizer(TokenizerType.APPROXIMATE)


# Convenience functions
def get_default_tokenizer() -> BaseTokenizer:
    """Get the default tokenizer with automatic fallback"""
    return TokenizerFactory.create_with_fallback()


def count_tokens(text: str, tokenizer: Optional[BaseTokenizer] = None) -> int:
    """Count tokens in text using specified or default tokenizer"""
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    return tokenizer.count_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int, tokenizer: Optional[BaseTokenizer] = None) -> str:
    """Truncate text to token limit using specified or default tokenizer"""
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    return tokenizer.truncate_to_tokens(text, max_tokens)


def split_to_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    tokenizer: Optional[BaseTokenizer] = None
) -> List[str]:
    """Split text into token-sized chunks using specified or default tokenizer"""
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    return tokenizer.split_to_chunks(text, chunk_size, overlap)


# Token budget utilities
class TokenBudget:
    """
    Manages token budget allocation
    @nist-controls: SI-12
    @evidence: Resource allocation management
    """
    
    def __init__(self, total_tokens: int, tokenizer: Optional[BaseTokenizer] = None):
        self.total_tokens = total_tokens
        self.allocated_tokens = 0
        self.allocations: Dict[str, int] = {}
        self.tokenizer = tokenizer or get_default_tokenizer()
    
    def allocate(self, name: str, tokens: int) -> bool:
        """Allocate tokens to a named component"""
        if self.allocated_tokens + tokens > self.total_tokens:
            return False
        
        self.allocations[name] = tokens
        self.allocated_tokens += tokens
        return True
    
    def allocate_for_text(self, name: str, text: str) -> Tuple[bool, int]:
        """Allocate tokens based on actual text"""
        required_tokens = self.tokenizer.count_tokens(text)
        success = self.allocate(name, required_tokens)
        return success, required_tokens
    
    def remaining(self) -> int:
        """Get remaining token budget"""
        return self.total_tokens - self.allocated_tokens
    
    def usage_percentage(self) -> float:
        """Get budget usage percentage"""
        return (self.allocated_tokens / self.total_tokens) * 100 if self.total_tokens > 0 else 0
    
    def get_allocation_summary(self) -> Dict[str, any]:
        """Get summary of token allocations"""
        return {
            "total_budget": self.total_tokens,
            "allocated": self.allocated_tokens,
            "remaining": self.remaining(),
            "usage_percentage": self.usage_percentage(),
            "allocations": self.allocations.copy()
        }
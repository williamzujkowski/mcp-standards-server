"""
Module following best practices
"""
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process data following standards."""
    
    def __init__(self, config: dict):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        required_keys = ['input_path', 'output_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")
    
    def process(self, data: List[dict]) -> List[dict]:
        """Process data with error handling."""
        try:
            logger.info(f"Processing {len(data)} items")
            return [self._transform(item) for item in data]
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def _transform(self, item: dict) -> dict:
        """Transform single item."""
        return {
            'id': item.get('id'),
            'processed': True,
            'timestamp': datetime.now().isoformat()
        }

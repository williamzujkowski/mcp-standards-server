"""
Test Redis Client Module
@nist-controls: SA-11, CA-7
@evidence: Unit tests for Redis client functionality
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.core.redis_client import close_redis_connection, get_redis_client


class TestRedisClient:
    """Test Redis client functionality"""

    @patch.dict(os.environ, {"REDIS_HOST": "test-host", "REDIS_PORT": "6380", "REDIS_DB": "1", "REDIS_PASSWORD": "test-pass"})
    @patch('src.core.redis_client.redis.ConnectionPool')
    @patch('src.core.redis_client.redis.Redis')
    def test_get_redis_client_success(self, mock_redis_class, mock_pool_class):
        """Test successful Redis client connection"""
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client
        
        # Call function
        client = get_redis_client()
        
        # Verify connection pool was created with correct params
        mock_pool_class.assert_called_once_with(
            host="test-host",
            port=6380,
            db=1,
            password="test-pass",
            decode_responses=True,
            max_connections=10,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        # Verify Redis client was created
        mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
        
        # Verify ping was called
        mock_client.ping.assert_called_once()
        
        # Verify client was returned
        assert client == mock_client

    @patch('src.core.redis_client.redis.ConnectionPool')
    @patch('src.core.redis_client.redis.Redis')
    def test_get_redis_client_default_config(self, mock_redis_class, mock_pool_class):
        """Test Redis client with default configuration"""
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client
        
        # Call function
        client = get_redis_client()
        
        # Verify default values were used
        mock_pool_class.assert_called_once()
        call_kwargs = mock_pool_class.call_args[1]
        assert call_kwargs["host"] == "localhost"
        assert call_kwargs["port"] == 6379
        assert call_kwargs["db"] == 0
        assert call_kwargs["password"] is None

    @patch('src.core.redis_client.redis.ConnectionPool')
    @patch('src.core.redis_client.redis.Redis')
    def test_get_redis_client_connection_error(self, mock_redis_class, mock_pool_class):
        """Test Redis client connection failure"""
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        mock_client = Mock()
        mock_client.ping.side_effect = ConnectionError("Connection refused")
        mock_redis_class.return_value = mock_client
        
        # Call function
        client = get_redis_client()
        
        # Verify None was returned on connection error
        assert client is None

    @patch('src.core.redis_client.redis.ConnectionPool')
    @patch('src.core.redis_client.redis.Redis')
    def test_get_redis_client_redis_error(self, mock_redis_class, mock_pool_class):
        """Test Redis client with RedisError"""
        from redis.exceptions import RedisError
        
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool
        
        mock_client = Mock()
        mock_client.ping.side_effect = RedisError("Redis error")
        mock_redis_class.return_value = mock_client
        
        # Call function
        client = get_redis_client()
        
        # Verify None was returned on Redis error
        assert client is None

    @patch('src.core.redis_client.redis.ConnectionPool')
    def test_get_redis_client_pool_creation_error(self, mock_pool_class):
        """Test Redis client with pool creation error"""
        # Setup mock to raise exception
        mock_pool_class.side_effect = Exception("Pool creation failed")
        
        # Call function
        client = get_redis_client()
        
        # Verify None was returned
        assert client is None

    def test_close_redis_connection_success(self):
        """Test successful Redis connection close"""
        # Create mock client
        mock_client = Mock()
        
        # Call function
        close_redis_connection(mock_client)
        
        # Verify close was called
        mock_client.close.assert_called_once()

    def test_close_redis_connection_with_none(self):
        """Test close with None client"""
        # Should not raise exception
        close_redis_connection(None)

    @patch('src.core.redis_client.logger')
    def test_close_redis_connection_error(self, mock_logger):
        """Test Redis connection close with error"""
        # Create mock client that raises on close
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close failed")
        
        # Call function
        close_redis_connection(mock_client)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
        assert "Error closing Redis connection" in mock_logger.error.call_args[0][0]
"""
Redis client configuration and initialization
@nist-controls: SC-28, AU-12
@evidence: Secure caching with connection management
"""

import logging
import os

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


def get_redis_client() -> redis.Redis | None:
    """
    Get Redis client instance with connection pooling
    @nist-controls: SC-28
    @evidence: Secure cache connection management
    """
    try:
        # Get Redis configuration from environment
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        redis_password = os.getenv("REDIS_PASSWORD")

        # Create connection pool
        pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            max_connections=10,
            socket_keepalive=True,
            socket_keepalive_options={}
        )

        # Create Redis client
        client = redis.Redis(connection_pool=pool)

        # Test connection
        client.ping()

        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        return client

    except (RedisError, ConnectionError, Exception) as e:
        logger.warning(f"Redis connection failed: {e}. Running without cache.")
        return None


def close_redis_connection(client: redis.Redis | None) -> None:
    """
    Safely close Redis connection
    @nist-controls: SC-28
    @evidence: Proper resource cleanup
    """
    if client:
        try:
            client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

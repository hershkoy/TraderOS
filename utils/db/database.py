"""
Database utility module for TimescaleDB connections.
"""

import psycopg2
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_database_connection():
    """
    Get a database connection to TimescaleDB.
    
    Returns:
        psycopg2 connection object
    """
    try:
        # Get connection parameters from environment or use defaults
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', '5432'))
        database = os.getenv('DB_NAME', 'backtrader')
        user = os.getenv('DB_USER', 'backtrader_user')
        password = os.getenv('DB_PASSWORD', 'backtrader_password')
        
        # Create connection
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=10,
            options='-c statement_timeout=300000'  # 5 minutes timeout
        )
        
        logger.debug(f"Connected to database {database} at {host}:{port}")
        return conn
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

"""
PostgreSQL COPY utility for bulk loading data
Provides high-performance bulk insert operations using COPY FROM STDIN
"""

import io
import csv
import logging
from typing import List, Dict, Any, Iterator, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class PGCopyLoader:
    """High-performance bulk loader using PostgreSQL COPY"""
    
    def __init__(self, connection):
        """
        Initialize the COPY loader
        
        Args:
            connection: PostgreSQL connection object
        """
        self.connection = connection
    
    def copy_rows(self, table: str, columns: List[str], rows_iter: Iterator[Dict[str, Any]], 
                  batch_size: int = 1000) -> int:
        """
        Bulk load rows using COPY FROM STDIN
        
        Args:
            table: Target table name
            columns: List of column names in order
            rows_iter: Iterator yielding dictionaries with row data
            batch_size: Number of rows to process in each batch
        
        Returns:
            Total number of rows processed
        """
        total_rows = 0
        
        with self.connection.cursor() as cursor:
            # Process rows in batches
            batch = []
            
            for row in rows_iter:
                batch.append(row)
                
                if len(batch) >= batch_size:
                    processed = self._copy_batch(cursor, table, columns, batch)
                    total_rows += processed
                    batch = []
            
            # Process remaining rows
            if batch:
                processed = self._copy_batch(cursor, table, columns, batch)
                total_rows += processed
            
            self.connection.commit()
        
        logger.info(f"COPY completed: {total_rows} rows loaded into {table}")
        return total_rows
    
    def _copy_batch(self, cursor, table: str, columns: List[str], 
                    rows: List[Dict[str, Any]]) -> int:
        """
        Copy a batch of rows using COPY FROM STDIN
        
        Args:
            cursor: Database cursor
            table: Target table name
            columns: List of column names
            rows: List of row dictionaries
        
        Returns:
            Number of rows processed
        """
        if not rows:
            return 0
        
        # Prepare CSV data in memory
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        
        for row in rows:
            # Extract values in column order
            values = [row.get(col) for col in columns]
            writer.writerow(values)
        
        csv_buffer.seek(0)
        
        # Execute COPY command
        copy_sql = f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH CSV"
        
        try:
            cursor.copy_expert(copy_sql, csv_buffer)
            return len(rows)
        except Exception as e:
            logger.error(f"COPY failed for table {table}: {e}")
            raise
        finally:
            csv_buffer.close()
    
    def copy_rows_with_upsert(self, table: str, columns: List[str], rows_iter: Iterator[Dict[str, Any]],
                              conflict_columns: List[str], update_columns: List[str],
                              batch_size: int = 1000) -> int:
        """
        Bulk load rows with upsert logic using temporary table + INSERT ... ON CONFLICT
        
        Args:
            table: Target table name
            columns: List of column names in order
            rows_iter: Iterator yielding dictionaries with row data
            conflict_columns: Columns that define the conflict (for ON CONFLICT)
            update_columns: Columns to update on conflict
            batch_size: Number of rows to process in each batch
        
        Returns:
            Total number of rows processed
        """
        total_rows = 0
        
        with self.connection.cursor() as cursor:
            # Process rows in batches
            batch = []
            
            for row in rows_iter:
                batch.append(row)
                
                if len(batch) >= batch_size:
                    processed = self._upsert_batch(cursor, table, columns, batch, 
                                                 conflict_columns, update_columns)
                    total_rows += processed
                    batch = []
            
            # Process remaining rows
            if batch:
                processed = self._upsert_batch(cursor, table, columns, batch,
                                             conflict_columns, update_columns)
                total_rows += processed
            
            self.connection.commit()
        
        logger.info(f"Upsert completed: {total_rows} rows processed in {table}")
        return total_rows
    
    def _upsert_batch(self, cursor, table: str, columns: List[str], 
                      rows: List[Dict[str, Any]], conflict_columns: List[str],
                      update_columns: List[str]) -> int:
        """
        Upsert a batch of rows using temporary table approach
        
        Args:
            cursor: Database cursor
            table: Target table name
            columns: List of column names
            rows: List of row dictionaries
            conflict_columns: Columns that define the conflict
            update_columns: Columns to update on conflict
        
        Returns:
            Number of rows processed
        """
        if not rows:
            return 0
        
        # Create temporary table
        temp_table = f"temp_{table}_{id(rows)}"
        create_temp_sql = f"CREATE TEMP TABLE {temp_table} (LIKE {table})"
        cursor.execute(create_temp_sql)
        
        try:
            # Load data into temporary table using COPY
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            
            for row in rows:
                values = [row.get(col) for col in columns]
                writer.writerow(values)
            
            csv_buffer.seek(0)
            
            # COPY to temp table
            copy_sql = f"COPY {temp_table} ({', '.join(columns)}) FROM STDIN WITH CSV"
            cursor.copy_expert(copy_sql, csv_buffer)
            
            # Build upsert SQL
            conflict_clause = f"ON CONFLICT ({', '.join(conflict_columns)})"
            update_clause = f"DO UPDATE SET {', '.join(f'{col} = EXCLUDED.{col}' for col in update_columns)}"
            
            upsert_sql = f"""
                INSERT INTO {table} ({', '.join(columns)})
                SELECT {', '.join(columns)} FROM {temp_table}
                {conflict_clause} {update_clause}
            """
            
            cursor.execute(upsert_sql)
            return len(rows)
            
        finally:
            csv_buffer.close()
            cursor.execute(f"DROP TABLE {temp_table}")


def copy_rows(conn, table: str, columns: List[str], rows_iter: Iterator[Dict[str, Any]], 
              batch_size: int = 1000) -> int:
    """
    Convenience function to copy rows using COPY
    
    Args:
        conn: PostgreSQL connection
        table: Target table name
        columns: List of column names
        rows_iter: Iterator yielding row dictionaries
        batch_size: Batch size for processing
    
    Returns:
        Number of rows processed
    """
    loader = PGCopyLoader(conn)
    return loader.copy_rows(table, columns, rows_iter, batch_size)


def copy_rows_with_upsert(conn, table: str, columns: List[str], rows_iter: Iterator[Dict[str, Any]],
                          conflict_columns: List[str], update_columns: List[str],
                          batch_size: int = 1000) -> int:
    """
    Convenience function to copy rows with upsert logic
    
    Args:
        conn: PostgreSQL connection
        table: Target table name
        columns: List of column names
        rows_iter: Iterator yielding row dictionaries
        conflict_columns: Columns that define conflicts
        update_columns: Columns to update on conflict
        batch_size: Batch size for processing
    
    Returns:
        Number of rows processed
    """
    loader = PGCopyLoader(conn)
    return loader.copy_rows_with_upsert(table, columns, rows_iter, 
                                       conflict_columns, update_columns, batch_size)

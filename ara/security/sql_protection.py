"""
SQL Injection Protection Module

Provides protection against SQL injection attacks through parameterized
queries and input validation.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


class SQLProtection:
    """Provides SQL injection protection"""
    
    # Dangerous SQL keywords and patterns
    DANGEROUS_PATTERNS = [
        r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE)\b",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b.*=.*|1=1|'=')",
        r"(xp_|sp_cmdshell|exec|execute)",
        r"(UNION.*SELECT)",
        r"(INTO\s+(OUT|DUMP)FILE)",
    ]
    
    @classmethod
    def validate_query_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate query parameters for SQL injection attempts
        
        Args:
            params: Dictionary of query parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If dangerous patterns detected
        """
        validated = {}
        
        for key, value in params.items():
            # Validate key
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid parameter name: {key}")
            
            # Validate value
            if isinstance(value, str):
                for pattern in cls.DANGEROUS_PATTERNS:
                    if re.search(pattern, value, re.IGNORECASE):
                        raise ValueError(
                            f"Parameter '{key}' contains potentially dangerous SQL content"
                        )
            
            validated[key] = value
        
        return validated
    
    @classmethod
    def safe_execute(
        cls,
        session: Session,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Safely execute a SQL query with parameterized values
        
        Args:
            session: SQLAlchemy session
            query: SQL query with named parameters (e.g., :param_name)
            params: Dictionary of parameter values
            
        Returns:
            Query result
            
        Raises:
            ValueError: If query or parameters are invalid
        """
        if params is None:
            params = {}
        
        # Validate parameters
        validated_params = cls.validate_query_params(params)
        
        # Ensure query uses parameterized format
        if params and not re.search(r':\w+', query):
            raise ValueError(
                "Query must use parameterized format (e.g., :param_name) "
                "when parameters are provided"
            )
        
        # Execute with SQLAlchemy's text() for proper escaping
        result = session.execute(text(query), validated_params)
        return result
    
    @classmethod
    def escape_like_pattern(cls, pattern: str) -> str:
        """
        Escape special characters in LIKE patterns
        
        Args:
            pattern: LIKE pattern string
            
        Returns:
            Escaped pattern
        """
        # Escape special LIKE characters
        pattern = pattern.replace('\\', '\\\\')
        pattern = pattern.replace('%', '\\%')
        pattern = pattern.replace('_', '\\_')
        return pattern
    
    @classmethod
    def validate_table_name(cls, table_name: str) -> str:
        """
        Validate table name to prevent SQL injection
        
        Args:
            table_name: Table name
            
        Returns:
            Validated table name
            
        Raises:
            ValueError: If table name is invalid
        """
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. "
                "Must start with letter or underscore and contain only "
                "alphanumeric characters and underscores"
            )
        
        if len(table_name) > 64:
            raise ValueError("Table name too long (max 64 characters)")
        
        return table_name
    
    @classmethod
    def validate_column_name(cls, column_name: str) -> str:
        """
        Validate column name to prevent SQL injection
        
        Args:
            column_name: Column name
            
        Returns:
            Validated column name
            
        Raises:
            ValueError: If column name is invalid
        """
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name):
            raise ValueError(
                f"Invalid column name: {column_name}. "
                "Must start with letter or underscore and contain only "
                "alphanumeric characters and underscores"
            )
        
        if len(column_name) > 64:
            raise ValueError("Column name too long (max 64 characters)")
        
        return column_name
    
    @classmethod
    def build_safe_where_clause(
        cls,
        conditions: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe WHERE clause with parameterized values
        
        Args:
            conditions: Dictionary of column: value conditions
            
        Returns:
            Tuple of (WHERE clause string, parameters dict)
            
        Example:
            >>> clause, params = build_safe_where_clause({
            ...     'symbol': 'AAPL',
            ...     'price': 150.0
            ... })
            >>> print(clause)
            'symbol = :symbol AND price = :price'
            >>> print(params)
            {'symbol': 'AAPL', 'price': 150.0}
        """
        if not conditions:
            return "", {}
        
        clauses = []
        params = {}
        
        for column, value in conditions.items():
            # Validate column name
            validated_column = cls.validate_column_name(column)
            
            # Build parameterized clause
            param_name = f"param_{validated_column}"
            clauses.append(f"{validated_column} = :{param_name}")
            params[param_name] = value
        
        where_clause = " AND ".join(clauses)
        return where_clause, params
    
    @classmethod
    def build_safe_insert(
        cls,
        table_name: str,
        data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe INSERT statement with parameterized values
        
        Args:
            table_name: Name of table
            data: Dictionary of column: value pairs
            
        Returns:
            Tuple of (INSERT statement, parameters dict)
        """
        # Validate table name
        validated_table = cls.validate_table_name(table_name)
        
        # Validate column names
        columns = []
        params = {}
        
        for column, value in data.items():
            validated_column = cls.validate_column_name(column)
            columns.append(validated_column)
            params[validated_column] = value
        
        # Build INSERT statement
        columns_str = ", ".join(columns)
        params_str = ", ".join([f":{col}" for col in columns])
        
        query = f"INSERT INTO {validated_table} ({columns_str}) VALUES ({params_str})"
        
        return query, params
    
    @classmethod
    def build_safe_update(
        cls,
        table_name: str,
        data: Dict[str, Any],
        conditions: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a safe UPDATE statement with parameterized values
        
        Args:
            table_name: Name of table
            data: Dictionary of column: value pairs to update
            conditions: Dictionary of WHERE conditions
            
        Returns:
            Tuple of (UPDATE statement, parameters dict)
        """
        # Validate table name
        validated_table = cls.validate_table_name(table_name)
        
        # Build SET clause
        set_clauses = []
        params = {}
        
        for column, value in data.items():
            validated_column = cls.validate_column_name(column)
            param_name = f"set_{validated_column}"
            set_clauses.append(f"{validated_column} = :{param_name}")
            params[param_name] = value
        
        set_clause = ", ".join(set_clauses)
        
        # Build WHERE clause
        where_clause, where_params = cls.build_safe_where_clause(conditions)
        params.update(where_params)
        
        # Build UPDATE statement
        query = f"UPDATE {validated_table} SET {set_clause}"
        if where_clause:
            query += f" WHERE {where_clause}"
        
        return query, params

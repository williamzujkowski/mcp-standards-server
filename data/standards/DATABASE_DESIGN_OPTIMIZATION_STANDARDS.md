# Database Design and Optimization Standards

## 1. Overview

This document provides comprehensive standards for database design, optimization, and management across relational and NoSQL systems.

### Purpose

- Establish best practices for database schema design and data modeling
- Define performance optimization strategies and techniques
- Ensure data security, integrity, and availability
- Guide technology selection and migration patterns

### Scope

These standards apply to:
- Relational database systems (PostgreSQL, MySQL, Oracle, SQL Server)
- NoSQL databases (MongoDB, Cassandra, Redis, DynamoDB)
- Time-series databases (InfluxDB, TimescaleDB)
- Graph databases (Neo4j, Amazon Neptune)
- Data warehouses and OLAP systems

## 2. Database Design Principles

### 2.1 Schema Design Best Practices

**Standard**: Follow normalized design patterns while balancing performance requirements

```sql
-- Example: Properly normalized e-commerce schema
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NOT NULL REFERENCES customers(customer_id),
    order_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    CONSTRAINT chk_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled'))
);

CREATE TABLE order_items (
    order_item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(order_id),
    product_id UUID NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0
);

-- Create indexes for common queries
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_status_date ON orders(status, order_date);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
```

**NIST Mappings**: SC-8 (data integrity), AC-4 (information flow)

### 2.2 Data Modeling Patterns

**Standard**: Choose appropriate data models based on use case requirements

```python
# Example: Document store model for product catalog
product_document = {
    "_id": "prod_12345",
    "name": "Premium Laptop",
    "category": "Electronics",
    "attributes": {
        "brand": "TechCorp",
        "processor": "Intel i7",
        "memory": "16GB",
        "storage": "512GB SSD"
    },
    "variants": [
        {
            "sku": "LAP-i7-16-512",
            "color": "Silver",
            "price": 1299.99,
            "inventory": 45
        },
        {
            "sku": "LAP-i7-16-512-BLK",
            "color": "Black",
            "price": 1299.99,
            "inventory": 32
        }
    ],
    "reviews": {
        "average_rating": 4.5,
        "total_reviews": 234,
        "rating_distribution": {
            "5": 150,
            "4": 60,
            "3": 20,
            "2": 3,
            "1": 1
        }
    }
}

# Graph model for social network
CREATE (user1:User {id: 'user_001', name: 'Alice'})
CREATE (user2:User {id: 'user_002', name: 'Bob'})
CREATE (post1:Post {id: 'post_001', content: 'Hello Graph DB!'})
CREATE (user1)-[:FOLLOWS]->(user2)
CREATE (user1)-[:CREATED]->(post1)
CREATE (user2)-[:LIKES]->(post1)
```

**NIST Mappings**: SA-8 (security engineering), SC-28 (data protection)

## 3. Query Optimization Techniques

### 3.1 Index Strategy

**Standard**: Design indexes based on query patterns and workload analysis

```sql
-- Example: Comprehensive indexing strategy
-- Analyze query patterns first
EXPLAIN (ANALYZE, BUFFERS) 
SELECT c.email, COUNT(o.order_id) as order_count, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2024-01-01'
GROUP BY c.customer_id, c.email
HAVING SUM(o.total_amount) > 1000;

-- Create appropriate indexes
CREATE INDEX idx_orders_date_customer ON orders(order_date, customer_id) 
INCLUDE (total_amount);

-- Partial index for active customers
CREATE INDEX idx_customers_active ON customers(customer_id, email)
WHERE deleted_at IS NULL;

-- Expression index for case-insensitive searches
CREATE INDEX idx_customers_email_lower ON customers(LOWER(email));

-- Multi-column index for compound queries
CREATE INDEX idx_products_category_price ON products(category, price DESC)
WHERE active = true;
```

**NIST Mappings**: SC-8 (transmission confidentiality), SI-10 (information accuracy)

### 3.2 Query Performance Tuning

**Standard**: Optimize queries through proper structure and execution planning

```sql
-- Example: Query optimization techniques
-- Bad: SELECT * with unnecessary joins
SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id;

-- Good: Select only needed columns and use CTEs
WITH recent_orders AS (
    SELECT order_id, customer_id, total_amount
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
    AND status = 'delivered'
),
customer_totals AS (
    SELECT 
        customer_id,
        COUNT(*) as order_count,
        SUM(total_amount) as total_spent
    FROM recent_orders
    GROUP BY customer_id
)
SELECT 
    c.email,
    ct.order_count,
    ct.total_spent
FROM customer_totals ct
JOIN customers c ON ct.customer_id = c.customer_id
WHERE ct.total_spent > 500
ORDER BY ct.total_spent DESC
LIMIT 100;

-- Use window functions for complex analytics
SELECT 
    customer_id,
    order_date,
    total_amount,
    SUM(total_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id 
        ORDER BY total_amount DESC
    ) as order_rank
FROM orders
WHERE order_date >= '2024-01-01';
```

**NIST Mappings**: SI-10 (information input validation), AU-4 (audit storage capacity)

## 4. Database Selection Criteria

### 4.1 SQL vs NoSQL Decision Matrix

**Standard**: Choose database technology based on specific requirements

```python
# Example: Database selection framework
class DatabaseSelector:
    def __init__(self):
        self.criteria_weights = {
            'consistency': 0.25,
            'scalability': 0.25,
            'flexibility': 0.20,
            'performance': 0.20,
            'ecosystem': 0.10
        }
    
    def evaluate_use_case(self, requirements):
        """Evaluate database options based on requirements"""
        scores = {}
        
        # RDBMS evaluation
        if requirements.get('acid_compliance') == 'required':
            scores['postgresql'] = self._calculate_score({
                'consistency': 10,
                'scalability': 7,
                'flexibility': 6,
                'performance': 8,
                'ecosystem': 9
            })
        
        # Document store evaluation
        if requirements.get('schema_flexibility') == 'high':
            scores['mongodb'] = self._calculate_score({
                'consistency': 7,
                'scalability': 9,
                'flexibility': 10,
                'performance': 8,
                'ecosystem': 8
            })
        
        # Key-value store evaluation
        if requirements.get('latency') == 'sub_millisecond':
            scores['redis'] = self._calculate_score({
                'consistency': 6,
                'scalability': 9,
                'flexibility': 7,
                'performance': 10,
                'ecosystem': 8
            })
        
        # Graph database evaluation
        if requirements.get('relationship_complexity') == 'high':
            scores['neo4j'] = self._calculate_score({
                'consistency': 8,
                'scalability': 7,
                'flexibility': 9,
                'performance': 8,
                'ecosystem': 7
            })
        
        return scores
    
    def _calculate_score(self, ratings):
        """Calculate weighted score"""
        total = 0
        for criterion, weight in self.criteria_weights.items():
            total += ratings[criterion] * weight
        return total

# Usage example
selector = DatabaseSelector()
requirements = {
    'acid_compliance': 'required',
    'schema_flexibility': 'medium',
    'latency': 'low',
    'data_volume': 'high'
}
recommendations = selector.evaluate_use_case(requirements)
```

**NIST Mappings**: SA-4 (acquisition process), SA-11 (developer testing)

## 5. Migration Patterns

### 5.1 Schema Migration Strategy

**Standard**: Implement versioned, reversible database migrations

```python
# Example: Database migration framework
from datetime import datetime
import logging

class DatabaseMigration:
    def __init__(self, version, description):
        self.version = version
        self.description = description
        self.executed_at = None
    
    def up(self, connection):
        """Apply migration"""
        raise NotImplementedError
    
    def down(self, connection):
        """Rollback migration"""
        raise NotImplementedError

class AddCustomerTierMigration(DatabaseMigration):
    def __init__(self):
        super().__init__(
            version="20240115_001",
            description="Add customer tier system"
        )
    
    def up(self, connection):
        """Add customer tier functionality"""
        with connection.cursor() as cursor:
            # Add tier column
            cursor.execute("""
                ALTER TABLE customers 
                ADD COLUMN tier VARCHAR(20) DEFAULT 'bronze'
                CHECK (tier IN ('bronze', 'silver', 'gold', 'platinum'))
            """)
            
            # Create tier benefits table
            cursor.execute("""
                CREATE TABLE customer_tier_benefits (
                    tier VARCHAR(20) PRIMARY KEY,
                    discount_percentage DECIMAL(5, 2),
                    free_shipping_threshold DECIMAL(10, 2),
                    priority_support BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default tier benefits
            cursor.execute("""
                INSERT INTO customer_tier_benefits 
                (tier, discount_percentage, free_shipping_threshold, priority_support)
                VALUES 
                ('bronze', 0, 100, false),
                ('silver', 5, 75, false),
                ('gold', 10, 50, true),
                ('platinum', 15, 0, true)
            """)
            
            # Update existing customers based on spending
            cursor.execute("""
                UPDATE customers c
                SET tier = CASE
                    WHEN total_spent >= 10000 THEN 'platinum'
                    WHEN total_spent >= 5000 THEN 'gold'
                    WHEN total_spent >= 1000 THEN 'silver'
                    ELSE 'bronze'
                END
                FROM (
                    SELECT customer_id, SUM(total_amount) as total_spent
                    FROM orders
                    WHERE status = 'delivered'
                    GROUP BY customer_id
                ) o
                WHERE c.customer_id = o.customer_id
            """)
    
    def down(self, connection):
        """Remove customer tier functionality"""
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS customer_tier_benefits")
            cursor.execute("ALTER TABLE customers DROP COLUMN IF EXISTS tier")

# Migration runner
class MigrationRunner:
    def __init__(self, connection):
        self.connection = connection
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create migration tracking table"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(50) PRIMARY KEY,
                    description TEXT,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms INTEGER
                )
            """)
    
    def run_migration(self, migration):
        """Execute a migration with proper tracking"""
        start_time = datetime.now()
        
        try:
            # Check if already executed
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM schema_migrations WHERE version = %s",
                    (migration.version,)
                )
                if cursor.fetchone():
                    logging.info(f"Migration {migration.version} already executed")
                    return
            
            # Run migration
            migration.up(self.connection)
            
            # Record execution
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO schema_migrations (version, description, execution_time_ms)
                    VALUES (%s, %s, %s)
                """, (migration.version, migration.description, execution_time))
            
            self.connection.commit()
            logging.info(f"Migration {migration.version} completed in {execution_time}ms")
            
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Migration {migration.version} failed: {str(e)}")
            raise
```

**NIST Mappings**: CM-3 (configuration change control), CM-4 (security impact analysis)

## 6. Performance Monitoring

### 6.1 Query Performance Monitoring

**Standard**: Implement comprehensive query performance tracking

```python
# Example: Query performance monitoring system
import time
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class QueryMetrics:
    query_hash: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    index_used: Optional[str]
    lock_wait_time_ms: float
    temp_tables_created: int

class PerformanceMonitor:
    def __init__(self, db_connection, threshold_ms=1000):
        self.connection = db_connection
        self.slow_query_threshold = threshold_ms
        self.metrics_buffer = []
    
    @contextmanager
    def monitor_query(self, query_description: str):
        """Monitor query execution"""
        start_time = time.time()
        initial_stats = self._get_connection_stats()
        
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000
            final_stats = self._get_connection_stats()
            
            metrics = QueryMetrics(
                query_hash=self._hash_query(query_description),
                execution_time_ms=execution_time,
                rows_examined=final_stats['rows_examined'] - initial_stats['rows_examined'],
                rows_returned=final_stats['rows_sent'] - initial_stats['rows_sent'],
                index_used=self._get_last_index_used(),
                lock_wait_time_ms=final_stats['lock_time'] - initial_stats['lock_time'],
                temp_tables_created=final_stats['created_tmp_tables'] - initial_stats['created_tmp_tables']
            )
            
            self._record_metrics(metrics, query_description)
    
    def _get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics"""
        with self.connection.cursor() as cursor:
            cursor.execute("SHOW STATUS LIKE 'Handler%'")
            handler_stats = dict(cursor.fetchall())
            
            cursor.execute("SHOW STATUS LIKE 'Created_tmp%'")
            tmp_stats = dict(cursor.fetchall())
            
            return {
                'rows_examined': int(handler_stats.get('Handler_read_next', 0)),
                'rows_sent': int(handler_stats.get('Handler_write', 0)),
                'created_tmp_tables': int(tmp_stats.get('Created_tmp_tables', 0)),
                'lock_time': 0  # Would need performance_schema for accurate lock time
            }
    
    def _record_metrics(self, metrics: QueryMetrics, description: str):
        """Record and analyze metrics"""
        if metrics.execution_time_ms > self.slow_query_threshold:
            self._log_slow_query(metrics, description)
        
        # Buffer metrics for batch insert
        self.metrics_buffer.append({
            'timestamp': time.time(),
            'metrics': metrics,
            'description': description
        })
        
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()
    
    def _log_slow_query(self, metrics: QueryMetrics, description: str):
        """Log slow query for analysis"""
        logging.warning(f"Slow query detected: {description}")
        logging.warning(f"Execution time: {metrics.execution_time_ms}ms")
        logging.warning(f"Rows examined: {metrics.rows_examined}")
        
        # Suggest optimizations
        if metrics.rows_examined > metrics.rows_returned * 10:
            logging.warning("Consider adding an index - examining too many rows")
        if metrics.temp_tables_created > 0:
            logging.warning("Query created temporary tables - consider optimization")
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns for optimization opportunities"""
        with self.connection.cursor() as cursor:
            # Get most frequent queries
            cursor.execute("""
                SELECT 
                    query_hash,
                    COUNT(*) as execution_count,
                    AVG(execution_time_ms) as avg_time,
                    MAX(execution_time_ms) as max_time,
                    SUM(rows_examined) as total_rows_examined
                FROM query_metrics
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                GROUP BY query_hash
                ORDER BY execution_count * avg_time DESC
                LIMIT 20
            """)
            
            problem_queries = []
            for row in cursor.fetchall():
                if row['avg_time'] > 500 or row['total_rows_examined'] > 1000000:
                    problem_queries.append({
                        'query_hash': row['query_hash'],
                        'impact_score': row['execution_count'] * row['avg_time'],
                        'optimization_potential': self._calculate_optimization_potential(row)
                    })
            
            return {
                'problem_queries': problem_queries,
                'recommendations': self._generate_recommendations(problem_queries)
            }
```

**NIST Mappings**: AU-6 (audit review), SI-4 (information system monitoring)

## 7. Security and Access Control

### 7.1 Database Security Implementation

**Standard**: Implement defense-in-depth security for database systems

```sql
-- Example: Comprehensive security implementation
-- Create roles with least privilege
CREATE ROLE app_read_only;
GRANT CONNECT ON DATABASE production TO app_read_only;
GRANT USAGE ON SCHEMA public TO app_read_only;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_read_only;

CREATE ROLE app_read_write;
GRANT CONNECT ON DATABASE production TO app_read_write;
GRANT USAGE ON SCHEMA public TO app_read_write;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_read_write;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_read_write;

-- Row-level security
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;

CREATE POLICY customer_isolation ON customers
    FOR ALL
    TO app_user
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Encrypt sensitive columns
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypted PII storage
CREATE TABLE customer_pii (
    customer_id UUID PRIMARY KEY REFERENCES customers(customer_id),
    ssn_encrypted BYTEA,
    credit_card_encrypted BYTEA,
    encryption_key_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function for secure data access
CREATE OR REPLACE FUNCTION get_customer_ssn(
    p_customer_id UUID,
    p_reason TEXT
) RETURNS TEXT AS $$
DECLARE
    v_ssn TEXT;
    v_user_id UUID;
BEGIN
    -- Audit the access
    v_user_id := current_setting('app.current_user')::uuid;
    
    INSERT INTO pii_access_log (
        user_id,
        customer_id,
        field_accessed,
        reason,
        accessed_at
    ) VALUES (
        v_user_id,
        p_customer_id,
        'ssn',
        p_reason,
        CURRENT_TIMESTAMP
    );
    
    -- Decrypt and return if authorized
    SELECT pgp_sym_decrypt(
        ssn_encrypted,
        current_setting('app.encryption_key')
    ) INTO v_ssn
    FROM customer_pii
    WHERE customer_id = p_customer_id;
    
    RETURN v_ssn;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Audit logging
CREATE TABLE audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100),
    operation VARCHAR(10),
    user_id UUID,
    changed_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Generic audit trigger
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, operation, user_id, changed_data)
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        current_setting('app.current_user')::uuid,
        to_jsonb(NEW) - to_jsonb(OLD)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

**NIST Mappings**: AC-3 (access enforcement), AC-6 (least privilege), AU-3 (audit record content)

### 7.2 Data Masking and Anonymization

**Standard**: Implement data masking for non-production environments

```python
# Example: Data masking framework
import hashlib
import random
from faker import Faker
from typing import Dict, Any, Callable

class DataMasker:
    def __init__(self, seed=None):
        self.faker = Faker()
        if seed:
            Faker.seed(seed)
            random.seed(seed)
        
        self.masking_rules = {
            'email': self._mask_email,
            'phone': self._mask_phone,
            'ssn': self._mask_ssn,
            'credit_card': self._mask_credit_card,
            'name': self._mask_name,
            'address': self._mask_address,
            'date_of_birth': self._mask_dob,
            'salary': self._mask_salary
        }
    
    def _mask_email(self, email: str) -> str:
        """Mask email while preserving format"""
        local, domain = email.split('@')
        if len(local) > 3:
            masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
        else:
            masked_local = '*' * len(local)
        return f"{masked_local}@{domain}"
    
    def _mask_phone(self, phone: str) -> str:
        """Mask phone number"""
        cleaned = ''.join(c for c in phone if c.isdigit())
        if len(cleaned) >= 10:
            return f"({cleaned[:3]}) ***-**{cleaned[-2:]}"
        return "*" * len(phone)
    
    def _mask_ssn(self, ssn: str) -> str:
        """Mask SSN completely"""
        return "***-**-****"
    
    def _mask_credit_card(self, cc: str) -> str:
        """Mask credit card preserving last 4"""
        cleaned = ''.join(c for c in cc if c.isdigit())
        if len(cleaned) >= 4:
            return "*" * (len(cleaned) - 4) + cleaned[-4:]
        return "*" * len(cc)
    
    def _mask_name(self, name: str) -> str:
        """Replace with fake name"""
        return self.faker.name()
    
    def _mask_address(self, address: str) -> str:
        """Replace with fake address"""
        return self.faker.address()
    
    def _mask_dob(self, dob: Any) -> Any:
        """Randomize birth year keeping month/day"""
        if isinstance(dob, str):
            # Parse and modify date
            parts = dob.split('-')
            if len(parts) == 3:
                year = random.randint(1950, 2000)
                return f"{year}-{parts[1]}-{parts[2]}"
        return dob
    
    def _mask_salary(self, salary: float) -> float:
        """Add random variance to salary"""
        variance = random.uniform(0.8, 1.2)
        return round(salary * variance, -3)  # Round to nearest thousand
    
    def mask_dataset(self, data: Dict[str, Any], field_types: Dict[str, str]) -> Dict[str, Any]:
        """Mask sensitive fields in dataset"""
        masked_data = data.copy()
        
        for field, field_type in field_types.items():
            if field in data and field_type in self.masking_rules:
                masked_data[field] = self.masking_rules[field_type](data[field])
        
        return masked_data
    
    def create_test_database(self, source_conn, target_conn, masking_config):
        """Create masked copy of production database"""
        for table_name, config in masking_config.items():
            # Copy schema
            self._copy_table_schema(source_conn, target_conn, table_name)
            
            # Copy and mask data
            with source_conn.cursor() as source_cursor:
                source_cursor.execute(f"SELECT * FROM {table_name}")
                columns = [desc[0] for desc in source_cursor.description]
                
                with target_conn.cursor() as target_cursor:
                    for row in source_cursor:
                        row_dict = dict(zip(columns, row))
                        masked_row = self.mask_dataset(row_dict, config['field_types'])
                        
                        # Insert masked data
                        placeholders = ', '.join(['%s'] * len(columns))
                        insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        target_cursor.execute(insert_query, list(masked_row.values()))
            
            target_conn.commit()

# Usage example
masker = DataMasker(seed=42)  # Consistent masking for testing
masking_config = {
    'customers': {
        'field_types': {
            'email': 'email',
            'phone_number': 'phone',
            'full_name': 'name',
            'billing_address': 'address',
            'date_of_birth': 'date_of_birth'
        }
    },
    'employees': {
        'field_types': {
            'ssn': 'ssn',
            'salary': 'salary',
            'personal_email': 'email'
        }
    }
}
```

**NIST Mappings**: SC-28 (protection of information at rest), MP-6 (media sanitization)

## 8. Backup and Disaster Recovery

### 8.1 Backup Strategy Implementation

**Standard**: Implement comprehensive backup and recovery procedures

```python
# Example: Automated backup system with verification
import subprocess
import os
import hashlib
import boto3
from datetime import datetime, timedelta
import logging

class DatabaseBackupManager:
    def __init__(self, db_config, storage_config):
        self.db_config = db_config
        self.storage_config = storage_config
        self.s3_client = boto3.client('s3')
        self.retention_policies = {
            'daily': 7,
            'weekly': 4,
            'monthly': 12,
            'yearly': 7
        }
    
    def perform_backup(self, backup_type='daily'):
        """Execute database backup with verification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{self.db_config['database']}_{backup_type}_{timestamp}"
        
        try:
            # Create backup
            backup_file = self._create_backup(backup_name)
            
            # Verify backup integrity
            if not self._verify_backup(backup_file):
                raise Exception("Backup verification failed")
            
            # Encrypt backup
            encrypted_file = self._encrypt_backup(backup_file)
            
            # Upload to storage
            s3_key = self._upload_backup(encrypted_file, backup_type)
            
            # Test restore capability
            if backup_type in ['weekly', 'monthly']:
                self._test_restore(s3_key)
            
            # Clean up local files
            self._cleanup_local_files([backup_file, encrypted_file])
            
            # Apply retention policy
            self._apply_retention_policy(backup_type)
            
            # Log success
            self._log_backup_success(backup_name, s3_key)
            
            return s3_key
            
        except Exception as e:
            self._handle_backup_failure(e, backup_name)
            raise
    
    def _create_backup(self, backup_name):
        """Create database backup using appropriate method"""
        backup_file = f"/tmp/{backup_name}.sql"
        
        if self.db_config['type'] == 'postgresql':
            cmd = [
                'pg_dump',
                '-h', self.db_config['host'],
                '-U', self.db_config['user'],
                '-d', self.db_config['database'],
                '--no-owner',
                '--no-privileges',
                '--if-exists',
                '--clean',
                '-f', backup_file
            ]
            
            # Set password via environment
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_config['password']
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Backup failed: {result.stderr}")
        
        elif self.db_config['type'] == 'mysql':
            cmd = [
                'mysqldump',
                f"--host={self.db_config['host']}",
                f"--user={self.db_config['user']}",
                f"--password={self.db_config['password']}",
                '--single-transaction',
                '--routines',
                '--triggers',
                '--events',
                self.db_config['database']
            ]
            
            with open(backup_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise Exception(f"Backup failed: {result.stderr}")
        
        return backup_file
    
    def _verify_backup(self, backup_file):
        """Verify backup integrity"""
        # Check file size
        file_size = os.path.getsize(backup_file)
        if file_size < 1000:  # Minimum expected size
            logging.error("Backup file too small")
            return False
        
        # Check for required content
        with open(backup_file, 'r') as f:
            content = f.read(10000)  # Read first 10KB
            
            # Verify expected content
            if self.db_config['type'] == 'postgresql':
                required_patterns = ['CREATE TABLE', 'COPY', 'PostgreSQL database dump']
            else:
                required_patterns = ['CREATE TABLE', 'INSERT INTO', 'MySQL dump']
            
            for pattern in required_patterns:
                if pattern not in content:
                    logging.error(f"Missing expected pattern: {pattern}")
                    return False
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_file)
        logging.info(f"Backup checksum: {checksum}")
        
        return True
    
    def _encrypt_backup(self, backup_file):
        """Encrypt backup file"""
        encrypted_file = f"{backup_file}.enc"
        
        cmd = [
            'openssl', 'enc',
            '-aes-256-cbc',
            '-salt',
            '-in', backup_file,
            '-out', encrypted_file,
            '-pass', f"pass:{self.storage_config['encryption_key']}"
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise Exception("Encryption failed")
        
        return encrypted_file
    
    def _test_restore(self, s3_key):
        """Test restore capability on a test database"""
        test_db = f"{self.db_config['database']}_restore_test"
        
        try:
            # Download backup
            local_file = self._download_backup(s3_key)
            
            # Decrypt
            decrypted_file = self._decrypt_backup(local_file)
            
            # Create test database
            self._create_test_database(test_db)
            
            # Restore backup
            self._restore_backup(decrypted_file, test_db)
            
            # Verify restore
            if not self._verify_restore(test_db):
                raise Exception("Restore verification failed")
            
            # Cleanup test database
            self._drop_test_database(test_db)
            
            logging.info(f"Restore test successful for {s3_key}")
            
        except Exception as e:
            logging.error(f"Restore test failed: {str(e)}")
            raise
    
    def _apply_retention_policy(self, backup_type):
        """Apply retention policy to remove old backups"""
        retention_days = self.retention_policies.get(backup_type, 7)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # List objects in S3
        prefix = f"{self.db_config['database']}_{backup_type}_"
        response = self.s3_client.list_objects_v2(
            Bucket=self.storage_config['bucket'],
            Prefix=prefix
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.storage_config['bucket'],
                        Key=obj['Key']
                    )
                    logging.info(f"Deleted old backup: {obj['Key']}")
    
    def create_recovery_plan(self):
        """Generate disaster recovery documentation"""
        recovery_plan = {
            'database_info': {
                'type': self.db_config['type'],
                'version': self._get_database_version(),
                'size': self._get_database_size()
            },
            'backup_schedule': {
                'daily': '02:00 UTC',
                'weekly': 'Sunday 03:00 UTC',
                'monthly': 'First Sunday 04:00 UTC',
                'yearly': 'January 1st 05:00 UTC'
            },
            'recovery_procedures': self._generate_recovery_procedures(),
            'rto_rpo': {
                'rto': '4 hours',
                'rpo': '24 hours',
                'tested_on': datetime.now().isoformat()
            },
            'contact_info': {
                'primary_dba': 'dba-team@company.com',
                'escalation': 'infrastructure@company.com'
            }
        }
        
        return recovery_plan
```

**NIST Mappings**: CP-9 (information system backup), CP-10 (information system recovery)

## 9. Best Practices Summary

### 9.1 Design Guidelines

1. **Normalization**: Aim for 3NF but denormalize strategically for performance
2. **Indexing**: Create indexes based on query patterns, not assumptions
3. **Partitioning**: Use table partitioning for large datasets
4. **Constraints**: Always define appropriate constraints and foreign keys

### 9.2 Performance Guidelines

1. **Query Optimization**: Use EXPLAIN plans and query analyzers
2. **Connection Pooling**: Implement appropriate connection pooling
3. **Caching**: Use application-level and database-level caching
4. **Monitoring**: Track slow queries and resource usage

### 9.3 Security Guidelines

1. **Access Control**: Implement least privilege access
2. **Encryption**: Encrypt data at rest and in transit
3. **Auditing**: Log all sensitive data access
4. **Masking**: Use data masking in non-production environments

### 9.4 Operational Guidelines

1. **Backups**: Test restore procedures regularly
2. **Monitoring**: Set up comprehensive monitoring and alerting
3. **Documentation**: Maintain up-to-date schema documentation
4. **Change Management**: Use version-controlled migrations

## 10. Compliance Requirements

### 10.1 Regulatory Compliance

- **GDPR**: Right to erasure, data portability
- **HIPAA**: Encryption, access controls, audit logs
- **PCI DSS**: Data retention, encryption, access monitoring
- **SOX**: Change tracking, separation of duties

### 10.2 Industry Standards

- **ISO 27001**: Information security management
- **NIST Cybersecurity Framework**: Identify, Protect, Detect, Respond, Recover
- **CIS Controls**: Database security benchmarks

## 11. Conclusion

These database design and optimization standards provide a comprehensive framework for building secure, performant, and maintainable database systems. Regular review and updates of these standards ensure alignment with evolving technologies and requirements.
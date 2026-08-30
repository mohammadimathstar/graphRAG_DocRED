import psycopg
from psycopg_pool import ConnectionPool
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize the pool
conninfo = f"""
    host={os.getenv("DB_HOST", "localhost")} 
    port={os.getenv("DB_PORT", "5432")} 
    dbname={os.getenv("DB_NAME", "graphrag_db")} 
    user={os.getenv("DB_USER", "admin")} 
    password={os.getenv("DB_PASS", "pass")}
"""

# Initialize the pool (e.g., min 2, max 10 concurrent connections)
pool = ConnectionPool(conninfo=conninfo, min_size=2, max_size=10, open=True)


def get_conn_from_pool():
    """Checks out a connection from the pool."""
    return pool.getconn()


def release_conn(conn):
    """Returns the connection to the pool."""
    pool.putconn(conn)


def get_connection():
    """Initializes and returns a psycopg connection."""
    conn = psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "graphrag_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "password"),
        port=os.getenv("DB_PORT", "5432"),
        autocommit=False,
    )
    return conn

from .connection import get_conn_from_pool, release_conn
from . import table_schemas as schema_module


def init_db(drop_if_exists: bool = False):
    """
    Initializes the database tables.
    If drop=True, drops existing tables before recreating them.
    """
    print("Checking out connection from pool for DB initialization...")
    conn = get_conn_from_pool()

    try:
        with conn.cursor() as cur:
            print("Enabling extensions...")
            cur.execute(schema_module.ENABLE_EXTENSIONS_SQL)

            if drop_if_exists:
                print("Dropping existing tables...")
                cur.execute(schema_module.DROP_TABLES_SQL)

            print("Creating tables...\n")
            cur.execute(schema_module.DOCUMENTS_SQL)
            cur.execute(schema_module.RUNS_SQL)
            cur.execute(schema_module.EXTRACTIONS_SQL)
            cur.execute(schema_module.EVALUATIONS_SQL)
            cur.execute(schema_module.ENTITIES_SQL)
            cur.execute(schema_module.TRIPLES_SQL)
            cur.execute(schema_module.QA_PAIRS_SQL)
            cur.execute(schema_module.RAG_EVALUATIONS_SQL)
            cur.execute(schema_module.PRODUCTION_TRACES_SQL)

        conn.commit()
        print("Database initialized successfully!")

    except Exception as e:
        # If ANY table fails to create, roll back the transaction
        conn.rollback()
        print(f"[ERROR] Failed to initialize database: {e}")
        raise e  
    finally:
        release_conn(conn)


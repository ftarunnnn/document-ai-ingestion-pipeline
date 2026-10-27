"""
Database Engine & Session Management.
Supports PostgreSQL (with pgvector) and local SQLite storage with auto-migration.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.models import Base

logger = logging.getLogger(__name__)

# Determine database URL
DB_URL = settings.DATABASE_URL
connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}

engine = create_engine(
    DB_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initializes tables in database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized with URL: {DB_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session scope."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

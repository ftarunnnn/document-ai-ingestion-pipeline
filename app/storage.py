"""
Document Storage & File System Manager.
Manages persistent local file storage for raw uploaded documents and metadata records.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


class StorageManager:
    """Handles raw file persistence, metadata indexing, and duplicate hash tracking."""

    def __init__(self, upload_dir: Path = settings.UPLOAD_FOLDER):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.upload_dir / "index.json"
        self._ensure_index()

    def _ensure_index(self) -> None:
        if not self.index_file.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Any]:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_index(self, data: Dict[str, Any]) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def find_duplicate_by_hash(self, sha256_hash: str) -> Optional[Dict[str, Any]]:
        """Checks if a document with this hash has already been stored."""
        index = self._read_index()
        for doc in index.values():
            if doc.get("sha256_hash") == sha256_hash:
                return doc
        return None

    def save_document(
        self,
        doc_metadata: Dict[str, Any],
        content: bytes,
    ) -> Dict[str, Any]:
        """
        Saves original document bytes to disk using unique document ID,
        and records its metadata in the registry.
        """
        doc_id = doc_metadata["document_id"]
        extension = doc_metadata["extension"]
        filename = f"{doc_id}{extension}"
        stored_path = self.upload_dir / filename

        # Write binary content
        with open(stored_path, "wb") as f:
            f.write(content)

        record = {
            **doc_metadata,
            "stored_filename": filename,
            "stored_path": str(stored_path.resolve()),
            "status": "UPLOADED",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "extracted_pages": 0,
            "chunk_count": 0,
        }

        index = self._read_index()
        index[doc_id] = record
        self._write_index(index)
        return record

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific document by ID."""
        index = self._read_index()
        return index.get(doc_id)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns all registered document records sorted by upload date descending."""
        index = self._read_index()
        docs = list(index.values())
        docs.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
        return docs

    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Updates metadata fields for a specific document."""
        index = self._read_index()
        if doc_id not in index:
            return None
        index[doc_id].update(updates)
        index[doc_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_index(index)
        return index[doc_id]

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document file and its index entry."""
        index = self._read_index()
        doc = index.pop(doc_id, None)
        if doc:
            stored_path = Path(doc.get("stored_path", ""))
            if stored_path.exists():
                try:
                    os.remove(stored_path)
                except OSError:
                    pass
            self._write_index(index)
            return True
        return False


storage_manager = StorageManager()

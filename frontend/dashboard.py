"""
Document AI Ingestion Pipeline - Streamlit Dashboard.
Modern, interactive control center for document uploading, validation,
pipeline monitoring, search, and AI question answering.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import requests
import streamlit as st

# Add parent directory to path so dashboard can import app modules directly if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.validator import DocumentValidationError, document_validator
from app.storage import storage_manager

# Page layout & styling
st.set_page_config(
    page_title="Document AI Ingestion Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern look
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3a 100%);
        border: 1px solid #3b4252;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .status-uploaded { background-color: #2e7d32; color: #fff; }
    .status-processing { background-color: #f57c00; color: #fff; }
    .status-indexed { background-color: #0288d1; color: #fff; }
    .status-failed { background-color: #c62828; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Endpoint configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def check_api_status() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL.rsplit('/api/v1', 1)[0]}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


api_online = check_api_status()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/documents.png", width=90)
    st.title("DocAI Pipeline")
    st.caption("End-to-End Multimodal Ingestion")
    st.markdown("---")

    mode = st.radio(
        "Navigation",
        [
            "📤 Document Upload",
            "📑 Document Registry",
            "⚙️ Pipeline Status",
            "🔍 Search & RAG Q&A",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("System Health")
    if api_online:
        st.success("🟢 FastAPI Backend: Online")
    else:
        st.info("🟡 Standalone Mode (Direct Storage)")

    st.caption(f"Allowed: {', '.join(sorted(list(settings.ALLOWED_EXTENSIONS))[:6])}...")
    st.caption(f"Max size: {settings.MAX_FILE_SIZE_MB} MB")


# View 1: Upload View
if mode == "📤 Document Upload":
    st.header("Upload Documents to Ingestion Pipeline")
    st.write(
        "Upload contracts, invoices, tables, reports, or scanned images. "
        "Files will be validated for security, MIME integrity, and cryptographic duplicates."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Select or drop files here",
            type=[ext.lstrip(".") for ext in settings.ALLOWED_EXTENSIONS],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, XLSX, TXT, Images (PNG, JPG, TIFF, WEBP)",
        )

        allow_duplicates = st.checkbox("Allow duplicate file uploads (Skip hash collision rejection)", value=False)

        if st.button("🚀 Process & Ingest Files", type="primary", disabled=not uploaded_files):
            progress_bar = st.progress(0)
            status_container = st.container()

            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                filename = uploaded_file.name

                try:
                    # Perform validation
                    meta = document_validator.validate(filename, file_bytes)

                    # Check duplicate
                    duplicate = storage_manager.find_duplicate_by_hash(meta["sha256_hash"])
                    if duplicate and not allow_duplicates:
                        st.warning(
                            f"⚠️ **{filename}** is a duplicate! "
                            f"(Matches document `{duplicate['document_id'][:8]}...` by SHA-256 hash)."
                        )
                    else:
                        saved = storage_manager.save_document(meta, file_bytes)
                        st.success(
                            f"✅ **{filename}** uploaded successfully! "
                            f"Document ID: `{saved['document_id']}` ({saved['file_size'] / 1024:.1f} KB)"
                        )

                except DocumentValidationError as err:
                    st.error(f"❌ Validation failed for **{filename}**: {str(err)}")
                except Exception as err:
                    st.error(f"❌ Error processing **{filename}**: {str(err)}")

                progress_bar.progress((i + 1) / len(uploaded_files))

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🛡️ Ingestion Security Checks</h4>
                <ul>
                    <li><b>Format Whitelist:</b> Restricts unsafe extensions</li>
                    <li><b>Magic Header Sniffing:</b> Detects spoofed extensions</li>
                    <li><b>SHA-256 Deduplication:</b> Prevents redundant storage</li>
                    <li><b>Path Sanitization:</b> Traversal protection</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# View 2: Registry View
elif mode == "📑 Document Registry":
    st.header("Document Ingestion Registry")
    docs = storage_manager.list_documents()

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Total Documents", len(docs))
    with col_m2:
        total_size_mb = sum(d.get("file_size", 0) for d in docs) / (1024 * 1024)
        st.metric("Total Storage", f"{total_size_mb:.2f} MB")
    with col_m3:
        unique_hashes = len(set(d.get("sha256_hash", "") for d in docs))
        st.metric("Unique Hashes", unique_hashes)

    st.markdown("---")

    if not docs:
        st.info("No documents have been uploaded yet. Head over to 'Upload Documents' to begin.")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc.get('original_filename')} — ID: `{doc.get('document_id')[:8]}...`"):
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Size:** {doc.get('file_size', 0) / 1024:.1f} KB")
                c1.write(f"**Type:** `{doc.get('mime_type')}`")
                c2.write(f"**Uploaded:** {doc.get('uploaded_at', '')[:19]}")
                c2.write(f"**Status:** `{doc.get('status', 'UPLOADED')}`")
                c3.write(f"**SHA-256:** `{doc.get('sha256_hash', '')[:16]}...`")

                if st.button("🗑️ Delete Document", key=f"del_{doc.get('document_id')}"):
                    storage_manager.delete_document(doc.get("document_id"))
                    st.rerun()


# View 3: Pipeline Status
elif mode == "⚙️ Pipeline Status":
    st.header("Pipeline Orchestration Stages")
    st.info("Pipeline stages: `UPLOADED` ➔ `EXTRACTING` ➔ `CLEANING` ➔ `CHUNKING` ➔ `INDEXING` ➔ `INDEXED`")
    docs = storage_manager.list_documents()
    for d in docs:
        st.write(f"• **{d.get('original_filename')}**: Status `{d.get('status')}`")


# View 4: Search & Q&A
elif mode == "🔍 Search & RAG Q&A":
    st.header("Semantic Search & Source-Grounded Q&A")
    query = st.text_input("Ask a question about your ingested documents:")
    if query:
        st.info("Semantic Search & RAG engine will be active in Phase 8.")

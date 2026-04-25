"""
AcityBot RAG System
====================
Manual RAG implementation — no LangChain / LlamaIndex.
"""
# Student Name : Malvin Yawson
# Index Number  : 10022200168
# Course        : CS4241 - Introduction to Artificial Intelligence
# Institution   : Academic City University College, Ghana
# Year          : 2026
# ─────────────────────────────────────────────────────────────────────────────

from .pipeline import RAGPipeline
from .logger import ExperimentLogger
from .data_loader import load_all_documents
from .chunker import chunk_documents, compare_chunking_strategies
from .embedder import EmbeddingPipeline
from .vector_store import FAISSVectorStore
from .retriever import Retriever
from .prompt_builder import PromptBuilder

__all__ = [
    "RAGPipeline",
    "ExperimentLogger",
    "load_all_documents",
    "chunk_documents",
    "compare_chunking_strategies",
    "EmbeddingPipeline",
    "FAISSVectorStore",
    "Retriever",
    "PromptBuilder",
]

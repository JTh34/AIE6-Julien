import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

from enum import Enum

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

class RAGMode(Enum):
    DISABLED = "disabled"
    RAG_FIRST = "rag_first"
    RAG_ONLY = "rag_only"
    WEB_FIRST = "web_first"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    max_web_research_loops: int = int(os.environ.get("MAX_WEB_RESEARCH_LOOPS", "3"))
    local_llm: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", SearchAPI.DUCKDUCKGO.value))  # Default to DUCKDUCKGO
    fetch_full_page: bool = os.environ.get("FETCH_FULL_PAGE", "False").lower() in ("true", "1", "t")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/")
    # RAG configuration
    rag_mode: RAGMode = RAGMode(os.environ.get("RAG_MODE", RAGMode.RAG_FIRST.value))
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
    rag_collection_name: str = os.environ.get("RAG_COLLECTION", "puppy_books")
    min_rag_confidence: float = float(os.environ.get("MIN_RAG_CONFIDENCE", "0.5"))
    max_rag_results: int = int(os.environ.get("MAX_RAG_RESULTS", "5"))

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
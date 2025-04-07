import os
from typing import List, Dict, Any
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from aimakerspace.openai_utils.embedding import EmbeddingModel


class DocumentLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path):
            if self.path.endswith(".txt"):
                self.load_text_file()
            elif self.path.endswith(".pdf"):
                self.load_pdf_file()
            else:
                raise ValueError(
                    "Provided file is neither a .txt nor a .pdf file."
                )
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a valid file."
            )

    def load_text_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_pdf_file(self):
        with open(self.path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding=self.encoding) as f:
                        self.documents.append(f.read())
                elif file.endswith(".pdf"):
                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text() + "\n"
                        self.documents.append(text)

    def load_documents(self):
        self.load()
        return self.documents


class TextFileLoader(DocumentLoader):
    """Class of TextFileLoader for compatibility with existing code."""
    pass


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class VectorDatabaseWithMetadata:
    def __init__(self, embedding_model: EmbeddingModel = None, embedding_model_name: str = None, dimensions: int = None):
        self.vectors = {}  # Dictionary with ID -> vector
        self.documents = {}  # Dictionary with ID -> text content
        self.metadata = {}  # Dictionary with ID -> metadata
        
        # Elegant initialization of the embedding model
        if embedding_model is not None:
            # Use the provided embedding model
            self.embedding_model = embedding_model
        else:
            # Create a new embedding model with the specified parameters
            kwargs = {}
            if embedding_model_name is not None:
                kwargs["embeddings_model_name"] = embedding_model_name
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            
            self.embedding_model = EmbeddingModel(**kwargs)
            
        self.id_counter = 0

    def insert(self, text: str, vector: np.array, metadata: Dict[str, Any] = None):
        doc_id = str(self.id_counter)
        self.id_counter += 1
        self.vectors[doc_id] = vector
        self.documents[doc_id] = text
        self.metadata[doc_id] = metadata or {}

    async def abuild_from_document_chunks(self, chunks: List[Dict[str, Any]]) -> "VectorDatabaseWithMetadata":
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await self.embedding_model.async_get_embeddings(texts)
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.insert(
                text=chunk["content"],
                vector=np.array(embedding),
                metadata=chunk["metadata"]
            )
        
        return self
    
    def search(self, query_vector: np.array, k: int, include_metadata: bool = True):
        scores = []
        # Reshape query_vector to 2D if it's 1D
        query_vector_2d = query_vector.reshape(1, -1)
        
        for doc_id, vector in self.vectors.items():
            # Reshape document vector to 2D if it's 1D
            vector_2d = vector.reshape(1, -1)
            # Calculate cosine similarity
            similarity = cosine_similarity(query_vector_2d, vector_2d)[0][0]
            scores.append((doc_id, similarity))
        
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        if include_metadata:
            return [(self.documents[doc_id], similarity, self.metadata[doc_id]) for doc_id, similarity in top_k]
        else:
            return [(self.documents[doc_id], similarity) for doc_id, similarity in top_k]

    async def search_by_text(self, query_text: str, k: int, include_metadata: bool = True):
        query_vector = await self.embedding_model.async_get_embedding(query_text)
        return self.search(np.array(query_vector), k, include_metadata)


class PDFLoader:
    def __init__(self, path: str):
        self.path = path
        self.documents = []
        self.metadata = {}
        
    def load(self):
        with open(self.path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            # Extract global document metadata
            doc_metadata = pdf_reader.metadata
            self.metadata = {
                "title": doc_metadata.get("/Title", ""),
                "author": doc_metadata.get("/Author", ""),
                "subject": doc_metadata.get("/Subject", ""),
                "keywords": doc_metadata.get("/Keywords", ""),
                "creator": doc_metadata.get("/Creator", ""),
                "producer": doc_metadata.get("/Producer", ""),
                "creation_date": doc_metadata.get("/CreationDate", ""),
                "modification_date": doc_metadata.get("/ModDate", ""),
                "total_pages": len(pdf_reader.pages),
                "filename": os.path.basename(self.path)
            }
            
            # Extract text and metadata for each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                page_metadata = {
                    **self.metadata,  # Include global metadata
                    "page_number": page_num + 1,
                    "page_size": {"width": page.mediabox.width, "height": page.mediabox.height},
                }
                
                self.documents.append({
                    "content": text,
                    "metadata": page_metadata
                })
                
        return self.documents


class MetadataCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = document["content"]
        metadata = document["metadata"]
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            chunk_metadata = {
                **metadata,
                "chunk_index": len(chunks),
                "chunk_start_char": i,
                "chunk_end_char": min(i + self.chunk_size, len(text)),
                "is_first_chunk": i == 0,
                "is_last_chunk": i + self.chunk_size >= len(text),
            }
            
            chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata
            })
            
        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for doc in documents:
            chunks.extend(self.split(doc))
        return chunks


if __name__ == "__main__":
    # Test with a text file
    loader = DocumentLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(f"Nombre de chunks dans le fichier texte : {len(chunks)}")
    print(chunks[0][:100])

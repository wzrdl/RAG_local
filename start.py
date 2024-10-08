from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
from data_process import DocumentLoader

class DocumentProcessor:
    def __init__(self, document_loader, embedding_model=None, chunk_size=500, chunk_overlap=150, persist_directory='docs/chroma/'):

        self.document_loader = document_loader
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory

        # Use the provided embedding model, or default to OpenAIEmbeddings
        self.embedding = embedding_model if embedding_model else OpenAIEmbeddings()
        
    def load_and_split_documents(self):
        docs = self.document_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(docs)
        return split_docs

    def embed_and_store(self, split_docs, batch_size=16):
        """
        Embed the document chunks and store them in the Chroma vector store.

        Args:
            split_docs: List of document chunks.
            batch_size: Number of chunks to process per batch (due to API limits).
        """
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=self.embedding,
                persist_directory=self.persist_directory
            )
            vectordb.persist()
        print(f"Documents stored successfully in {self.persist_directory}")

# Example Usage:
from data_process import DocumentLoader

# Initialize the document loader (custom class, e.g., for PDFs)
pdf_loader = DocumentLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf", "pdf")

# Initialize the DocumentProcessor
doc_processor = DocumentProcessor(document_loader=pdf_loader)

# Load and split the documents
split_docs = doc_processor.load_and_split_documents()

# Embed and store them in the Chroma vector store
doc_processor.embed_and_store(split_docs)


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

pdf_loader = DocumentLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf", "pdf")
pdf_pages = pdf_loader.load()

docs = pdf_pages
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)# 这里替换需要加载的文本

# print(splits[0])
# 使用开源的embedding模型
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# 这里选择自己需要的embedding模型
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
embedding = OpenAIEmbeddings()

persist_directory = 'docs/chroma/'

# 由于接口限制，每次只能传16个文本块，需要循环分批传入
for i in range(0, len(split_docs), 16):
    batch = split_docs[i:i+16]
    vectordb = Chroma.from_documents(
        documents=batch,
        embedding=embedding,
        persist_directory=persist_directory
    )
vectordb.persist() #保存下来，后面可以直接使用
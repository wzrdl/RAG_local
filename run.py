from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.document_loaders import Docx2txtLoader, UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from local_llm import My_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class DocumentQA:
    def __init__(self, model_path: str, persist_directory: str):
        # Initialize LLM
        self.llm = My_LLM(model_path=model_path)
        # Initialize embeddings
        self.embedding = OpenAIEmbeddings()
        # Load vector database
        self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        # Define the prompt template
        self.template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
                        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
                        \n{context}\n问题: {question}"""
        self.qa_chain_prompt = PromptTemplate(input_variables=["context", "question"], template=self.template)
        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_chain_prompt}
        )

    def search_documents(self, question: str, k: int = 3) -> List[str]:
        # Perform similarity search
        docs = self.vectordb.similarity_search(question, k=k)
        return docs

    def answer_question(self, question: str) -> str:
        # Get the answer from the QA chain
        result = self.qa_chain({"query": question})
        return result["result"]

    def get_document_count(self) -> int:
        # Get the count of documents in the vector database
        return self.vectordb._collection.count()


if __name__ == "__main__":
    # Initialize the DocumentQA system
    model_path = "/root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b"
    persist_directory = 'docs/chroma/'
    document_qa = DocumentQA(model_path=model_path, persist_directory=persist_directory)
    # Print the number of documents in the database
    #print(f"Number of documents in the database: {document_qa.get_document_count()}")

    # Example question
    question = "什么是南瓜书？"
    print(f"Answer to '{question}':")
    print(document_qa.answer_question(question))

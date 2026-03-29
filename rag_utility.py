import os

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))  # automatically locate the path of the documents

PERSIST_DIR = os.path.join(working_dir, "doc_vectorstore")

_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Single PersistentClient per process avoids Chroma shared-system / tenant init issues on reruns."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


embedding = HuggingFaceEmbeddings()

# Load the model
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature=0
)




def process_document_to_chromadb(file_name):
    loader = PyPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )

    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        client=_get_chroma_client(),
    )
    return 0



def answer_question(user_question):
    vectordb = Chroma(
        client=_get_chroma_client(),
        embedding_function=embedding,
    )
    retriever = vectordb.as_retriever()

    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]

    return answer

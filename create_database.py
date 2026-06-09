# Load PDF
# Split into chunks
# Create embeddings
# Store into ChromaDB

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "document loaders/deeplearning.pdf"
)

docs = loader.load()

# Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Mistral Embedding Model
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Embeddings stored successfully!")
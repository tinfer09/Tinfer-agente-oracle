import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(documents, persist_directory="./docs/chroma_db"):
    """
    Divide los documentos en fragmentos, crea los embeddings con Google Generative AI
    y los guarda en una base de datos vectorial ChromaDB.
    """
    print("Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total de fragmentos generados: {len(chunks)}")
    
    # Crear embeddings con Gemini
    print("Generando embeddings y guardando en ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Chroma en LangChain >= 0.1 hace persist automático, pero es buena práctica indicarlo.
    vectorstore.persist()
    print(f"Base de datos vectorial guardada en {persist_directory}")
    return vectorstore

def get_vector_store(persist_directory="./docs/chroma_db"):
    """Carga la base de datos vectorial existente."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

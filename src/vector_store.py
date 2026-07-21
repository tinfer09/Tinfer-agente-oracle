import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def create_vector_store(documents, persist_directory="./docs/chroma_db"):
    """
    Divide los documentos en fragmentos, crea los embeddings con un modelo local
    y los guarda en una base de datos vectorial ChromaDB.
    """
    print("Dividiendo documentos en fragmentos (chunks)...")
    
    # Si la base de datos ya existía con otro modelo, la borramos para evitar conflictos de dimensiones
    if os.path.exists(persist_directory):
        print("Borrando base de datos anterior para regenerarla con el nuevo modelo de embeddings...")
        shutil.rmtree(persist_directory)
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total de fragmentos generados: {len(chunks)}")
    
    # Usamos FastEmbed (completamente local, rápido y sin PyTorch)
    print("Generando embeddings locales y guardando en ChromaDB...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Base de datos vectorial guardada en {persist_directory}")
    return vectorstore

def get_vector_store(persist_directory="./docs/chroma_db"):
    """Carga la base de datos vectorial existente para realizar consultas."""
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore

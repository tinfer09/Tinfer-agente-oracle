import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(pdf_dir: str) -> list[Document]:
    """Carga todos los PDFs del directorio y los devuelve como una lista de Documents."""
    documents = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    for pdf_path in pdf_files:
        print(f"Cargando PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

def load_csvs(csv_dir: str) -> dict[str, pd.DataFrame]:
    """Carga todos los CSVs del directorio y los devuelve en un diccionario de DataFrames."""
    dataframes = {}
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    for csv_path in csv_files:
        print(f"Cargando CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Usar el nombre del archivo sin extensión como clave
        filename = os.path.basename(csv_path).replace(".csv", "")
        dataframes[filename] = df
    return dataframes

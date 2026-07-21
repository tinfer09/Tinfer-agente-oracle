import os
import glob
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(pdf_dir: str) -> list[Document]:
    """Carga todos los PDFs del directorio y los devuelve como una lista de Documents."""
    documents = []
    
    # Busco todos los archivos con extensión .pdf en la carpeta que le indico (docs/pdfs).
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    for pdf_path in pdf_files:
        print(f"Cargando PDF: {pdf_path}")
        
        # Uso PyPDFLoader, que es la herramienta de LangChain para leer y extraer texto de PDFs.
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Añado el contenido a mi lista general de documentos.
        documents.extend(docs)
    return documents

def load_csvs(csv_dir: str) -> dict[str, pd.DataFrame]:
    """Carga todos los CSVs del directorio y los devuelve en un diccionario de DataFrames."""
    dataframes = {}
    
    # Busco todos los archivos con extensión .csv en la carpeta que le indico (docs/csv).
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    for csv_path in csv_files:
        print(f"Cargando CSV: {csv_path}")
        
        # Uso Pandas para leer el archivo CSV y cargarlo en memoria como un DataFrame (una tabla).
        df = pd.read_csv(csv_path)
        
        # Uso el nombre del archivo (sin la extensión) como clave para guardar la tabla en mi diccionario.
        filename = os.path.basename(csv_path).replace(".csv", "")
        dataframes[filename] = df
        
    return dataframes

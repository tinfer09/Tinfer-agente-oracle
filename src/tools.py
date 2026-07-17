from langchain.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from .vector_store import get_vector_store

@tool
def buscar_en_documentos_tinfer(query: str) -> str:
    """Busca y devuelve información de los manuales, políticas y arquitecturas de Tinfer."""
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

def create_rag_tool():
    """Devuelve la herramienta para consultar PDFs."""
    return buscar_en_documentos_tinfer

def create_csv_tools(dataframes_dict: dict[str, pd.DataFrame]):
    """
    Crea herramientas basadas en Pandas DataFrame Agent para cada CSV.
    Retorna una lista de tools o configuraciones para el agente.
    """
    # En LangChain, un agente Pandas no es una "Tool" simple por defecto,
    # sino que es un agente en sí mismo. Para integrarlo como tool,
    # podemos usar una Tool personalizada que llame al agente Pandas.
    from langchain.tools import Tool
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    tools = []
    
    for name, df in dataframes_dict.items():
        # Creamos un agente para este dataframe
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            agent_type="openai-tools", # OpenAI-tools format is well supported by many LLMs
            allow_dangerous_code=True # Requerido en versiones nuevas para ejecutar código Python (pandas)
        )
        
        description = f"Útil para responder preguntas sobre los datos del archivo {name}.csv. Entrada: la pregunta del usuario."
        
        # Creamos una tool que invoca al agente de Pandas
        tool = Tool(
            name=f"consultar_datos_{name}",
            func=agent.invoke,
            description=description
        )
        tools.append(tool)
        
    return tools

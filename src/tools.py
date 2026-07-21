from langchain_core.tools import tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import pandas as pd
from .vector_store import get_vector_store

from pydantic import BaseModel, Field

class RAGToolInput(BaseModel):
    query: str = Field(description="La consulta o pregunta específica para buscar en los documentos. Ej: 'políticas de la empresa'")

# Defino mi primera herramienta usando el decorador @tool. 
# Esto le dice al agente que esta función puede ser invocada por la IA.
@tool("buscar_en_documentos_tinfer", args_schema=RAGToolInput)
def buscar_en_documentos_tinfer(query: str) -> str:
    """Útil y OBLIGATORIO para buscar información en los manuales, políticas, reglas y arquitectura de Tinfer. DEBES usar esto cuando te pregunten sobre la empresa, políticas, o documentos."""
    # Obtengo la conexión a mi base de datos vectorial ChromaDB.
    vectorstore = get_vector_store()
    
    # Configuro el "retriever" para que me devuelva los 4 fragmentos (chunks) más relevantes de mis PDFs.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Hago la búsqueda usando la consulta (query) del usuario.
    docs = retriever.invoke(query)
    
    # Concateno el texto de los fragmentos encontrados y los devuelvo como respuesta para que el LLM lo lea.
    return "\n\n".join([d.page_content for d in docs])

def create_rag_tool():
    """Devuelve la herramienta para consultar PDFs."""
    return buscar_en_documentos_tinfer

def create_csv_tools(dataframes_dict: dict[str, pd.DataFrame]):
    """
    Crea herramientas basadas en Pandas DataFrame Agent para cada CSV.
    Retorna una lista de tools listas para que el agente principal las use.
    """
    from langchain_core.tools import Tool
    
    # Vuelvo a instanciar el modelo de Groq para el agente de Pandas, 
    # usando una temperatura de 0 para que sea estrictamente analítico con los datos.
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = []
    
    # Itero sobre cada CSV que cargué en memoria.
    for name, df in dataframes_dict.items():
        # Creo un agente especializado en entender este DataFrame específico.
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            agent_type="tool-calling", # Cambié a 'tool-calling' que es más compatible con Gemini 1.5.
            allow_dangerous_code=True # Esto es necesario para que el agente pueda ejecutar código Python de Pandas internamente de forma segura aquí.
        )
        
        # Le doy una descripción clara para que el agente principal sepa CUÁNDO debe usar esta herramienta.
        description = f"Útil para responder preguntas sobre los datos del archivo {name}.csv. Entrada: la pregunta del usuario."
        
        # Definimos una función envoltorio (wrapper) para evitar errores de firmas de tipos 
        # (TypeError: 'function' object is not subscriptable) que ocurren cuando LangGraph
        # intenta inspeccionar la función interna de agent.invoke en versiones nuevas de Python.
        def wrapper_func(query: str, agent_instance=agent) -> str:
            # El agente de Pandas recibe un diccionario con la clave "input"
            response = agent_instance.invoke({"input": query})
            # Devolvemos solo el texto de la respuesta final
            return str(response.get("output", response))
            
        from pydantic import BaseModel, Field
        
        class CSVToolInput(BaseModel):
            query: str = Field(description="La pregunta o consulta sobre los datos del archivo")

        # Envuelvo el agente de Pandas en una herramienta estándar de LangChain con su schema explícito.
        csv_tool = Tool(
            name=f"consultar_datos_{name}",
            func=wrapper_func,
            description=description,
            args_schema=CSVToolInput
        )
        tools.append(csv_tool)
        
    return tools

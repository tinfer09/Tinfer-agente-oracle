import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from .document_loader import load_csvs
from .tools import create_rag_tool, create_csv_tools

# Aquí cargo las variables de entorno, como mi GOOGLE_API_KEY que configuré en el .env
load_dotenv()

def create_agent():
    """Inicializa y configura el agente de IA para Tinfer usando LangGraph."""
    # 1. Instancio el modelo LLM de Groq. 
    # Usamos llama-3.3-70b-versatile para obtener razonamiento complejo de forma rápida y gratuita.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.3 # Mantengo una temperatura baja para que las respuestas sean precisas y menos "creativas" (menos alucinaciones).
    )
    
    # 2. Cargo las herramientas que mi agente podrá usar.
    # Primero, cargo la herramienta RAG que busca en la base de datos de mis PDFs.
    rag_tool = create_rag_tool()
    
    # Luego, defino dónde están mis archivos CSV y los cargo en memoria.
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "csv")
    dataframes = load_csvs(csv_dir)
    
    # Creo herramientas específicas para consultar esos DataFrames de Pandas.
    csv_tools = create_csv_tools(dataframes)
    
    # Combino todas las herramientas en una sola lista que le pasaré a mi agente.
    tools = [rag_tool] + csv_tools
    
    # 3. Creo el mensaje de sistema (system prompt). 
    # Aquí le doy a mi agente su personalidad y las reglas que debe seguir.
    system_prompt = """Eres TinferBot, el asistente de inteligencia artificial interno de Tinfer, una empresa SaaS.
Tu objetivo es ayudar a los empleados de Tinfer a encontrar información en los documentos y bases de datos internas.

Reglas:
1. Responde siempre en español, de forma amable y profesional.
2. Si te preguntan sobre políticas, manuales o arquitectura, usa la herramienta `buscar_en_documentos_tinfer`.
3. Si te preguntan sobre datos de clientes, empleados o incidentes, usa las herramientas de consulta de datos correspondientes.
4. Si no sabes la respuesta, di amablemente que no tienes esa información.
"""
    
    # 4. Creo el agente usando LangGraph. 
    # Esto orquesta cómo el LLM razona (ReAct) y decide qué herramienta usar según la pregunta del usuario.
    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
    
    return agent_executor

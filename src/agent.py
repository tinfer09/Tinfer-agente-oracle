import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from .document_loader import load_csvs
from .tools import create_rag_tool, create_csv_tools

load_dotenv()

def create_agent():
    """Inicializa y configura el agente de IA para Tinfer usando LangGraph."""
    # 1. Instanciar el modelo LLM de Google
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.3 # Baja temperatura para respuestas más precisas
    )
    
    # 2. Cargar las herramientas
    rag_tool = create_rag_tool()
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "csv")
    dataframes = load_csvs(csv_dir)
    csv_tools = create_csv_tools(dataframes)
    
    # Combinar todas las herramientas
    tools = [rag_tool] + csv_tools
    
    # 3. Crear el mensaje de sistema
    system_prompt = """Eres TinferBot, el asistente de inteligencia artificial interno de Tinfer, una empresa SaaS.
Tu objetivo es ayudar a los empleados de Tinfer a encontrar información en los documentos y bases de datos internas.

Reglas:
1. Responde siempre en español, de forma amable y profesional.
2. Si te preguntan sobre políticas, manuales o arquitectura, usa la herramienta `buscar_en_documentos_tinfer`.
3. Si te preguntan sobre datos de clientes, empleados o incidentes, usa las herramientas de consulta de datos correspondientes.
4. Si no sabes la respuesta, di amablemente que no tienes esa información.
"""
    
    # 4. Crear el agente usando LangGraph (estándar moderno)
    agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    return agent_executor

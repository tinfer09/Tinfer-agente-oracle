import os
import streamlit as st
from dotenv import load_dotenv
from src.agent import create_agent
from src.document_loader import load_pdfs
from src.vector_store import create_vector_store

load_dotenv()

# --- Configuración de la página ---
st.set_page_config(
    page_title="TinferBot - IA Interna",
    page_icon="🤖",
    layout="wide"
)

# Estilos CSS personalizados para darle look de empresa SaaS moderna
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
    }
    .user-msg {
        background-color: #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #3b82f6;
    }
    .bot-msg {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de inicialización ---
@st.cache_resource
def initialize_system():
    """Inicializa la base de datos vectorial si no existe y carga el agente."""
    # Verificar si existe la base de datos Chroma
    chroma_path = os.path.join(os.path.dirname(__file__), "docs", "chroma_db")
    if not os.path.exists(chroma_path):
        st.info("🔄 Inicializando base de conocimientos (procesando PDFs)... Esto puede tomar unos minutos la primera vez.")
        pdf_dir = os.path.join(os.path.dirname(__file__), "docs", "pdfs")
        documents = load_pdfs(pdf_dir)
        create_vector_store(documents, persist_directory=chroma_path)
        st.success("✅ Base de conocimientos creada exitosamente.")
    
    return create_agent()

# --- UI Principal ---
st.markdown("<h1 class='main-header'>🤖 TinferBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>Tu asistente de IA para consultar manuales, políticas y datos internos de Tinfer.</p>", unsafe_allow_html=True)

# Inicializar agente
try:
    agent_executor = initialize_system()
except Exception as e:
    st.error(f"Error inicializando el sistema: {str(e)}")
    st.stop()

# Inicializar historial del chat en la sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy TinferBot. Puedes preguntarme sobre políticas internas, arquitectura, incidentes pasados o datos de clientes y empleados. ¿En qué puedo ayudarte hoy?"}
    ]

# Mostrar mensajes anteriores
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'><b>Tú:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'><b>TinferBot:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

# Input del usuario
user_input = st.chat_input("Escribe tu pregunta aquí (ej: '¿Cuál es la política de devoluciones?')")

if user_input:
    # Mostrar mensaje del usuario
    st.markdown(f"<div class='user-msg'><b>Tú:</b><br>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Mostrar spinner mientras procesa
    with st.spinner("TinferBot está buscando información y pensando..."):
        try:
            # Invocar al agente (formato LangGraph)
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            # La respuesta está en el último mensaje generado
            bot_response = response["messages"][-1].content
            
            # Mostrar respuesta del bot
            st.markdown(f"<div class='bot-msg'><b>TinferBot:</b><br>{bot_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            st.error(f"Ocurrió un error al procesar tu consulta: {str(e)}")

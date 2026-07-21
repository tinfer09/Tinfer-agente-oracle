import os
import streamlit as st
from dotenv import load_dotenv
from src.agent import create_agent
from src.document_loader import load_pdfs
from src.vector_store import create_vector_store

# Cargar las variables de entorno, principalmente mi GOOGLE_API_KEY.
load_dotenv()

# --- Configuración de la página ---
# Aquí configuro los detalles básicos de cómo se verá la pestaña de mi aplicación en el navegador.
st.set_page_config(
    page_title="TinferBot - IA Interna",
    page_icon="🤖",
    layout="wide"
)

# Estilos CSS personalizados para darle look de empresa SaaS moderna (colores corporativos, bordes redondeados).
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a !important;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0;
    }
    .user-msg {
        background-color: #e2e8f0;
        color: #1e293b !important;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #3b82f6;
    }
    .bot-msg {
        background-color: #ffffff;
        color: #1e293b !important;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #10b981;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    /* Aseguramos que el subtítulo también sea oscuro */
    .sub-header {
        text-align: center; 
        color: #64748b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones de inicialización ---
# Uso el decorador @st.cache_resource para asegurar que esto solo se ejecute una vez
# y no cada vez que interactúo con la interfaz.
@st.cache_resource
def initialize_system():
    """Inicializa la base de datos vectorial si no existe y carga el agente."""
    
    # Defino las rutas a mis carpetas de conocimiento
    chroma_path = os.path.join(os.path.dirname(__file__), "docs", "chroma_db")
    pdf_dir = os.path.join(os.path.dirname(__file__), "docs", "pdfs")
    
    # Aseguro que la carpeta de pdfs exista (si no, da error la primera vez)
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Si mi base de datos de PDFs (ChromaDB) no existe, significa que es la primera vez que arranco la app.
    if not os.path.exists(chroma_path):
        st.info("🔄 Inicializando base de conocimientos (procesando PDFs)... Esto puede tomar unos minutos la primera vez.")
        # Cargo los PDFs de la carpeta y los vectorizo
        documents = load_pdfs(pdf_dir)
        if documents: # Solo si hay documentos para procesar
            create_vector_store(documents, persist_directory=chroma_path)
            st.success("✅ Base de conocimientos creada exitosamente.")
        else:
            st.warning("⚠️ No se encontraron PDFs en docs/pdfs para vectorizar. Por favor, asegúrate de colocarlos.")
            
    # Una vez que la base está lista, creo y devuelvo el agente configurado.
    return create_agent()

# --- UI Principal ---
# Cabecera de bienvenida en la interfaz.
st.markdown("<h1 class='main-header'>🤖 TinferBot</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Tu asistente de IA para consultar manuales, políticas y datos internos de Tinfer.</p>", unsafe_allow_html=True)

# Intento inicializar el sistema (cargar el agente y la base de datos vectorial).
try:
    agent_executor = initialize_system()
except Exception as e:
    st.error(f"Error inicializando el sistema: {str(e)}")
    st.stop()

# Inicializar historial del chat en la sesión de Streamlit para no perder los mensajes al recargar la página.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy TinferBot. Puedes preguntarme sobre políticas internas, arquitectura, incidentes pasados o datos de clientes y empleados. ¿En qué puedo ayudarte hoy?"}
    ]

# Dibujar todos los mensajes anteriores que están guardados en la sesión.
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'><b>Tú:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'><b>TinferBot:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

# Cuadro de entrada (input) del usuario.
user_input = st.chat_input("Escribe tu pregunta aquí (ej: '¿Cuál es la política de devoluciones?')")

if user_input:
    # 1. Muestro el mensaje del usuario en la pantalla y lo guardo en la sesión.
    st.markdown(f"<div class='user-msg'><b>Tú:</b><br>{user_input}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Muestro un indicador de carga (\"spinner\") mientras espero a que Gemini responda.
    with st.spinner("TinferBot está buscando información y pensando..."):
        try:
            # Invoco a mi agente de LangGraph pasándole la consulta del usuario.
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            
            # La respuesta real del bot estará en el último mensaje de la lista de respuestas de LangGraph.
            bot_response = response["messages"][-1].content
            
            # 3. Muestro la respuesta del bot en la pantalla y la guardo en la sesión.
            st.markdown(f"<div class='bot-msg'><b>TinferBot:</b><br>{bot_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            st.error(f"Ocurrió un error al procesar tu consulta: {str(e)}")

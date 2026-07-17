# 🤖 TinferBot - Agente de Inteligencia Artificial

TinferBot es un agente de IA conversacional diseñado para uso interno en **Tinfer** (empresa SaaS). Permite a los empleados consultar manuales, políticas corporativas y bases de datos (clientes, incidentes, empleados) usando lenguaje natural.

## 🏗️ Arquitectura de la Solución

El sistema utiliza un enfoque **RAG (Retrieval-Augmented Generation)** y herramientas de análisis de datos estructurados (Pandas):

1. **Agente Orquestador**: Usa LangChain (Tool Calling Agent) para decidir qué herramienta usar según la pregunta del usuario.
2. **LLM**: Google Gemini 2.0 Flash (`gemini-2.0-flash`).
3. **RAG para Documentos (PDFs)**:
   - Los PDFs se dividen usando `RecursiveCharacterTextSplitter`.
   - Se vectorizan con `GoogleGenerativeAIEmbeddings`.
   - Se almacenan localmente en **ChromaDB**.
4. **Análisis de Datos (CSVs)**:
   - Se utiliza `create_pandas_dataframe_agent` de LangChain Experimental para convertir preguntas en código pandas internamente y responder consultas sobre los CSVs.
5. **Interfaz de Usuario**: Construida con **Streamlit**, ofreciendo una experiencia moderna y fluida.

## 🛠️ Tecnologías y Herramientas

- **Python 3.10+**
- **LangChain** (Framework principal)
- **Google Generative AI** (Gemini LLM y Embeddings)
- **ChromaDB** (Vector Store)
- **PyPDF** (Lectura de PDFs)
- **Pandas** (Análisis de CSVs)
- **Streamlit** (UI/UX Frontend)

## 🚀 Cómo ejecutar el proyecto localmente

1. Clona este repositorio o asegúrate de estar en el directorio `Tinfer-agente-oracle`.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configura tu API Key de Gemini creando un archivo `.env` en la raíz (puedes basarte en este ejemplo):
   ```
   GOOGLE_API_KEY=tu_api_key_aqui
   ```
4. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```
5. La primera vez que se ejecute, el sistema procesará los PDFs y creará la base vectorial `docs/chroma_db/`. Esto puede tomar un minuto.

## 💡 Ejemplos de Interacción

### ❓ Preguntas al Agente
- **Sobre manuales (PDFs):** *"¿Qué debo hacer si ocurre un incidente de severidad 1?"*
- **Sobre manuales (PDFs):** *"¿Cómo es la arquitectura de microservicios de Tinfer?"*
- **Sobre datos (CSVs):** *"¿Cuántos clientes SaaS tenemos y cuál es el de mayor ARR?"*
- **Sobre datos (CSVs):** *"¿Quiénes son los ingenieros de software de la empresa y en qué modalidad trabajan?"*

### 💬 Ejemplo de Respuesta Generada
> **Tú:** ¿Cuál fue el incidente más largo registrado y cuál fue su causa?
> 
> **TinferBot:** El incidente más largo fue el INC-2026-03 en el Dashboard de Analíticas, que duró 240 minutos. La causa raíz fue una falla en el pipeline de Snowflake. La responsable del post-mortem es Valentina Rios y actualmente está en estado "Investigando".

---
*Proyecto desarrollado para el Challenge Alura Agente - Oracle Next Education*
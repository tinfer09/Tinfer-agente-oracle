# 🤖 TinferBot - AI Corporate Assistant

[![Deploy Status](https://img.shields.io/badge/Render-Deployed-success?logo=render)](#) 
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-orange)](#)
[![Groq](https://img.shields.io/badge/LLM-Llama_3-purple)](#)

TinferBot es un agente de Inteligencia Artificial conversacional diseñado para uso interno en **Tinfer** (empresa SaaS). 
Su objetivo es centralizar el conocimiento corporativo, permitiendo a los empleados consultar manuales, políticas de la empresa y bases de datos tabulares (clientes, incidentes, empleados) utilizando lenguaje natural.

🚀 **Prueba la versión en vivo:** [TinferBot en Render](https://tinfer-agente-oracle.onrender.com/)

---

## 🏗️ Arquitectura de la Solución

El sistema utiliza un enfoque moderno de **Agentes Autónomos** y **RAG (Retrieval-Augmented Generation)**:

1. **Agente Orquestador (LangGraph)**: Utiliza un flujo ReAct (Reasoning and Acting) con memoria conversacional para decidir dinámicamente qué herramienta invocar según el contexto de la pregunta y mantener el hilo de la charla.
2. **LLM Principal**: Modelo `llama-3.3-70b-versatile` procesado a ultra alta velocidad a través de **Groq**.
3. **Módulo RAG (Búsqueda Documental en PDFs)**:
   - División de texto inteligente mediante `RecursiveCharacterTextSplitter`.
   - Embeddings locales de alta eficiencia utilizando `FastEmbed (BAAI/bge-small-en-v1.5)`.
   - Almacenamiento y recuperación vectorial rápida mediante **ChromaDB**.
4. **Análisis de Datos Estructurados (CSVs)**:
   - Herramientas dinámicas generadas con `create_pandas_dataframe_agent` capaces de escribir y ejecutar código Python internamente para analizar DataFrames en tiempo real.
5. **Frontend**: Interfaz de usuario limpia, responsiva y con estilos corporativos construida con **Streamlit**.

## ☁️ Despliegue en la Nube (Render)

El proyecto está configurado bajo la metodología de **Infraestructura como Código (IaC)**.
Para desplegar automáticamente en [Render](https://render.com/), el repositorio incluye el blueprint `render.yaml`. 
El servicio web descargará las dependencias, inicializará el servidor en el puerto correcto y generará la base de datos vectorial de embeddings "al vuelo" a partir de los PDFs puros.

*(Nota: Solo debes asegurarte de configurar la variable de entorno `GROQ_API_KEY` en tu dashboard de Render).*

## 💻 Ejecución Local

Sigue estos 5 pasos exactos para levantar el agente en tu propia máquina desde cero:

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/tinfer09/Tinfer-agente-oracle.git
   cd Tinfer-agente-oracle
   ```

2. **Crear y activar el entorno virtual aisaldo**:
   ```bash
   python -m venv .venv
   
   # En Windows:
   .venv\Scripts\activate
   # En Mac/Linux:
   source .venv/bin/activate
   ```

3. **Instalar las dependencias exactas**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar las credenciales**:
   Crea un archivo `.env` en la raíz del proyecto y añade tu API Key de Groq:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```

5. **Iniciar la aplicación**:
   ```bash
   streamlit run app.py
   ```
   > ⏳ *La primera vez que arranques el proyecto localmente, la consola procesará los PDFs y construirá la base de datos de ChromaDB (pesa unos 100MB). Esto puede tomar un minuto.*

## 💡 Ejemplos de Interacción

El agente es capaz de enrutar tu pregunta a la herramienta correcta según lo que necesites saber. ¡Prueba preguntarle estas cosas!

### 📚 Consultas Documentales (RAG)
- *"¿Cómo es la arquitectura de microservicios de Tinfer?"*
- *"¿Cuál es el protocolo a seguir si ocurre un incidente de severidad 1?"*

### 📊 Consultas de Datos (Pandas Agent)
- *"¿Cuántos clientes SaaS activos tenemos y cuál es el que genera mayor ARR?"*
- *"Haz un resumen de los incidentes históricos del último mes."*

Demostración

![Image Alt](https://github.com/tinfer09/Tinfer-agente-oracle/blob/2b4dddca0ee03c278efa6406acc78ea1b7d99f15/Captura.PNG)
---
*Desarrollado como Proyecto Final para el Challenge Alura Agente - Oracle Next Education*

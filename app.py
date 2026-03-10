import streamlit as st
from openai import OpenAI
import os
import glob
import streamlit.components.v1 as components
from streamlit_mic_recorder import mic_recorder
import io
import zipfile

# IMPORTACIONES PARA LANGCHAIN
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    st.error("Faltan librerías. Instala con: pip install langchain-community langchain-text-splitters faiss-cpu pypdf sentence-transformers")
    st.stop()

# ────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA (DEBE SER LO PRIMERO)
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IA Prometeo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None, 'Report a bug': None, 'About': "IA Prometeo - Asistente Inteligente"}
)

# ────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE CARPETA Y CARGA DE DATOS
# ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "documentos")

# Usamos caché para no recargar los PDFs cada vez que interactuamos
@st.cache_resource(show_spinner="Cargando base de conocimiento...")
def load_knowledge_base():
    if not os.path.exists(DOCS_FOLDER):
        try:
            os.makedirs(DOCS_FOLDER)
        except OSError as e:
            st.error(f"Error al crear la carpeta 'documentos': {e}")
            return None, []

    pdf_files = glob.glob(os.path.join(DOCS_FOLDER, "*.pdf"))
    if not pdf_files:
        return None, []

    all_docs = []
    valid_files = []
    error_files = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            if not docs:
                continue
            filename = os.path.basename(pdf_path)
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            valid_files.append(filename)
        except Exception as e:
            error_files.append((os.path.basename(pdf_path), str(e)))

    if error_files:
        st.warning(f"⚠️ No se pudieron leer {len(error_files)} archivos.")
    
    if not all_docs:
        return None, []

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore, valid_files
    except Exception as e:
        st.error(f"Error al procesar embeddings: {e}")
        return None, []

# ────────────────────────────────────────────────────────────────
# CSS NEUTRO Y PROFESIONAL
# ────────────────────────────────────────────────────────────────
css_neutral = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* OCULTAR ELEMENTOS STREAMLIT POR DEFECTO */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stToolbar"] {display: none;}

    /* VARIABLES DE COLOR */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #6366F1;
        --bg-color: #F3F4F6;
        --sidebar-bg: #FFFFFF;
        --text-color: #1F2937;
    }

    /* FONDO GENERAL */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    /* HEADER */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: clamp(2rem, 6vw, 3rem);
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #6B7280;
        font-size: 1rem;
    }

    /* CHAT CONTENEDOR */
    .fixed-chat-wrapper {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* INPUT CHAT */
    [data-testid="stChatInput"] {
        border: 1px solid #D1D5DB !important;
        border-radius: 16px !important;
        background-color: #FFFFFF !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid #E5E7EB;
    }
    
    /* BOTONES */
    .stButton button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 10px !important;
    }
</style>
"""
st.markdown(css_neutral, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────────
header_html = """
<div class="main-header">
    <h1 class="main-title">IA PROMETEO 🧠</h1>
    <p class="subtitle">Tu asistente inteligente para documentos y planeación</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE API KEY
# ────────────────────────────────────────────────────────────────
api_key = None
if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
    api_key = st.secrets["groq"]["api_key"]

# ────────────────────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2>⚙️ Panel de Control</h2>", unsafe_allow_html=True)
    st.markdown("#### 🔑 Configuración")
    
    if not api_key:
        api_key_input = st.text_input("API Key de Groq", type="password", key="api_key_input_groq")
        if api_key_input:
            api_key = api_key_input
        else:
            st.warning("Necesitas API Key de Groq.")
            st.info("Obtén una en: console.groq.com")
    else:
        st.success("API Key configurada ✅")

    voice_enabled = st.checkbox("Activar respuestas de voz", value=True)
    
    st.markdown("---")
    st.markdown("#### 📂 Base de Conocimiento")
    st.caption(f"Carpeta: `documentos/`")
    
    uploaded_zip = st.file_uploader("Sube un ZIP con PDFs", type="zip", key="zip_uploader")
    if uploaded_zip:
        if "processed_zip_name" not in st.session_state or st.session_state.processed_zip_name != uploaded_zip.name:
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as z:
                    z.extractall(DOCS_FOLDER)
                st.session_state.processed_zip_name = uploaded_zip.name
                st.toast(f"✅ Archivos extraídos. Recargando...")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error al descomprimir: {e}")

    st.markdown("---")
    st.markdown("#### 📊 Estado")
    
    if st.button("🔄 Recargar Base de Datos", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    # Cargar base de datos
    if "vectorstore" not in st.session_state:
        vectorstore, loaded_files = load_knowledge_base()
        st.session_state.vectorstore = vectorstore
        st.session_state.loaded_files = loaded_files
    
    if st.session_state.get("loaded_files"):
        st.success(f"📚 {len(st.session_state.loaded_files)} Documentos Activos")
        with st.expander("Ver lista"):
            for f in st.session_state.loaded_files:
                st.write(f"📄 {f}")
    else:
        st.info("📂 Repositorio Vacío. Añade PDFs.")

if not api_key:
    st.stop()

try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
except Exception as e:
    st.error(f"Error al conectar con Groq: {e}")
    st.stop()

# ────────────────────────────────────────────────────────────────
# PERSONALIDAD Y MODO PLANEACIÓN
# ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_BASE = """
Eres **IA Prometeo**, un asistente inteligente, conciso y profesional.
Tu objetivo es ayudar al usuario a analizar documentos y realizar tareas de planeación o consulta.
Tono: Profesional, cercano y eficiente.
"""

loaded_files_list_str = "No hay archivos cargados."
if st.session_state.get("loaded_files"):
    loaded_files_list_str = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(st.session_state.loaded_files)])

SYSTEM_PROMPT_PLANNING = f"""
Eres **IA Prometeo - Experto en Planeación**.
**ARCHIVOS DISPONIBLES:**
{loaded_files_list_str}
-----------------------------------------
**REGLAS:**
1. Usa solo información del contexto.
2. Busca patrones como "Unidad", "Módulo", "Bloque".
3. Si no tienes información, indícalo.
**FLUJO:**
**PASO 1: ACTIVACIÓN**
Si el usuario dice "vamos a planear":
1. Muestra la lista de archivos.
2. Pregunta: "¿Cuál es el **número** del programa a utilizar?"
**PASO 2: LECTURA**
Cuando el usuario elija un número:
1. Identifica el archivo.
2. Lista las unidades encontradas numeradas.
3. Pregunta: "¿Qué **número** de unidad(es) vamos a planear?"
(Continúa con los pasos de sesiones, días, criterios, fechas...)"
"""

# ────────────────────────────────────────────────────────────────
# INICIALIZACIÓN DE SESIÓN
# ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "planning_mode" not in st.session_state:
    st.session_state.planning_mode = False

# ────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ────────────────────────────────────────────────────────────────
def get_audio_button_html(text, key):
    text_clean = text.replace("'", "").replace('"', '').replace("\n", " ")
    return f"""
    <div style="margin-top: 10px; text-align: right;">
        <button onclick="
            var u = new SpeechSynthesisUtterance('{text_clean}');
            u.lang = 'es-MX';
            u.rate = 0.95;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        " style="
            background-color: #4F46E5;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 500;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        ">🔊 Escuchar</button>
    </div>
    """

def get_context_for_planning(user_input, vectorstore, loaded_files):
    # Lógica simplificada de recuperación
    if not vectorstore: return "", None
    try:
        # Búsqueda simple por similitud
        docs = vectorstore.similarity_search(user_input, k=5)
        return "\n\n---\n\n".join([f"Fuente: {doc.metadata.get('source', 'Desconocido')}\n{doc.page_content}" for doc in docs]), None
    except Exception:
        return "", None

# ────────────────────────────────────────────────────────────────
# INTERFAZ DE CHAT Y AUDIO
# ────────────────────────────────────────────────────────────────
st.markdown("<div class='mic-container-top'>", unsafe_allow_html=True)
try:
    audio_data = mic_recorder(
        start_prompt="🎙️ Iniciar Grabación",
        stop_prompt="🛑 Detener Grabación",
        just_once=False,
        key="mic_main_btn"
    )
except Exception:
    pass
st.markdown("</div>", unsafe_allow_html=True)

# Procesar audio
if 'audio_data' in locals() and audio_data:
    try:
        audio_bytes = audio_data['bytes']
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{audio_data['format']}"
        
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language="es"
        )
        
        if transcription.text:
            st.toast(f"👂 Escuché: {transcription.text}")
            prompt = transcription.text
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            if "vamos a planear" in prompt.lower():
                st.session_state.planning_mode = True
            
            # Lógica de respuesta (se ejecutará en el bloque de abajo)
            # Forzamos la ejecución inmediata de la respuesta
            st.session_state.process_audio = True
            
    except Exception as e:
        st.error(f"Error de audio: {e}")

# Input de chat
if prompt := st.chat_input("Escribe tu mensaje..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if "vamos a planear" in prompt.lower():
        st.session_state.planning_mode = True
        st.toast("🚀 Modo Planeación Activado")

# Lógica de generación de respuesta
# Se ejecuta si hay nuevo mensaje de texto o si se activó la bandera de audio
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else SYSTEM_PROMPT_BASE
    context_text = ""
    
    if st.session_state.get("vectorstore"):
        context_text, _ = get_context_for_planning(
            st.session_state.messages[-1]["content"], 
            st.session_state.vectorstore, 
            st.session_state.loaded_files
        )
    
    full_prompt = current_prompt
    if context_text:
        full_prompt += f"\n\nContexto de documentos:\n{context_text}"
    
    formatted_messages = [{"role": "system", "content": full_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    try:
        with st.spinner("Pensando..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=formatted_messages
            )
            ai_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            # Limpiar bandera de audio si existía
            if "process_audio" in st.session_state:
                del st.session_state.process_audio
            
            st.rerun() # Recargar para mostrar la respuesta limpiamente
            
    except Exception as e:
        st.error(f"Error generando respuesta: {e}")

# Contenedor de chat
st.markdown("<div class='fixed-chat-wrapper'>", unsafe_allow_html=True)
chat_container = st.container(height=450, key="chat_container")
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        if message["role"] != "system":
            avatar = "🤖" if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
                if message["role"] == "assistant" and voice_enabled:
                    components.html(
                        get_audio_button_html(message["content"], f"audio_{i}"),
                        height=50,
                    )
st.markdown("</div>", unsafe_allow_html=True)

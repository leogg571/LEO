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
st.error(&quot;Faltan librerías. Instala con: pip install langchain-community langchain-text-splitters
faiss-cpu pypdf sentence-transformers&quot;)
st.stop()
#
═══════════════════════════════════════════════════════
════════
# CONFIGURACIÓN DE CARPETA (RUTA ABSOLUTA)
#
═══════════════════════════════════════════════════════
════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, &quot;documentos&quot;)
def load_knowledge_base():
if not os.path.exists(DOCS_FOLDER):
try:
os.makedirs(DOCS_FOLDER)
except OSError as e:
st.error(f&quot;Error al crear la carpeta &#39;documentos&#39;: {e}&quot;)
return None, []
pdf_files = glob.glob(os.path.join(DOCS_FOLDER, &quot;*.pdf&quot;))
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
doc.metadata[&quot;source&quot;] = filename
all_docs.extend(docs)
valid_files.append(filename)
except Exception as e:
error_files.append((os.path.basename(pdf_path), str(e)))
if error_files:
st.warning(f&quot;⚠️ No se pudieron leer {len(error_files)} archivos.&quot;)
if not all_docs:
return None, []
try:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)
embeddings = HuggingFaceEmbeddings(model_name=&quot;sentence-transformers/all-MiniLM-
L6-v2&quot;)
vectorstore = FAISS.from_documents(splits, embeddings)
return vectorstore, valid_files
except Exception as e:
st.error(f&quot;Error al procesar embeddings: {e}&quot;)
return None, []
# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
page_title=&quot;IA Prometeo&quot;,
page_icon=&quot;��&quot;,
layout=&quot;wide&quot;,
initial_sidebar_state=&quot;expanded&quot;,
menu_items={&#39;Get Help&#39;: None, &#39;Report a bug&#39;: None, &#39;About&#39;: &quot;IA Prometeo - Asistente
Inteligente&quot;}
)

#
═══════════════════════════════════════════════════════
════════
# CSS NEUTRO Y PROFESIONAL (LISTO PARA EDITAR)
#
═══════════════════════════════════════════════════════
════════
css_neutral = &quot;&quot;&quot;
&lt;style&gt;
@import
url(&#39;https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&amp;display=swap
&#39;);
/* OCULTAR ELEMENTOS STREAMLIT POR DEFECTO */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid=&quot;stDecoration&quot;] {display: none;}
[data-testid=&quot;stToolbar&quot;] {display: none;}
/* VARIABLES DE COLOR (FÁCIL DE EDITAR) */
:root {
--primary-color: #4F46E5; /* Índigo */
--secondary-color: #6366F1; /* Índigo claro */
--bg-color: #F3F4F6; /* Gris muy claro */
--sidebar-bg: #FFFFFF; /* Blanco */
--text-color: #1F2937; /* Gris oscuro */
--accent-color: #4F46E5;
}
/* FONDO GENERAL */
.stApp {
background-color: var(--bg-color);
color: var(--text-color);
font-family: &#39;Inter&#39;, sans-serif;
}
section[data-testid=&quot;stMain&quot;] {
position: relative;
z-index: 1 !important;
}
/* HEADER */
.main-header {
text-align: center;
padding: 2rem 1rem 1rem 1rem;
}

.main-title {
font-family: &#39;Inter&#39;, sans-serif;
font-weight: 700;
font-size: clamp(2rem, 6vw, 3rem);
color: var(--primary-color);
margin-bottom: 0.5rem;
}
.subtitle {
font-family: &#39;Inter&#39;, sans-serif;
color: #6B7280;
font-size: 1rem;
font-weight: 400;
}
/* CONTENEDOR PRINCIPAL */
.content-card {
background: #FFFFFF;
border-radius: 16px;
padding: 1.5rem;
box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
margin-bottom: 1rem;
border: 1px solid #E5E7EB;
}
/* BOTONES */
div[data-testid=&quot;stLinkButton&quot;] button {
background-color: var(--primary-color) !important;
color: white !important;
border: none !important;
border-radius: 12px !important;
padding: 0.6rem 1.2rem !important;
font-weight: 500 !important;
transition: all 0.2s ease !important;
}
div[data-testid=&quot;stLinkButton&quot;] button:hover {
background-color: var(--secondary-color) !important;
transform: translateY(-2px);
box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}
/* MIC BUTTON TOP */
.mic-container-top {
display: flex;
justify-content: center;
margin: 1rem auto;
}

.st-key-mic_main_btn button {
background-color: var(--primary-color) !important;
color: white !important;
border-radius: 50px !important;
padding: 0.8rem 1.5rem !important;
box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
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
.st-key-chat_container &gt; div &gt; div {
border: none !important;
background: transparent !important;
}
[data-testid=&quot;stChatMessage&quot;] {
background: #F9FAFB;
border: 1px solid #E5E7EB;
border-radius: 12px;
padding: 1rem;
margin-bottom: 0.8rem;
}
[data-testid=&quot;stChatMessageContent&quot;] {
color: var(--text-color) !important;
}
/* INPUT CHAT */
[data-testid=&quot;stChatInput&quot;] {
border: 1px solid #D1D5DB !important;
border-radius: 16px !important;
background-color: #FFFFFF !important;
box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
[data-testid=&quot;stChatInput&quot;] textarea,
[data-testid=&quot;stChatInputTextArea&quot;] {
background-color: transparent !important;
color: var(--text-color) !important;

}
[data-testid=&quot;stChatInput&quot;] textarea::placeholder,
[data-testid=&quot;stChatInputTextArea&quot;]::placeholder {
color: #9CA3AF !important;
}
[data-testid=&quot;stChatInput&quot;] button {
background-color: var(--primary-color) !important;
color: white !important;
}
/* SIDEBAR */
[data-testid=&quot;stSidebar&quot;] {
background: var(--sidebar-bg) !important;
border-right: 1px solid #E5E7EB;
}
[data-testid=&quot;stSidebar&quot;] h2, [data-testid=&quot;stSidebar&quot;] h3 {
color: var(--primary-color) !important;
font-family: &#39;Inter&#39;, sans-serif !important;
}
.stButton button {
background-color: var(--primary-color) !important;
color: white !important;
border-radius: 10px !important;
}
.stAlert {
background: #FFFFFF !important;
border-left: 4px solid var(--primary-color) !important;
border-radius: 8px !important;
color: var(--text-color) !important;
}
/* SCROLLBAR */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #F3F4F6; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
/* UTILIDADES */
.info-card {
background: #EFF6FF;
border: 1px solid #BFDBFE;
border-radius: 12px;
padding: 1rem;
margin-bottom: 0.5rem;

color: #1E40AF;
}
&lt;/style&gt;
&quot;&quot;&quot;
st.markdown(css_neutral, unsafe_allow_html=True)
#
═══════════════════════════════════════════════════════
════════
# HEADER
#
═══════════════════════════════════════════════════════
════════
header_html = &quot;&quot;&quot;
&lt;div class=&quot;main-header&quot;&gt;
&lt;h1 class=&quot;main-title&quot;&gt;IA PROMETEO ��&lt;/h1&gt;
&lt;p class=&quot;subtitle&quot;&gt;Tu asistente inteligente para documentos y planeación&lt;/p&gt;
&lt;/div&gt;
&quot;&quot;&quot;
st.markdown(header_html, unsafe_allow_html=True)
#
═══════════════════════════════════════════════════════
════════
# ZONA DE CONTENIDO (LIMPIA)
#
═══════════════════════════════════════════════════════
════════
# Aquí puedes agregar videos, imágenes o enlaces usando st.video, st.image, st.link_button
# Ejemplo:
# st.markdown(&quot;&lt;div class=&#39;content-card&#39;&gt;&quot;, unsafe_allow_html=True)
# st.video(&quot;URL_DE_TU_VIDEO&quot;)
# st.markdown(&quot;&lt;/div&gt;&quot;, unsafe_allow_html=True)
#
═══════════════════════════════════════════════════════
════════
# CONFIGURACIÓN DE API KEY
#
═══════════════════════════════════════════════════════
════════
api_key = None
if &quot;groq&quot; in st.secrets and &quot;api_key&quot; in st.secrets[&quot;groq&quot;]:
api_key = st.secrets[&quot;groq&quot;][&quot;api_key&quot;]
#
═══════════════════════════════════════════════════════
════════
# SIDEBAR

#
═══════════════════════════════════════════════════════
════════
with st.sidebar:
st.markdown(&quot;&lt;h2&gt;⚙️ Panel de Control&lt;/h2&gt;&quot;, unsafe_allow_html=True)
st.markdown(&quot;#### �� Configuración&quot;)
if not api_key:
api_key_input = st.text_input(&quot;API Key de Groq&quot;, type=&quot;password&quot;,
key=&quot;api_key_input_groq&quot;)
if api_key_input:
api_key = api_key_input
else:
st.warning(&quot;Necesitas API Key de Groq.&quot;)
st.info(&quot;Obtén una en: console.groq.com&quot;)
else:
st.success(&quot;API Key configurada ✅&quot;)
voice_enabled = st.checkbox(&quot;Activar respuestas de voz&quot;, value=True)
st.markdown(&quot;---&quot;)
# Lógica de carga de archivos
st.markdown(&quot;#### �� Base de Conocimiento&quot;)
st.caption(f&quot;Carpeta: `documentos/`&quot;)
uploaded_zip = st.file_uploader(&quot;Sube un ZIP con PDFs&quot;, type=&quot;zip&quot;, key=&quot;zip_uploader&quot;)
if uploaded_zip:
if &quot;processed_zip_name&quot; not in st.session_state or st.session_state.processed_zip_name !=
uploaded_zip.name:
try:
with zipfile.ZipFile(uploaded_zip, &#39;r&#39;) as z:
z.extractall(DOCS_FOLDER)
st.session_state.processed_zip_name = uploaded_zip.name
st.toast(f&quot;✅ Archivos extraídos. Recargando...&quot;)
st.cache_resource.clear()
st.rerun()
except Exception as e:
st.error(f&quot;Error al descomprimir: {e}&quot;)
st.markdown(&quot;---&quot;)
st.markdown(&quot;#### �� Estado&quot;)
if st.button(&quot;�� Recargar Base de Datos&quot;, use_container_width=True):
st.cache_resource.clear()
st.rerun()

if st.session_state.get(&quot;loaded_files&quot;):
st.success(f&quot;�� {len(st.session_state.loaded_files)} Documentos Activos&quot;)
with st.expander(&quot;Ver lista&quot;):
for f in st.session_state.loaded_files:
st.write(f&quot;�� {f}&quot;)
else:
st.info(f&quot;�� Repositorio Vacío. Añade PDFs.&quot;)
st.markdown(&quot;---&quot;)
# Sección neutral lista para editar
st.markdown(&quot;#### �� Tips&quot;)
st.markdown(&#39;&lt;div class=&quot;info-card&quot;&gt;Puedes preguntarme sobre los documentos cargados o
pedir ayuda para planear clases.&lt;/div&gt;&#39;, unsafe_allow_html=True)
st.markdown(&quot;&lt;br&gt;&lt;p style=&#39;text-align:center; font-size:0.8rem;
color:#9CA3AF;&#39;&gt;Desarrollado por IA Prometeo&lt;/p&gt;&quot;, unsafe_allow_html=True)
if not api_key:
st.stop()
try:
client = OpenAI(
base_url=&quot;https://api.groq.com/openai/v1&quot;,
api_key=api_key
)
except Exception as e:
st.error(f&quot;Error al conectar con Groq: {e}&quot;)
st.stop()
#
═══════════════════════════════════════════════════════
════════
# PERSONALIDAD Y MODO PLANEACIÓN
#
═══════════════════════════════════════════════════════
════════
SYSTEM_PROMPT_BASE = &quot;&quot;&quot;
Eres **IA Prometeo**, un asistente inteligente, conciso y profesional.
Tu objetivo es ayudar al usuario a analizar documentos y realizar tareas de planeación o
consulta.
Tono: Profesional, cercano y eficiente.
&quot;&quot;&quot;
loaded_files_list_str = &quot;No hay archivos cargados.&quot;
if st.session_state.get(&quot;loaded_files&quot;):
loaded_files_list_str = &quot;\n&quot;.join([f&quot;{i+1}. {fname}&quot; for i, fname in
enumerate(st.session_state.loaded_files)])

SYSTEM_PROMPT_PLANNING = f&quot;&quot;&quot;
Eres **IA Prometeo - Experto en Planeación**.
**ARCHIVOS DISPONIBLES:**
{loaded_files_list_str}
-----------------------------------------
**REGLAS:**
1. Usa solo información del contexto.
2. Busca patrones como &quot;Unidad&quot;, &quot;Módulo&quot;, &quot;Bloque&quot;.
3. Si no tienes información, indícalo.
**FLUJO:**
**PASO 1: ACTIVACIÓN**
Si el usuario dice &quot;vamos a planear&quot;:
1. Muestra la lista de archivos.
2. Pregunta: &quot;¿Cuál es el **número** del programa a utilizar?&quot;
**PASO 2: LECTURA**
Cuando el usuario elija un número:
1. Identifica el archivo.
2. Lista las unidades encontradas numeradas.
3. Pregunta: &quot;¿Qué **número** de unidad(es) vamos a planear?&quot;
**PASO 3: SESIONES**
&quot;¿Cuántas **sesiones** en total necesita?&quot;
**PASO 4: DIAS**
&quot;¿Qué **días de la semana** se imparten las clases?&quot;
**PASO 5: CRITERIOS**
&quot;¿Cuáles son los **criterios de evaluación**?&quot;
**PASO 6: FECHAS**
&quot;Indica **fecha de inicio** y **fecha de término**.&quot;
**PASO 7: BORRADOR**
Genera ejemplos.
**PASO 8: FINAL**
Genera planeación completa.
&quot;&quot;&quot;
#
═══════════════════════════════════════════════════════
════════
# INICIALIZACIÓN DE SESIÓN Y BASE DE DATOS

#
═══════════════════════════════════════════════════════
════════
if &quot;messages&quot; not in st.session_state:
st.session_state.messages = []
if &quot;planning_mode&quot; not in st.session_state:
st.session_state.planning_mode = False
if &quot;vectorstore&quot; not in st.session_state:
vectorstore, loaded_files = load_knowledge_base()
st.session_state.vectorstore = vectorstore
st.session_state.loaded_files = loaded_files
#
═══════════════════════════════════════════════════════
════════
# FUNCIONES AUXILIARES
#
═══════════════════════════════════════════════════════
════════
def get_audio_button_html(text, key):
text_clean = text.replace(&quot;&#39;&quot;, &quot;&quot;).replace(&#39;&quot;&#39;, &#39;&#39;).replace(&quot;\n&quot;, &quot; &quot;)
# Botón neutro para voz
return f&quot;&quot;&quot;
&lt;div style=&quot;margin-top: 10px; text-align: right;&quot;&gt;
&lt;button onclick=&quot;
var u = new SpeechSynthesisUtterance(&#39;{text_clean}&#39;);
u.lang = &#39;es-MX&#39;;
u.rate = 0.95;
window.speechSynthesis.cancel();
window.speechSynthesis.speak(u);
&quot; style=&quot;
background-color: #4F46E5;
color: white;
border: none;
padding: 8px 16px;
border-radius: 20px;
font-weight: 500;
cursor: pointer;
font-family: &#39;Inter&#39;, sans-serif;
font-size: 0.85rem;
box-shadow: 0 2px 5px rgba(0,0,0,0.1);
&quot;&gt;�� Escuchar&lt;/button&gt;
&lt;/div&gt;
&quot;&quot;&quot;

def get_context_for_planning(user_input, vectorstore, loaded_files):
selected_file_index = None
if loaded_files:
try:
potential_index = int(user_input.strip()) - 1
if 0 &lt;= potential_index &lt; len(loaded_files):
selected_file_index = potential_index
except ValueError:
pass
if selected_file_index is not None:
target_filename = loaded_files[selected_file_index]
structure_queries = [
&quot;Unidades de aprendizaje&quot;,
&quot;Contenido temático desglosado&quot;,
&quot;Bloques Módulos Temas&quot;,
&quot;Índice de contenidos estructura&quot;
]
all_docs = []
seen_content = set()
try:
for query in structure_queries:
docs = vectorstore.similarity_search(
query=query,
k=4,
filter={&quot;source&quot;: target_filename}
)
for doc in docs:
if doc.page_content not in seen_content:
all_docs.append(doc)
seen_content.add(doc.page_content)
final_docs = all_docs[:12]
if not final_docs:
return f&quot;No se encontró información estructural en {target_filename}.&quot;,
target_filename
context_text = &quot;\n\n---\n\n&quot;.join([f&quot;Fragmento:\n{doc.page_content}&quot; for doc in
final_docs])
return context_text, target_filename
except Exception as e:
return f&quot;Error al leer {target_filename}: {e}&quot;, target_filename
else:

try:
retriever = vectorstore.as_retriever(search_kwargs={&quot;k&quot;: 5})
docs = retriever.invoke(user_input)
return &quot;\n\n---\n\n&quot;.join([f&quot;Fuente: {doc.metadata.get(&#39;source&#39;,
&#39;Desconocido&#39;)}\n{doc.page_content}&quot; for doc in docs]), None
except Exception as e:
return &quot;&quot;, None
#
═══════════════════════════════════════════════════════
════════
# INTERFAZ DE CHAT
#
═══════════════════════════════════════════════════════
════════
audio_data = None
st.markdown(&quot;&lt;div class=&#39;mic-container-top&#39;&gt;&quot;, unsafe_allow_html=True)
try:
audio_data = mic_recorder(
start_prompt=&quot;�� Iniciar Grabación&quot;,
stop_prompt=&quot;�� Detener Grabación&quot;,
just_once=False,
key=&quot;mic_main_btn&quot;
)
except Exception as e:
pass
st.markdown(&quot;&lt;/div&gt;&quot;, unsafe_allow_html=True)
# Procesar audio
if audio_data:
try:
audio_bytes = audio_data[&#39;bytes&#39;]
audio_file = io.BytesIO(audio_bytes)
audio_file.name = f&quot;audio.{audio_data[&#39;format&#39;]}&quot;
transcription = client.audio.transcriptions.create(
file=audio_file,
model=&quot;whisper-large-v3&quot;,
language=&quot;es&quot;
)
if transcription.text:
st.toast(f&quot;�� Escuché: {transcription.text}&quot;)
prompt = transcription.text
st.session_state.messages.append({&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: prompt})
if &quot;vamos a planear&quot; in prompt.lower():

st.session_state.planning_mode = True
st.toast(&quot;�� Modo Planeación Activado&quot;)
current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else
SYSTEM_PROMPT_BASE
context_text = &quot;&quot;
if st.session_state.get(&quot;vectorstore&quot;):
context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore,
st.session_state.loaded_files)
full_prompt = current_prompt
if context_text:
full_prompt += f&quot;\n\nContexto de documentos:\n{context_text}&quot;
formatted_messages = [{&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: full_prompt}] + [{&quot;role&quot;: m[&quot;role&quot;],
&quot;content&quot;: m[&quot;content&quot;]} for m in st.session_state.messages]
response = client.chat.completions.create(
model=&quot;llama-3.1-8b-instant&quot;,
messages=formatted_messages
)
ai_response = response.choices[0].message.content
st.session_state.messages.append({&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: ai_response})
except Exception as e:
st.error(f&quot;Error de audio: {e}&quot;)
# Input de chat
if prompt := st.chat_input(&quot;Escribe tu mensaje...&quot;):
st.session_state.messages.append({&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: prompt})
if &quot;vamos a planear&quot; in prompt.lower():
st.session_state.planning_mode = True
st.toast(&quot;�� Modo Planeación Activado&quot;)
current_prompt = SYSTEM_PROMPT_PLANNING if st.session_state.planning_mode else
SYSTEM_PROMPT_BASE
context_text = &quot;&quot;
if st.session_state.get(&quot;vectorstore&quot;):
context_text, _ = get_context_for_planning(prompt, st.session_state.vectorstore,
st.session_state.loaded_files)
full_prompt = current_prompt
if context_text:
full_prompt += f&quot;\n\nContexto de documentos:\n{context_text}&quot;

formatted_messages = [{&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: full_prompt}] + [{&quot;role&quot;: m[&quot;role&quot;],
&quot;content&quot;: m[&quot;content&quot;]} for m in st.session_state.messages]
try:
response = client.chat.completions.create(
model=&quot;llama-3.1-8b-instant&quot;,
messages=formatted_messages
)
ai_response = response.choices[0].message.content
st.session_state.messages.append({&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: ai_response})
except Exception as e:
st.error(f&quot;Error: {e}&quot;)
# Contenedor de chat
st.markdown(&quot;&lt;div class=&#39;fixed-chat-wrapper&#39;&gt;&quot;, unsafe_allow_html=True)
chat_container = st.container(height=450, key=&quot;chat_container&quot;)
with chat_container:
for i, message in enumerate(st.session_state.messages):
if message[&quot;role&quot;] != &quot;system&quot;:
avatar = &quot;��&quot; if message[&quot;role&quot;] == &quot;assistant&quot; else &quot;��&quot;
with st.chat_message(message[&quot;role&quot;], avatar=avatar):
st.markdown(message[&quot;content&quot;])
if message[&quot;role&quot;] == &quot;assistant&quot; and voice_enabled:
components.html(
get_audio_button_html(message[&quot;content&quot;], f&quot;audio_{i}&quot;),
height=50,
)
st.markdown(&quot;&lt;/div&gt;&quot;, unsafe_allow_html=True)

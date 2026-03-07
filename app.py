## importing libraries
import os 
import tempfile
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

## Page setup
st.set_page_config(page_title="🤖 RAG Chatbot", layout="wide")
st.title("🔍 RAG Q&A with multiple PDFs + Chat History")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Controls")
    api_key_input = st.text_input("Groq Api Key", type="password")
    st.caption("Upload PDFs To Get Questions Answered from Pdf")
    st.divider()

    # 🔀 Mode Toggle
    st.subheader("🔀 Answer Mode")
    mode = st.radio(
        "How should the AI answer?",
        options=["🗂️ RAG Only", "🤖 LLM Only", "🔀 RAG + LLM"],
        index=0,
        help=(
            "🗂️ RAG Only — answers strictly from your PDFs\n\n"
            "🤖 LLM Only — answers from AI's own knowledge (ignores PDFs)\n\n"
            "🔀 RAG + LLM — tries PDFs first, AI fills gaps if not found"
        )
    )
    st.divider()

    # 🎭 Tone selector
    st.subheader("🎭 Response Tone")
    tone = st.selectbox(
        "Choose tone",
        ["Formal", "Simple", "Bullet Points", "ELI5 (Explain Like I'm 5)"],
        index=0
    )

    # 📝 Answer length
    st.subheader("📝 Answer Length")
    answer_length = st.radio(
        "Prefer",
        ["Short & Concise", "Detailed"],
        index=0
    )

    # 🌐 Language selector
    st.subheader("🌐 Response Language")
    language = st.selectbox(
        "Respond in",
        ["English", "Urdu", "Arabic", "French", "Spanish", "German", "Chinese"],
        index=0
    )

    st.divider()

    # 🗑️ Clear Chat + Session ID
    st.subheader("🗑️ Session")
    session_id = st.text_input("🆔 Session ID", value="default_session")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        if "chat_histories" in st.session_state:
            st.session_state.chat_histories[session_id] = ChatMessageHistory()
        st.success("Chat cleared!")
        st.rerun()

# ── API Key ────────────────────────────────────────────────────────────────────
api_key = api_key_input or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.warning('Enter your Groq API key to continue')
    st.stop()

# ── Models (cached - created only once) ───────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"
    )

embeddings = load_embeddings()
llm = load_llm(api_key)

# ── PDF Upload ─────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload pdf Files Here",
    type="pdf",
    accept_multiple_files=True
)

# If new files uploaded, save them to session state
if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files
# Fall back to previously saved files if uploader is empty
elif "uploaded_files" in st.session_state:
    uploaded_files = st.session_state["uploaded_files"]

if not uploaded_files:
    st.info("Upload one or more pdfs to continue")
    st.stop()

# Store files BEFORE calling build_retriever
st.session_state["uploaded_files"] = uploaded_files

# ── Build Retriever (cached - rebuilds only when files change) ─────────────────
@st.cache_resource(show_spinner="📄 Processing PDFs...")
def build_retriever(file_keys: tuple, _embeddings):
    all_docs = []
    tmp_paths = []

    for pdf in st.session_state["uploaded_files"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = pdf.name
        all_docs.extend(docs)

    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120
    )
    splits = text_splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        splits,
        _embeddings,
        persist_directory="chroma_index"
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    return retriever, len(all_docs), len(splits)

file_keys = tuple((f.name, f.size) for f in uploaded_files)
retriever, total_pages, total_chunks = build_retriever(file_keys, embeddings)

st.success(f"✅ Loaded {total_pages} pages | {total_chunks} chunks indexed")
st.sidebar.write(f"🔎 Indexed {total_chunks} chunks for retrieval")

# ── Helper: join docs ──────────────────────────────────────────────────────────
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# ── Helper: style instructions ─────────────────────────────────────────────────
def get_style_instructions(tone: str, length: str, language: str) -> str:
    tone_map = {
        "Formal":                    "Use formal, professional language. Be precise and structured.",
        "Simple":                    "Use simple, everyday language. Avoid jargon.",
        "Bullet Points":             "Always respond using clear bullet points and short sentences.",
        "ELI5 (Explain Like I'm 5)": "Explain as if talking to a 5-year-old. Use analogies and very simple words."
    }
    length_map = {
        "Short & Concise": "Keep your answer brief and to the point. 2-4 sentences max unless bullet points are needed.",
        "Detailed":         "Give a thorough, detailed answer covering all relevant aspects."
    }
    return (
        f"Tone: {tone_map[tone]}\n"
        f"Length: {length_map[length]}\n"
        f"Language: Always respond in {language}, regardless of the language of the question."
    )

# ── Prompts ────────────────────────────────────────────────────────────────────
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question into a standalone search query "
     "that can retrieve documents from a vector database. "
     "Use the chat history for context. "
     "Return ONLY the rewritten query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a STRICT RAG assistant. Answer using ONLY the provided context.\n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge.\n\n"
     "{style_instructions}\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm_only_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Answer the user's question using your own knowledge.\n"
     "Be honest if you are unsure about something.\n\n"
     "{style_instructions}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

hybrid_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful RAG assistant. "
     "First try to answer using the provided context from the user's documents.\n"
     "If the context does not fully answer the question, you may supplement "
     "with your own knowledge — but clearly label it like this:\n"
     "📄 From documents: <answer from context>\n"
     "🧠 From AI knowledge: <answer from own knowledge>\n\n"
     "{style_instructions}\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ── Session State ──────────────────────────────────────────────────────────────
# ✅ MUST be defined BEFORE get_history() is called
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

def get_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[s

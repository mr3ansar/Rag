## importing libraries
import os 
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

## loading api and setting page
st.set_page_config(page_title="🤖 RAG Chatbot", layout="wide")
st.title("🔍 RAG Q&A with multiple PDFs + Chat History")

# Making sidebar
with st.sidebar:
    st.header("⚙ Controls")
    api_key_input = st.text_input("Groq Api Key", type="password")
    st.caption("Upload PDFs To Get Questions Answered from Pdf")

api_key = api_key_input or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.warning('Enter your Groq API key to continue (set GROQ_API_KEY in secrets)')
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

if not uploaded_files:
    st.info("Upload one or more pdfs to continue")
    st.stop()

# ── Store files in session state BEFORE calling build_retriever ───────────────
st.session_state["uploaded_files"] = uploaded_files

# ── Load + Chunk + Embed (cached - rebuilds only when files change) ────────────

@st.cache_resource(show_spinner="📄 Processing PDFs...")
def build_retriever(file_keys: tuple, _embeddings):
    """
    file_keys  → tuple of (filename, size) used as the cache key.
    _embeddings → prefixed with _ so Streamlit skips hashing it.
    Rebuilds only when uploaded files actually change.
    """
    all_docs = []
    tmp_paths = []

    # Step 1: Load PDFs from session state
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

    # Step 2: Clean up temp files
    for p in tmp_paths:
        try:
            os.unlink(p)
        except Exception:
            pass

    # Step 3: Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120
    )
    splits = text_splitter.split_documents(all_docs)  # ✅ consistent variable name

    # Step 4: Build vectorstore
    vectorstore = Chroma.from_documents(
        splits,
        _embeddings,
        persist_directory="chroma_index"
    )

    # Step 5: Build retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    return retriever, len(all_docs), len(splits)  # ✅ inside the function

# ── Build cache key and call the function ─────────────────────────────────────

file_keys = tuple((f.name, f.size) for f in uploaded_files)
retriever, total_pages, total_chunks = build_retriever(file_keys, embeddings)

# ✅ Use return values from the function, not variables from inside it
st.success(f"✅ Loaded {total_pages} pages | {total_chunks} chunks indexed")
st.sidebar.write(f"🔎 Indexed {total_chunks} chunks for retrieval")

# ── Helper ─────────────────────────────────────────────────────────────────────

def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

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
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ── Session State for Chat History ────────────────────────────────────────────

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

def get_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

# ── Chat UI ────────────────────────────────────────────────────────────────────

session_id = st.text_input("🆔 Session ID", value="default_session")
history = get_history(session_id)

# Render previous messages
for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

user_q = st.chat_input("Ask a question....")

if user_q:
    history = get_history(session_id)

    # Step 1: Rewrite question
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )
    standalone_q = llm.invoke(rewrite_msgs).content.strip()

    # Step 2: Retrieve chunks
    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "Out of scope — not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        st.stop()

    # Step 3: Build context
    context_str = _join_docs(docs)

    # Step 4: Generate answer
    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_q,
        context=context_str
    )
    answer = llm.invoke(qa_msgs).content

    st.chat_message("user").write(user_q)
    st.chat_message("assistant").write(answer)

    history.add_user_message(user_q)
    history.add_ai_message(answer)

    # Debug panels
    with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📑 Retrieved Chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

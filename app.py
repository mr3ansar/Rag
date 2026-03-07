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

#Making sidebar

with st.sidebar:
    st.header("⚙ Controls")
    api_key_input = st.text_input("Groq Api Key", type="password")
    st.caption("Upload PDFs To Get Questions Answered from Pdf")

api_key = api_key_input or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.warning("""Enter your Groq API key to continue(set it up in the .env as "GROQ_API_KEY")""")
    st.stop()

# selecting embedding model and LLM

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs = {"normalize_embeddings": True}
)

llm = ChatGroq(
    groq_api_key = api_key,
    model_name = "llama-3.3-70b-versatile"
)

# Uploading multiple pdfs setup

uploaded_files = st.file_uploader(
    "📁 Upload pdf Files Here",
    type="pdf",
    accept_multiple_files = True
)

if not uploaded_files:
    st.info("Upload one or more pdfs to continue")
    st.stop()

all_docs = []
tmp_paths = []

for pdf in uploaded_files:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf.getvalue())
    tmp.close()
    tmp_paths.append(tmp.name)
    loader = PyPDFLoader(tmp.name)
    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf.name
    all_docs.extend(docs)

st.success(f"✅ Loaded {len(all_docs)} pages from {len(uploaded_files)} PDFs")

# Clean up temp files

for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

# Making Chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 120
)

@st.cache_data
def get_splits(docs):
    return text_splitter.split_documents(docs)

splits = get_splits(all_docs)
# Vectorstoring Splits and Embeddings and retrieving

INDEX_DIR = "chroma_index"

@st.cache_resources
def build_vectorstore(splits):
    return Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=INDEX_DIR
    )
vectorstore = build_vectorstore(splits)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 5, "fetch_k": 20}
)

st.sidebar.write(f" 🔎 Indexed {len(splits)} chunks for retrieval")

# Helper: format docs for stuffing

def _join_docs(docs, max_chars = 7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# Prompts

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
     "You are a STRICT RAG assistant. You must answer using only the provided context \n"
     "If the context does NOT contain the answer, reply exactly:\n"
     "'Out of scope - not found in provided documents.'\n"
     "Do NOT use outside knowledge. \n\n"
     "Context:\n{context}"),
     MessagesPlaceholder("chat_history"),
     ("human","{input}")
])

#session stae for chat history

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

def get_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]
#chat ui

session_id = st.text_input(" 🆔 Session ID ", value="default_session")

history = get_history(session_id)

# Show previous chat messages
for msg in history.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

user_q = st.chat_input("Ask a question....")
# ── Session state for chat history here

if user_q:
    history = get_history(session_id)
    # 1) Rewrite question with history
    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages,
        input=user_q
    )
    standalone_q = llm.invoke(rewrite_msgs).content.strip()
    #Retrieve chunks
    docs = retriever.invoke(standalone_q)
    
    if not docs:
        answer = "Out of scope — not found in provided documents."
        st.chat_message("user").write(user_q)
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        st.stop()

    # 3) Build context string
    context_str = _join_docs(docs)

    # Asking final question with stuffed context
    qa_msgs = qa_prompt.format_messages(
        chat_history= history.messages,
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










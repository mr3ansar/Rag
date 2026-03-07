# importing libraries
import os 
import tempfile
import streamlit as st
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Page setup
st.set_page_config(page_title="🤖 RAG Chatbot", layout="wide")
st.title("🔍 RAG Q&A with multiple PDFs + Chat History")

# Sidebar
with st.sidebar:
    st.header("⚙ Controls")
    api_key_input = st.text_input("Groq Api Key", type="password")
    st.caption("Upload PDFs To Get Questions Answered from Pdf")
    st.divider()

    # 🤖 Model selector
    st.subheader("🤖 Model")
    model_options = {
        "llama-3.3-70b-versatile       — Best overall":      "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768            — Long documents":    "mixtral-8x7b-32768",
        "gemma2-9b-it                  — Google model":      "gemma2-9b-it",
        "llama-3.1-8b-instant          — Fastest responses": "llama-3.1-8b-instant",
        "deepseek-r1-distill-llama-70b — Deep reasoning":    "deepseek-r1-distill-llama-70b",
    }
    selected_model_label = st.selectbox(
        "Choose model",
        options=list(model_options.keys()),
        index=0
    )
    selected_model = model_options[selected_model_label]
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
        ["English", "Roman Urdu", "Arabic", "French", "Spanish", "German", "Chinese"],
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
    st.divider()

    # 📊 Feedback Summary
    st.subheader("📊 Feedback Summary")
    if "feedback" in st.session_state and st.session_state.feedback:
        thumbs_up   = sum(1 for v in st.session_state.feedback.values() if v == "up")
        thumbs_down = sum(1 for v in st.session_state.feedback.values() if v == "down")
        total       = thumbs_up + thumbs_down
        st.metric("👍 Helpful",     thumbs_up)
        st.metric("👎 Not Helpful", thumbs_down)
        if total > 0:
            pct = int((thumbs_up / total) * 100)
            st.progress(pct / 100, text=f"{pct}% positive")
    else:
        st.caption("No feedback yet.")

# API Key
api_key = api_key_input or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.warning('Enter your Groq API key to continue')
    st.stop()

# Models (cached - created only once per unique combination) 
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_llm(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name
    )

embeddings     = load_embeddings()
llm            = load_llm(api_key, selected_model)

# Models that do NOT support system messages
NO_SYSTEM_MODELS = ["deepseek-r1-distill-llama-70b", "gemma2-9b-it"]

# PDF Upload
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

# Build Retriever (cached - rebuilds only when files change)
@st.cache_resource(show_spinner="📄 Processing PDFs...")
def build_retriever(file_keys: tuple, _embeddings):
    all_docs  = []
    tmp_paths = []

    for pdf in st.session_state["uploaded_files"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.getvalue())
        tmp.close()
        tmp_paths.append(tmp.name)

        loader = PyPDFLoader(tmp.name)
        docs   = loader.load()
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

file_keys                            = tuple((f.name, f.size) for f in uploaded_files)
retriever, total_pages, total_chunks = build_retriever(file_keys, embeddings)

st.success(f"✅ Loaded {total_pages} pages | {total_chunks} chunks indexed")
st.sidebar.write(f"🔎 Indexed {total_chunks} chunks for retrieval")

# Helper: join docs
def _join_docs(docs, max_chars=7000):
    chunks, total = [], 0
    for d in docs:
        piece = d.page_content + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunks)

# Helper: style instructions
def get_style_instructions(tone: str, length: str, language: str) -> str:
    tone_map = {
        "Formal":                    "Use formal, professional language. Be precise and structured.",
        "Simple":                    "Use simple, everyday language. Avoid jargon.",
        "Bullet Points":             "Always respond using clear bullet points and short sentences.",
        "ELI5 (Explain Like I'm 5)": "Explain as if talking to a 5-year-old. Use analogies and very simple words."
    }
    length_map = {
        "Short & Concise": "Keep your answer brief and to the point. 2-4 sentences max unless bullet points are needed.",
        "Detailed":        "Give a thorough, detailed answer covering all relevant aspects."
    }
    language_map = {
        "English":    "Always respond in English.",
        "Roman Urdu": (
            "CRITICAL INSTRUCTION: You MUST write in Roman Urdu ONLY. "
            "Roman Urdu = Urdu language but written with English/Latin letters. "
            "FORBIDDEN: ا ب پ ت ٹ ث ج — ANY Arabic/Urdu script characters are STRICTLY FORBIDDEN. "
            "ALLOWED: Only a b c d e f g h i j k l m n o p q r s t u v w x y z "
            "Example of CORRECT output: 'Yeh cheez bohat achi hai, main samajh gaya' "
            "Example of WRONG output: 'یہ چیز بہت اچھی ہے' "
            "If you use even ONE Urdu script character, you have failed. "
            "Write EVERY word using English letters only."
        ),
        "Arabic":  "Always respond in Arabic script.",
        "French":  "Always respond in French.",
        "Spanish": "Always respond in Spanish.",
        "German":  "Always respond in German.",
        "Chinese": "Always respond in Chinese (Simplified)."
    }
    return (
        f"Tone: {tone_map[tone]}\n"
        f"Length: {length_map[length]}\n"
        f"Language: {language_map[language]}"
    )

# Helper: safe rewrite (handles models that don't support system messages)
def rewrite_question(user_q: str, history, model_name: str) -> str:
    if model_name in NO_SYSTEM_MODELS:
        # Flatten full conversation into a single human message
        history_text = ""
        for msg in history.messages:
            role = "User" if msg.type == "human" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        flat_prompt = (
            f"Given this conversation history:\n{history_text}\n"
            f"Rewrite this question into a standalone search query: '{user_q}'\n"
            f"Return ONLY the rewritten query, nothing else."
        )
        return llm.invoke([HumanMessage(content=flat_prompt)]).content.strip()
    else:
        rewrite_msgs = contextualize_q_prompt.format_messages(
            chat_history=history.messages,
            input=user_q
        )
        return llm.invoke(rewrite_msgs).content.strip()

# ── Helper: safe LLM invoke (handles models that don't support system messages) ─
def safe_llm_invoke(prompt_template, model_name: str, fallback_text: str, **kwargs) -> str:
    if model_name in NO_SYSTEM_MODELS:
        return llm.invoke([HumanMessage(content=fallback_text)]).content
    else:
        msgs = prompt_template.format_messages(**kwargs)
        return llm.invoke(msgs).content

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

# Session State
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "feedback" not in st.session_state:
    st.session_state.feedback = {}

def get_history(session_id: str):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

# Chat UI
history = get_history(session_id)

# Chat export
def build_export_text(history, session_id: str) -> str:
    lines = [
        f"Chat Export — Session: {session_id}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50, ""
    ]
    for msg in history.messages:
        role = "You" if msg.type == "human" else "Assistant"
        lines.append(f"[{role}]\n{msg.content}\n")
    return "\n".join(lines)

if history.messages:
    export_text = build_export_text(history, session_id)
    st.download_button(
        label="📥 Export Chat (.txt)",
        data=export_text,
        file_name=f"chat_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Render previous messages with feedback buttons
ai_msg_index = 0

for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

    if msg.type == "ai":
        col1, col2, col3 = st.columns([1, 1, 10])
        fb_key     = f"{session_id}_{ai_msg_index}"
        current_fb = st.session_state.feedback.get(fb_key)

        with col1:
            if st.button("👍", key=f"up_{fb_key}",
                         help="Good answer",
                         type="primary" if current_fb == "up" else "secondary"):
                st.session_state.feedback[fb_key] = "up"
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{fb_key}",
                         help="Bad answer",
                         type="primary" if current_fb == "down" else "secondary"):
                st.session_state.feedback[fb_key] = "down"
                st.rerun()
        ai_msg_index += 1

# Chat input
user_q = st.chat_input("Ask a question....")

# Chat Pipeline
if user_q:
    history = get_history(session_id)
    style_instructions = get_style_instructions(tone, answer_length, language)
    st.chat_message("user").write(user_q)

    # MODE: LLM Only
    if mode == "🤖 LLM Only":
        # Build fallback flat prompt for no-system models
        fallback = (
            f"Answer this question using your own knowledge.\n"
            f"{style_instructions}\n\n"
            f"Question: {user_q}"
        )
        answer = safe_llm_invoke(
            llm_only_prompt,
            selected_model,
            fallback,
            chat_history=history.messages,
            input=user_q,
            style_instructions=style_instructions
        )
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)

    # MODE: RAG Only
    elif mode == "🗂️ RAG Only":
        standalone_q = rewrite_question(user_q, history, selected_model)
        docs         = retriever.invoke(standalone_q)

        if not docs:
            answer = "Out of scope — not found in provided documents."
            st.chat_message("assistant").write(answer)
            history.add_user_message(user_q)
            history.add_ai_message(answer)
            st.stop()

        context_str = _join_docs(docs)
        fallback = (
            f"Answer using ONLY this context. "
            f"If not found say 'Out of scope'.\n"
            f"{style_instructions}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {user_q}"
        )
        answer = safe_llm_invoke(
            qa_prompt,
            selected_model,
            fallback,
            chat_history=history.messages,
            input=user_q,
            context=context_str,
            style_instructions=style_instructions
        )
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)

        with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
            st.write("**Rewritten (standalone) query:**")
            st.code(standalone_q or "(empty)", language="text")
            st.write(f"**Retrieved {len(docs)} chunk(s).**")
            st.write(f"**Model:** {selected_model}")
            st.write(f"**Style:** {tone} | {answer_length} | {language}")

        with st.expander("📑 Retrieved Chunks"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
                st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

    # MODE: RAG + LLM Hybrid
    elif mode == "🔀 RAG + LLM":
        standalone_q = rewrite_question(user_q, history, selected_model)
        docs         = retriever.invoke(standalone_q)
        context_str  = _join_docs(docs) if docs else "No relevant documents found."

        fallback = (
            f"Answer using the context below. If context is weak, use your own knowledge "
            f"and label it '🧠 From AI knowledge:'.\n"
            f"{style_instructions}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {user_q}"
        )
        answer = safe_llm_invoke(
            hybrid_prompt,
            selected_model,
            fallback,
            chat_history=history.messages,
            input=user_q,
            context=context_str,
            style_instructions=style_instructions
        )
        st.chat_message("assistant").write(answer)
        history.add_user_message(user_q)
        history.add_ai_message(answer)
        #Debug Panel
        with st.expander("🧪 Debug: Rewritten Query & Retrieval"):
            st.write("**Rewritten (standalone) query:**")
            st.code(standalone_q or "(empty)", language="text")
            st.write(f"**Retrieved {len(docs)} chunk(s).**")
            st.write(f"**Model:** {selected_model}")

        with st.expander("📑 Retrieved Chunks"):
            if docs:
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p{doc.metadata.get('page','?')})**")
                    st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            else:
                st.info("No chunks retrieved — AI answered from its own knowledge.")

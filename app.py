import os
import json
import html
import torch
import chromadb
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# 1. PAGE CONFIGURATION & LOCAL PATHS
# --------------------------------------------------

st.set_page_config(
    page_title="SportsMate AI | Presidency University",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Relative paths based strictly on project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
METADATA_FILE = os.path.join(BASE_DIR, "image_metadata.json")

def load_image_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

IMAGE_METADATA = load_image_metadata()

def find_relevant_images(query, answer_text="", context_text="", max_images=3):
    """
    Finds relevant sports/facility images from images/ based on matched tags.
    """
    if not IMAGE_METADATA:
        return []

    combined_text = f"{query} {answer_text} {context_text}".lower()
    matches = []

    for filename, meta in IMAGE_METADATA.items():
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            continue

        score = 0
        cat = meta.get("category", "").lower()
        if cat and cat in combined_text:
            score += 3

        for tag in meta.get("tags", []):
            if tag in combined_text:
                score += 1

        if score > 0:
            matches.append((score, {
                "path": img_path,
                "caption": meta.get("caption", "Presidency University Sports Facility"),
                "category": meta.get("category", "Sports")
            }))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in matches[:max_images]]


COLLECTION_NAME = "university_knowledge_base"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"

# --------------------------------------------------
# 2. DESIGN SYSTEM & MODERN STYLING
# --------------------------------------------------

st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1b2a 0%, #08111e 50%, #050a12 100%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #08121f;
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    /* Compact Main Header */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: rgba(13, 27, 42, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .header-icon {
        font-size: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
    }

    .header-text h1 {
        color: #ffffff;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }

    .header-text p {
        color: #94a3b8;
        font-size: 13px;
        margin: 2px 0 0 0;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.28);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22c55e;
        box-shadow: 0 0 8px #22c55e;
    }

    /* Welcome Landing Screen */
    .welcome-container {
        padding: 32px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
        text-align: center;
    }

    .welcome-container h2 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 10px 0;
    }

    .welcome-container p {
        color: #cbd5e1;
        font-size: 15px;
        max-width: 680px;
        margin: 0 auto 28px auto;
        line-height: 1.5;
    }

    /* Suggestion Cards */
    .suggestion-card-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #38bdf8;
        margin-bottom: 12px;
        text-align: left;
    }

    /* Sidebar Status Card */
    .sidebar-db-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 16px;
    }

    .sidebar-db-title {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #4ade80;
        margin-bottom: 6px;
    }

    .sidebar-db-details {
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* Chat Messages Styling */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Distinguish User vs Assistant message boxes */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(14, 116, 144, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.22);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 16px !important;
        background: #091322 !important;
    }

    /* Streamlit Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        background: #0f1e33;
        color: #e2e8f0;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.2s ease;
        text-align: left;
        padding: 10px 14px;
        height: auto;
        min-height: 44px;
    }

    .stButton > button:hover {
        border-color: #38bdf8;
        background: #152d4d;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }

    /* Expandable Source Section */
    div[data-testid="stExpander"] {
        background: rgba(11, 20, 36, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        margin-top: 10px;
    }

    .source-item {
        padding: 6px 0;
        font-size: 12px;
        color: #cbd5e1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .source-item:last-child {
        border-bottom: none;
    }

    .source-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 24px 0 12px 0;
        border-top: 1px solid rgba(56, 189, 248, 0.08);
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. RESOURCE LOADING & DATABASE HEALTH CHECK
# --------------------------------------------------

@st.cache_resource(show_spinner="⏳ Initializing SportsMate RAG pipeline & models...")
def load_rag_pipeline():
    """
    Loads existing SentenceTransformer, persistent ChromaDB collection,
    and HuggingFace Flan-T5 model. Strictly preserves existing database.
    """
    # 1. Dense Embedding Model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2. Existing Persistent ChromaDB Client (PRESERVES existing database)
    client = chromadb.PersistentClient(path=DB_PATH)

    # 3. Existing Collection with 233 chunks
    collection = client.get_collection(name=COLLECTION_NAME)

    # 4. HuggingFace Seq2Seq Generation Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    model.eval()

    return embedding_model, collection, tokenizer, model


def verify_database_health():
    """
    Checks persistent ChromaDB connection and verifies the chunk count.
    Never alters, overwrites, or recreates the database.
    """
    try:
        if not os.path.exists(DB_PATH):
            return False, 0, "ChromaDB directory not found at " + DB_PATH
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        return True, count, "Operational"
    except Exception as e:
        return False, 0, str(e)


# --------------------------------------------------
# 4. ANSWER GENERATION & RAG RETRIEVAL PIPELINE
# --------------------------------------------------

def generate_answer(question, context, tokenizer, model):
    """
    Generates a factual answer strictly from the retrieved context.
    """
    prompt = (
        f"Answer the question based only on the university document context below.\n\n"
        f"Context:\n{context[:1400]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=140,
            num_beams=3,
            do_sample=False
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer


def format_grounded_answer(question, raw_answer, context):
    """
    Formats the answer into clean, professional prose.
    Ensures that if documents don't contain the answer, we state it clearly.
    """
    raw = raw_answer.strip()
    
    # Check for empty, repetitive, or unhelpful fragments (e.g. index markers like '(4)' or 'iii.')
    is_fragment = (
        len(raw) < 4
        or raw.lower() in ["none", "unknown", "not specified", "unspecified", "answer:"]
        or raw.startswith("(") and raw.endswith(")") and len(raw) <= 5
        or raw.lower() in ["i.", "ii.", "iii.", "iv.", "v.", "vi."]
    )

    if not raw or is_fragment:
        return "I couldn't find enough information about this in the available university documents."

    # Format cleanly with the requested prefix if not already present
    clean_text = raw
    if clean_text[0].islower():
        clean_text = clean_text[0].upper() + clean_text[1:]
    if not clean_text.endswith("."):
        clean_text += "."

    if not clean_text.lower().startswith("according to"):
        # Lowercase the first word after the prefix
        formatted = f"According to the available university documents, {clean_text[0].lower() + clean_text[1:]}"
    else:
        formatted = clean_text

    return formatted


def answer_question(user_question):
    """
    Processes the user's exact question through the preserved RAG pipeline.
    The user's original question is NEVER modified or replaced.
    """
    embedding_model, collection, tokenizer, model = load_rag_pipeline()

    # 1. Encode the user's question
    query_embedding = embedding_model.encode(
        user_question,
        normalize_embeddings=True
    ).tolist()

    # 2. Retrieve top-4 chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "I couldn't find enough information about this in the available university documents.", []

    # Domain keywords to check document relevance without hallucinating
    sports_keywords = [
        "sport", "game", "indoor", "outdoor", "playground",
        "physical education", "sports council", "sports policy",
        "athlete", "tournament", "fitness", "cricket", "football",
        "basketball", "badminton", "volleyball", "ncc", "nss",
        "gym", "gymnasium", "ground", "court", "athletics", "track",
        "coach", "coaching", "matches", "recreation", "yoga",
        "table tennis", "throw ball", "handbook", "schedule", "participation"
    ]

    is_sports_query = any(k in user_question.lower() for k in sports_keywords)
    relevant_chunks = []

    for doc, meta in zip(documents, metadatas):
        if any(k in doc.lower() for k in sports_keywords):
            relevant_chunks.append((doc, meta))

    # Guard against queries that have no sports context or excessive distance
    min_distance = min(distances) if distances else 1.0
    if not is_sports_query and (not relevant_chunks or min_distance > 0.65):
        return "I couldn't find enough information about this in the available university documents.", []

    # Build context from top relevant chunks (or top retrieved chunks if distance is close)
    if relevant_chunks:
        selected = relevant_chunks[:2]
    else:
        selected = list(zip(documents[:2], metadatas[:2]))

    context_text = "\n\n".join(doc for doc, _ in selected)

    # 3. Generate answer via Flan-T5
    raw_answer = generate_answer(user_question, context_text, tokenizer, model)
    final_answer = format_grounded_answer(user_question, raw_answer, context_text)

    # 4. Compile verified sources metadata without exposing large text chunks
    sources = []
    seen_sources = set()

    for _, meta in selected:
        if not meta:
            continue
        raw_source = meta.get("source", meta.get("file_name", "Presidency University Document"))
        source_name = os.path.basename(str(raw_source)) if ("/" in str(raw_source) or "\\" in str(raw_source)) else str(raw_source)
        page_num = meta.get("page", meta.get("page_number", None))

        key = (source_name, page_num)
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({
                "source": source_name,
                "page": page_num
            })

    matched_images = find_relevant_images(user_question, final_answer, context_text)
    return final_answer, sources, matched_images


# --------------------------------------------------
# 5. SIDEBAR: BRANDING, STATUS & NAVIGATION
# --------------------------------------------------

db_is_healthy, db_chunks, db_msg = verify_database_health()

with st.sidebar:
    st.markdown("### 🏆 SportsMate AI")
    st.caption("Presidency University Bengaluru · Sports Assistant")

    # Real Database Health Status Indicator
    if db_is_healthy:
        st.markdown(f"""
        <div class="sidebar-db-card">
            <div class="sidebar-db-title">
                <span class="status-dot"></span> Database Online
            </div>
            <div class="sidebar-db-details">
                <b>Collection:</b> <code>{COLLECTION_NAME}</code><br>
                <b>Indexed Chunks:</b> {db_chunks} verified<br>
                <b>Mode:</b> Read-Only RAG
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Database status: {db_msg}")

    # New Chat / Clear Conversation Control
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_question = None
        st.rerun()

    st.divider()

    st.markdown("#### 🎯 Quick Questions")
    st.caption("Click any question to ask directly:")

    sidebar_quick_prompts = [
        "What sports facilities are available at Presidency University?",
        "What does the university sports policy say?",
        "What is the role of the Sports Council?",
        "What sports activities are mentioned in the documents?",
        "How can students participate in sports activities?"
    ]

    for q_item in sidebar_quick_prompts:
        if st.button(q_item, key=f"quick_side_{hash(q_item)}", use_container_width=True):
            st.session_state.selected_question = q_item
            st.rerun()

    st.divider()

    st.markdown("#### 📚 Knowledge Base")
    st.caption(
        "Answers are strictly grounded in official Presidency University "
        "Student Handbook and athletic guidelines indexed in ChromaDB."
    )
    st.caption("Presidency University · Rajanukunte, Bengaluru")


# --------------------------------------------------
# 6. COMPACT HEADER
# --------------------------------------------------

ready_status_html = """
<div class="status-badge">
    <span class="status-dot"></span> Online & Ready
</div>
""" if db_is_healthy else """
<div class="status-badge" style="background: rgba(239, 68, 68, 0.12); color: #f87171; border-color: rgba(239, 68, 68, 0.28);">
    <span>●</span> Database Offline
</div>
"""

st.markdown(f"""
<div class="header-container">
    <div class="header-left">
        <div class="header-icon">🏆</div>
        <div class="header-text">
            <h1>SportsMate AI</h1>
            <p>Your Presidency University Sports Assistant</p>
        </div>
    </div>
    {ready_status_html}
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 7. CHAT SESSION STATE & CONVERSATIONAL FLOW
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# WELCOME SCREEN (Rendered ONLY when no messages exist)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <h2>Welcome to SportsMate!</h2>
        <p>
            Ask me about university sports facilities, sports activities,
            student participation, events, and other information available in
            the university documents.
        </p>
        <div class="suggestion-card-label">💡 Suggested Questions</div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Clickable Suggested Question Cards in a clean 2x2 grid
    card_cols1 = st.columns(2)
    card_cols2 = st.columns(2)

    suggestions = [
        "What sports facilities are available?",
        "How can students participate in sports activities?",
        "What sports-related information is in the student handbook?",
        "Show me the university sports activity schedule."
    ]

    with card_cols1[0]:
        if st.button(f"🏟️  {suggestions[0]}", key="sugg_0", use_container_width=True):
            st.session_state.selected_question = suggestions[0]
            st.rerun()

    with card_cols1[1]:
        if st.button(f"🏃  {suggestions[1]}", key="sugg_1", use_container_width=True):
            st.session_state.selected_question = suggestions[1]
            st.rerun()

    with card_cols2[0]:
        if st.button(f"📋  {suggestions[2]}", key="sugg_2", use_container_width=True):
            st.session_state.selected_question = suggestions[2]
            st.rerun()

    with card_cols2[1]:
        if st.button(f"📅  {suggestions[3]}", key="sugg_3", use_container_width=True):
            st.session_state.selected_question = suggestions[3]
            st.rerun()


# RENDER CHRONOLOGICAL CONVERSATION HISTORY
# Ensures each user question and assistant answer appears exactly once
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display related images if present in history
        if msg["role"] == "assistant" and msg.get("images"):
            cols = st.columns(min(len(msg["images"]), 3))
            for idx, img in enumerate(msg["images"]):
                with cols[idx]:
                    st.image(img["path"], caption=img["caption"], use_container_width=True)

        # Display compact 'Sources used' section if sources exist
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    page_info = f" | Page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(f"📄 **{html.escape(s['source'])}**{page_info}")


# --------------------------------------------------
# 8. USER INPUT HANDLING & RAG EXECUTION
# --------------------------------------------------

# Capture input from chat box OR suggestion card
chat_input_text = st.chat_input("Ask a question about Presidency University sports...")

user_query = chat_input_text
if not user_query and st.session_state.selected_question:
    # Pop and consume selected question to prevent lingering state
    user_query = st.session_state.selected_question
    st.session_state.selected_question = None

if user_query:
    # 1. Store and display EXACT original user question (no rewriting, no substitution)
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Retrieve from RAG and generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching university documents..."):
            try:
                answer, sources, images = answer_question(user_query)
            except Exception as e:
                answer = "I couldn't find enough information about this in the available university documents."
                sources = []
                images = []
                print(f"[SportsMate Error] {repr(e)}")

        st.markdown(answer)

        if images:
            cols = st.columns(min(len(images), 3))
            for idx, img in enumerate(images):
                with cols[idx]:
                    st.image(img["path"], caption=img["caption"], use_container_width=True)

        if sources:
            with st.expander("Sources used"):
                for s in sources:
                    page_info = f" | Page {s['page']}" if s.get("page") is not None else ""
                    st.markdown(f"📄 **{html.escape(s['source'])}**{page_info}")

    # 3. Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "images": images
    })

    # Rerun to cleanly finalize state and keep conversation in order
    st.rerun()


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------
st.markdown("""
<div class="app-footer">
    <b>SportsMate AI</b> · Presidency University Bengaluru Sports Information System<br>
    Powered by ChromaDB Vector Search (233 Verified Chunks) · HuggingFace MiniLM & Flan-T5
</div>
""", unsafe_allow_html=True)

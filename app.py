"""
Academic City RAG Chatbot — Streamlit UI
Author: Maukewonge Yaw Nyarko-Tetteh | Index: 10022200107
CS4241 - Introduction to Artificial Intelligence, 2026

Features:
- Chat interface with query input
- Retrieved chunks display with similarity scores
- Final LLM response display
- Memory-based context (session memory)
- Pipeline debug view (logs each stage)
- Prompt template selector
- Provider / API key configuration
"""

import os
import sys
import streamlit as st

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ACity RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub-title  { font-size: 1rem; color: #555; margin-top: -10px; }
    .chunk-card {
        background: #f8f9fa;
        border-left: 4px solid #4361ee;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        font-size: 0.85rem;
    }
    .score-badge {
        display: inline-block;
        background: #4361ee;
        color: white;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .source-badge-budget  { background: #e76f51; }
    .source-badge-election { background: #2a9d8f; }
    .memory-badge {
        display: inline-block;
        background: #7b2d8b;
        color: white;
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.72rem;
    }
    .failure-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — CONFIGURATION
# ─────────────────────────────────────────────

# Load API key from Streamlit secrets (stored in .streamlit/secrets.toml)
secrets_provider = st.secrets.get("LLM_PROVIDER", "groq")
secrets_api_key = st.secrets.get("LLM_API_KEY", "")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Academic_City_University_College_Logo.png/220px-Academic_City_University_College_Logo.png",
             width=120)
    st.markdown("### ⚙️ Configuration")

    provider = st.selectbox(
        "LLM Provider",
        ["anthropic", "openai", "google", "groq"],
        index=["anthropic", "openai", "google", "groq"].index(secrets_provider) if secrets_provider in ["anthropic", "openai", "google", "groq"] else 3
    )
    
    # Show API key status from secrets
    if secrets_api_key:
        st.success(f"✅ API Key loaded from secrets ({secrets_provider.upper()})")
        api_key_input = st.text_input(
            "API Key (override secrets)",
            type="password",
            placeholder="Leave blank to use secrets",
            help="Leave blank to use the API key stored in .streamlit/secrets.toml"
        )
        # Use override if provided, otherwise use secrets
        api_key = api_key_input if api_key_input else secrets_api_key
    else:
        st.warning("⚠️ No API key found in secrets")
        api_key = st.text_input(
            "API Key (required)",
            type="password",
            placeholder="Paste your API key here",
            help="Store in .streamlit/secrets.toml to avoid re-entering"
        )

    template_version = st.radio(
        "Prompt Template",
        ["v1 — Basic", "v2 — Structured (recommended)", "v3 — Chain-of-Thought"],
        index=1,
        help="v1: simple, v2: structured with hallucination guard, v3: CoT reasoning"
    )
    tv = template_version.split(" ")[0]   # extract "v1", "v2", "v3"

    top_k = st.slider("Top-K Retrieval", min_value=2, max_value=10, value=5)

    show_debug = st.toggle("Show Pipeline Debug", value=False)
    pure_llm_compare = st.toggle("Also run Pure-LLM (no retrieval)", value=False)

    st.markdown("---")
    st.markdown("### 🧠 Session Memory")
    if "pipeline" in st.session_state:
        mem_count = len(st.session_state.pipeline.memory)
        st.info(f"{mem_count} memory entries stored")
        if st.button("🗑️ Clear Memory"):
            st.session_state.pipeline.memory.clear()
            st.success("Memory cleared!")

    st.markdown("---")
    st.markdown("**Data Sources:**")
    st.markdown("📄 2025 Ghana Budget Statement")
    st.markdown("🗳️ Ghana Election Results")
    st.caption("CS4241 · Academic City University · 2026")


# ─────────────────────────────────────────────
# PIPELINE INITIALISATION (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline(provider: str):
    """Cache one pipeline per provider; api_key is injected at query time."""
    from pipeline import build_pipeline
    return build_pipeline(BASE_DIR, api_key="", provider=provider)


def load_pipeline():
    with st.spinner("🔧 Building RAG index (first run may take ~2 minutes)..."):
        st.session_state.pipeline = get_pipeline(provider)
        st.session_state["_pipeline_provider"] = provider


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.markdown('<p class="main-title">🎓 ACity RAG Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Chat with Ghana\'s 2025 Budget Statement & Election Results</p>',
    unsafe_allow_html=True
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if ("pipeline" not in st.session_state
        or st.session_state.get("_pipeline_provider") != provider):
    load_pipeline()

# Welcome message
if not st.session_state.messages:
    st.markdown("### Welcome! 👋")
    st.markdown("I'm here to help you explore Ghana's 2025 Budget and Election Results. Feel free to ask questions or try the sample queries below.")

# Render existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

def process_query(user_input):
    """Process a user query and generate response."""
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run pipeline
    pipeline = st.session_state.pipeline
    pipeline.llm.api_key = api_key
    pipeline.llm.provider = provider
    pipeline.llm.model = pipeline.llm._default_model()

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            response, log = pipeline.query(
                user_input,
                top_k=top_k,
                template_version=tv,
                pure_llm_mode=False
            )

        # ── Response ─────────────────────────────────────────────
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # ── Retrieved chunks ──────────────────────────────────────
        with st.expander(f"📄 Retrieved Chunks ({len(log.retrieved_chunks)})", expanded=False):
            for i, chunk in enumerate(log.retrieved_chunks, 1):
                src_class = ("source-badge-budget" if chunk["source"] == "budget"
                             else "source-badge-election")
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span class="score-badge">#{i} score={chunk["score"]}</span>'
                    f'<span class="score-badge {src_class}">'
                    f'{chunk["source"].upper()} p.{chunk["page"]}</span><br><br>'
                    f'{chunk["text_snippet"]}...'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Memory context ────────────────────────────────────────
        if log.memory_context:
            with st.expander("🧠 Memory Context Used", expanded=False):
                st.markdown(
                    f'<span class="memory-badge">SESSION MEMORY</span><br><br>'
                    f'<pre style="font-size:0.8rem">{log.memory_context[:600]}</pre>',
                    unsafe_allow_html=True
                )

        # ── Failure warning ───────────────────────────────────────
        if log.failure_detected:
            st.markdown(
                f'<div class="failure-warning">⚠️ <strong>Retrieval warning:</strong> '
                f'{log.failure_reason} — fallback retrieval was used.</div>',
                unsafe_allow_html=True
            )

        # ── Debug view ────────────────────────────────────────────
        if show_debug:
            with st.expander("🔬 Pipeline Debug", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", log.prompt_tokens)
                col2.metric("Latency (ms)", f"{log.latency_ms:.0f}")
                col3.metric("Template", log.template_version)

                st.markdown("**Expanded Query:**")
                st.code(log.expanded_query or user_input)

                st.markdown("**Final Prompt (truncated):**")
                st.code(log.final_prompt[:800] + "...", language="text")

                st.markdown("**Selected Chunks:**")
                for sc in log.selected_chunks:
                    st.write(f"• `{sc['chunk_id']}` | score={sc['score']} | "
                             f"{sc['text'][:100]}...")

        # ── Pure LLM comparison ───────────────────────────────────
        if pure_llm_compare:
            with st.expander("🤖 Pure-LLM Response (no retrieval)", expanded=False):
                with st.spinner("Running pure LLM..."):
                    llm_response, _ = pipeline.query(
                        user_input, pure_llm_mode=True
                    )
                st.markdown("**RAG Response:**")
                st.info(response[:400])
                st.markdown("**Pure LLM Response:**")
                st.warning(llm_response[:400] if llm_response else
                           "[No API key — cannot call LLM]")
                st.caption("RAG uses retrieved documents; Pure LLM uses only "
                           "the model's training data (may hallucinate).")

user_input = st.chat_input("Ask about Ghana's 2025 Budget or Election Results...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run pipeline
    pipeline = st.session_state.pipeline
    pipeline.llm.api_key = api_key
    pipeline.llm.provider = provider
    pipeline.llm.model = pipeline.llm._default_model()

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            response, log = pipeline.query(
                user_input,
                top_k=top_k,
                template_version=tv,
                pure_llm_mode=False
            )

        # ── Response ─────────────────────────────────────────────
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # ── Retrieved chunks ──────────────────────────────────────
        with st.expander(f"📄 Retrieved Chunks ({len(log.retrieved_chunks)})", expanded=False):
            for i, chunk in enumerate(log.retrieved_chunks, 1):
                src_class = ("source-badge-budget" if chunk["source"] == "budget"
                             else "source-badge-election")
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<span class="score-badge">#{i} score={chunk["score"]}</span>'
                    f'<span class="score-badge {src_class}">'
                    f'{chunk["source"].upper()} p.{chunk["page"]}</span><br><br>'
                    f'{chunk["text_snippet"]}...'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Memory context ────────────────────────────────────────
        if log.memory_context:
            with st.expander("🧠 Memory Context Used", expanded=False):
                st.markdown(
                    f'<span class="memory-badge">SESSION MEMORY</span><br><br>'
                    f'<pre style="font-size:0.8rem">{log.memory_context[:600]}</pre>',
                    unsafe_allow_html=True
                )

        # ── Failure warning ───────────────────────────────────────
        if log.failure_detected:
            st.markdown(
                f'<div class="failure-warning">⚠️ <strong>Retrieval warning:</strong> '
                f'{log.failure_reason} — fallback retrieval was used.</div>',
                unsafe_allow_html=True
            )

        # ── Debug view ────────────────────────────────────────────
        if show_debug:
            with st.expander("🔬 Pipeline Debug", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", log.prompt_tokens)
                col2.metric("Latency (ms)", f"{log.latency_ms:.0f}")
                col3.metric("Template", log.template_version)

                st.markdown("**Expanded Query:**")
                st.code(log.expanded_query or user_input)

                st.markdown("**Final Prompt (truncated):**")
                st.code(log.final_prompt[:800] + "...", language="text")

                st.markdown("**Selected Chunks:**")
                for sc in log.selected_chunks:
                    st.write(f"• `{sc['chunk_id']}` | score={sc['score']} | "
                             f"{sc['text'][:100]}...")

        # ── Pure LLM comparison ───────────────────────────────────
        if pure_llm_compare:
            with st.expander("🤖 Pure-LLM Response (no retrieval)", expanded=False):
                with st.spinner("Running pure LLM..."):
                    llm_response, _ = pipeline.query(
                        user_input, pure_llm_mode=True
                    )
                st.markdown("**RAG Response:**")
                st.info(response[:400])
                st.markdown("**Pure LLM Response:**")
                st.warning(llm_response[:400] if llm_response else
                           "[No API key — cannot call LLM]")
                st.caption("RAG uses retrieved documents; Pure LLM uses only "
                           "the model's training data (may hallucinate).")

# ─────────────────────────────────────────────
# BOTTOM — EXAMPLE QUERIES
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("#### 💡 Try these example queries:")
examples = [
    "What is Ghana's projected GDP growth for 2025?",
    "Who got the highest votes in Ashanti Region?",
    "What are Ghana's key revenue measures in the 2025 budget?",
    "What is the inflation target set in the 2025 budget?",
    "How did NDC perform in the Greater Accra Region?",
    "What is Ghana's debt situation according to the 2025 budget?",
]
cols = st.columns(2)
for i, ex in enumerate(examples):
    if cols[i % 2].button(ex, use_container_width=True):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": ex})
        with st.chat_message("user"):
            st.markdown(ex)

        # Run pipeline
        pipeline = st.session_state.pipeline
        pipeline.llm.api_key = api_key
        pipeline.llm.provider = provider
        pipeline.llm.model = pipeline.llm._default_model()

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                response, log = pipeline.query(
                    ex,
                    top_k=top_k,
                    template_version=tv,
                    pure_llm_mode=False
                )

            # ── Response ─────────────────────────────────────────────
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            # ── Retrieved chunks ──────────────────────────────────────
            with st.expander(f"📄 Retrieved Chunks ({len(log.retrieved_chunks)})", expanded=False):
                for i, chunk in enumerate(log.retrieved_chunks, 1):
                    src_class = ("source-badge-budget" if chunk["source"] == "budget"
                                 else "source-badge-election")
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<span class="score-badge">#{i} score={chunk["score"]}</span>'
                        f'<span class="score-badge {src_class}">'
                        f'{chunk["source"].upper()} p.{chunk["page"]}</span><br><br>'
                        f'{chunk["text_snippet"]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ── Memory context ────────────────────────────────────────
            if log.memory_context:
                with st.expander("🧠 Memory Context Used", expanded=False):
                    st.markdown(
                        f'<span class="memory-badge">SESSION MEMORY</span><br><br>'
                        f'<pre style="font-size:0.8rem">{log.memory_context[:600]}</pre>',
                        unsafe_allow_html=True
                    )

            # ── Failure warning ───────────────────────────────────────
            if log.failure_detected:
                st.markdown(
                    f'<div class="failure-warning">⚠️ <strong>Retrieval warning:</strong> '
                    f'{log.failure_reason} — fallback retrieval was used.</div>',
                    unsafe_allow_html=True
                )

            # ── Debug view ────────────────────────────────────────────
            if show_debug:
                with st.expander("🔬 Pipeline Debug", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt Tokens", log.prompt_tokens)
                    col2.metric("Latency (ms)", f"{log.latency_ms:.0f}")
                    col3.metric("Template", log.template_version)

                    st.markdown("**Expanded Query:**")
                    st.code(log.expanded_query or ex)

                    st.markdown("**Final Prompt (truncated):**")
                    st.code(log.final_prompt[:800] + "...", language="text")

                    st.markdown("**Selected Chunks:**")
                    for sc in log.selected_chunks:
                        st.write(f"• `{sc['chunk_id']}` | score={sc['score']} | "
                                 f"{sc['text'][:100]}...")

            # ── Pure LLM comparison ───────────────────────────────────
            if pure_llm_compare:
                with st.expander("🤖 Pure-LLM Response (no retrieval)", expanded=False):
                    with st.spinner("Running pure LLM..."):
                        llm_response, _ = pipeline.query(
                            ex, pure_llm_mode=True
                        )
                    st.markdown("**RAG Response:**")
                    st.info(response[:400])
                    st.markdown("**Pure LLM Response:**")
                    st.warning(llm_response[:400] if llm_response else
                               "[No API key — cannot call LLM]")
                    st.caption("RAG uses retrieved documents; Pure LLM uses only "
                               "the model's training data (may hallucinate).")

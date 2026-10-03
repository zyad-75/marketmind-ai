from pathlib import Path

import streamlit as st

from src.agent import ask_agent


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "data"
)

VECTOR_STORE_DIR = (
    BASE_DIR
    / "vector_store"
)


st.set_page_config(
    page_title="MarketMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Welcome Message
# ============================================================

def welcome_message():

    return {
        "role": "assistant",
        "content": (
            "Hi! I'm **MarketMind AI** 🧠\n\n"
            "I can answer questions using NovaGlow's "
            "knowledge base, generate marketing campaigns, "
            "and analyze customer segments."
        ),
        "intent": None,
        "confidence": None,
        "sources": [],
    }


# ============================================================
# Confidence Helper
# ============================================================

def get_confidence(
    result: dict
) -> str:

    structured = (
        result.get(
            "structured"
        )
        or {}
    )

    confidence = (
        structured.get(
            "confidence",
            "Unknown",
        )
    )

    if confidence not in {
        "High",
        "Medium",
        "Low",
    }:

        return "Unknown"

    return confidence


# ============================================================
# Sources Helper
# ============================================================

def get_sources(
    result: dict
) -> list[str]:

    structured = (
        result.get(
            "structured"
        )
        or {}
    )

    sources = (
        result.get(
            "sources"
        )
        or structured.get(
            "sources"
        )
        or []
    )

    # Remove duplicate sources
    # while preserving order.

    return list(
        dict.fromkeys(
            sources
        )
    )


# ============================================================
# Intent UI
# ============================================================

def render_intent(
    intent: str
):

    mapping = {
        "FACT_LOOKUP": "🔎",
        "CAMPAIGN_GENERATION": "📣",
        "SEGMENT_ANALYSIS": "👥",
        "ERROR": "⚠️",
    }

    icon = mapping.get(
        intent,
        "🤖",
    )

    st.caption(
        f"{icon} Intent: {intent}"
    )


# ============================================================
# Confidence UI
# ============================================================

def render_confidence(
    confidence: str
):

    if confidence == "High":

        st.success(
            "🟢 Confidence: High"
        )

    elif confidence == "Medium":

        st.warning(
            "🟡 Confidence: Medium"
        )

    elif confidence == "Low":

        st.error(
            "🔴 Confidence: Low"
        )

    else:

        st.info(
            "⚪ Confidence: Unknown"
        )


# ============================================================
# Sources UI
# ============================================================

def render_sources(
    sources: list[str]
):

    if not sources:
        return

    with st.expander(
        "📚 Knowledge Sources",
        expanded=False,
    ):

        for source in sources:

            st.markdown(
                f"- `{source}`"
            )


# ============================================================
# Saved Assistant Message Renderer
# ============================================================

def render_assistant_message(
    message: dict
):

    intent = (
        message.get(
            "intent"
        )
    )

    if intent:

        render_intent(
            intent
        )

        render_confidence(
            message.get(
                "confidence",
                "Unknown",
            )
        )

    st.markdown(
        message.get(
            "content",
            "",
        )
    )

    render_sources(
        message.get(
            "sources",
            [],
        )
    )


# ============================================================
# Session State
# ============================================================

if (
    "messages"
    not in st.session_state
):

    st.session_state.messages = [
        welcome_message()
    ]


if (
    "pending_prompt"
    not in st.session_state
):

    st.session_state.pending_prompt = None


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title(
        "🧠 MarketMind AI"
    )

    st.caption(
        "Local RAG Marketing Agent"
    )

    st.divider()

    # ========================================================
    # System Status
    # ========================================================

    st.subheader(
        "⚙️ System Status"
    )

    if DATA_DIR.exists():

        pdf_files = list(
            DATA_DIR.glob(
                "*.pdf"
            )
        )

    else:

        pdf_files = []

    if pdf_files:

        st.success(
            f"Knowledge Base: "
            f"{len(pdf_files)} PDFs"
        )

    else:

        st.error(
            "Knowledge Base: Missing"
        )

    index_file = (
        VECTOR_STORE_DIR
        / "index.faiss"
    )

    metadata_file = (
        VECTOR_STORE_DIR
        / "index.pkl"
    )

    if (
        index_file.exists()
        and metadata_file.exists()
    ):

        st.success(
            "FAISS Vector Store: Ready"
        )

    else:

        st.warning(
            "FAISS Vector Store: Missing"
        )

    st.success(
        "LLM: Llama 3.2 1B"
    )

    st.success(
        "Runtime: Ollama"
    )

    st.divider()

    # ========================================================
    # AI Stack
    # ========================================================

    st.subheader(
        "🧩 AI Stack"
    )

    st.markdown(
        """
**LLM:** Llama 3.2 1B  
**Runtime:** Ollama  
**Embeddings:** MiniLM-L6-v2  
**Vector DB:** FAISS  
**Framework:** LangChain  
**Validation:** Pydantic  
**UI:** Streamlit
"""
    )

    st.divider()

    # ========================================================
    # Capabilities
    # ========================================================

    st.subheader(
        "🎯 Capabilities"
    )

    st.markdown(
        """
- 🔎 Fact Lookup
- 📣 Campaign Generation
- 👥 Segment Analysis
- 📚 RAG Grounding
- 🛡️ Hallucination Guardrails
- 🔢 Deterministic Fact Extraction
- 📦 Structured Output
"""
    )

    st.divider()

    # ========================================================
    # Clear
    # ========================================================

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = [
            welcome_message()
        ]

        st.session_state.pending_prompt = None

        st.rerun()


# ============================================================
# Main Header
# ============================================================

st.title(
    "🧠 MarketMind AI"
)

st.subheader(
    "AI Marketing Strategist for NovaGlow"
)

st.write(
    "MarketMind uses Retrieval-Augmented Generation "
    "(RAG) to ground its answers in NovaGlow's "
    "internal marketing knowledge base."
)


# ============================================================
# Architecture
# ============================================================

with st.expander(
    "🏗️ How MarketMind Works"
):

    architecture = """
User Query
    |
    v
Intent Router
    |
    +-- FACT_LOOKUP
    +-- CAMPAIGN_GENERATION
    +-- SEGMENT_ANALYSIS
    |
    v
RAG Retrieval
    |
    v
FAISS Vector Store
    |
    v
Relevant NovaGlow Knowledge
    |
    +-- Deterministic Fact Extraction
    |
    +-- Llama 3.2 1B Reasoning
             |
             v
         Guardrails
             |
             v
      Pydantic Validation
             |
             v
      Structured Response
             |
       +-----+-----+
       |           |
       v           v
    Sources    Confidence
"""

    st.code(
        architecture,
        language="text",
    )


# ============================================================
# Example Questions
# ============================================================

st.markdown(
    "### 💡 Example Questions"
)

col1, col2 = (
    st.columns(
        2
    )
)


with col1:

    if st.button(
        "🔎 Morning Made Simple CTR",
        use_container_width=True,
    ):

        st.session_state.pending_prompt = (
            "What was the CTR of the "
            "Morning Made Simple campaign?"
        )

        st.rerun()


    if st.button(
        "👥 Analyze Young Professional",
        use_container_width=True,
    ):

        st.session_state.pending_prompt = (
            "Analyze the Young Professional persona."
        )

        st.rerun()


with col2:

    if st.button(
        "📣 Generate HydraBoost Campaign",
        use_container_width=True,
    ):

        st.session_state.pending_prompt = (
            "Create a marketing campaign for "
            "HydraBoost Serum targeting young professionals."
        )

        st.rerun()


    if st.button(
        "🛡️ Hallucination Test",
        use_container_width=True,
    ):

        st.session_state.pending_prompt = (
            "What was NovaGlow's Facebook "
            "campaign conversion rate?"
        )

        st.rerun()


st.divider()


# ============================================================
# Existing Chat History
# ============================================================

for message in (
    st.session_state.messages
):

    role = (
        message.get(
            "role",
            "assistant",
        )
    )

    with st.chat_message(
        role
    ):

        if (
            role
            == "assistant"
        ):

            render_assistant_message(
                message
            )

        else:

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )


# ============================================================
# User Input
# ============================================================

typed_prompt = st.chat_input(
    "Ask MarketMind about NovaGlow..."
)

prompt = None


if (
    st.session_state.pending_prompt
):

    prompt = (
        st.session_state.pending_prompt
    )

    st.session_state.pending_prompt = None


elif typed_prompt:

    prompt = typed_prompt


# ============================================================
# Process Query
# ============================================================

if prompt:

    # ========================================================
    # Save User Message
    # ========================================================

    user_message = {
        "role": "user",
        "content": prompt,
    }

    st.session_state.messages.append(
        user_message
    )

    # ========================================================
    # Display User
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    # ========================================================
    # Assistant Response
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "MarketMind is retrieving knowledge "
            "and reasoning..."
        ):

            try:

                result = (
                    ask_agent(
                        prompt
                    )
                )

                # ============================================
                # Extract Data
                # ============================================

                intent = (
                    result.get(
                        "intent",
                        "UNKNOWN",
                    )
                )

                answer = (
                    result.get(
                        "answer",
                        "No response generated.",
                    )
                )

                confidence = (
                    get_confidence(
                        result
                    )
                )

                sources = (
                    get_sources(
                        result
                    )
                )

                # ============================================
                # Render
                # ============================================

                render_intent(
                    intent
                )

                render_confidence(
                    confidence
                )

                st.markdown(
                    answer
                )

                render_sources(
                    sources
                )

                # ============================================
                # Save Assistant Message
                # ============================================

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "confidence": confidence,
                    "sources": sources,
                }

                st.session_state.messages.append(
                    assistant_message
                )

            except Exception as error:

                error_text = str(
                    error
                )

                st.error(
                    "MarketMind could not complete "
                    "the request."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        error_text,
                        language="text",
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "MarketMind could not complete "
                            "the request."
                        ),
                        "intent": "ERROR",
                        "confidence": "Unknown",
                        "sources": [],
                    }
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "MarketMind AI • "
    "Llama 3.2 1B + Ollama + RAG + FAISS + "
    "LangChain + Pydantic"
)
"""My personal AI assistant: a romantic chatbot built with Ollama, LangChain and Streamlit."""

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from PIL import Image

# ---------------------------------------------------------------- config ---
load_dotenv()

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"
DEFAULT_MODEL = "deepseek-r1:1.5b"
SYSTEM_PROMPT = (
    "You are a smart, romantic AI assistant. Respond to questions in a warm, "
    "romantic way whenever it fits. Otherwise, just give a clear, good answer "
    "like a gentleman."
)

# LangSmith tracing (optional: only switched on if a key exists in .env)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("OLLAMA_PROJECT", "AppWithOllama")

logo = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None
avatar = str(LOGO_PATH) if LOGO_PATH.exists() else "🤖"

st.set_page_config(
    page_title="My Personal AI Assistant",
    page_icon=logo or "💬",
    layout="centered",
)

# ------------------------------------------------------------------ style ---
st.markdown(
    """
    <style>
    .hero-title {
        font-size: 2.4rem; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #f7a35c, #c471ed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub { text-align: center; opacity: .7; margin-top: .2rem; margin-bottom: 1.5rem; }
    .block-container { padding-top: 5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers ---
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    """Separate DeepSeek-R1's <think> reasoning from the final answer."""
    thoughts = "\n".join(m.strip() for m in THINK_RE.findall(text))
    answer = THINK_RE.sub("", text).strip()
    return answer, thoughts


@st.cache_resource(show_spinner=False)
def get_chain(model: str, temperature: float):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{question}"),
        ]
    )
    llm = OllamaLLM(model=model, temperature=temperature)
    return prompt | llm | StrOutputParser()


def to_langchain_history(messages: list[dict]):
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in messages
    ]


def render_message(msg: dict, show_reasoning: bool):
    with st.chat_message(msg["role"], avatar=avatar if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if show_reasoning and msg.get("thinking"):
            with st.expander("💭 See the model's reasoning"):
                st.markdown(msg["thinking"])


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    if logo:
        st.image(logo, width=90)
    st.header("Settings")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL, help="Must be pulled first: `ollama pull <model>`")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    show_reasoning = st.toggle("Show model reasoning", value=False)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Runs 100% locally with Ollama. Nothing leaves your machine.")

# ------------------------------------------------------------------- main ---
st.markdown('<p class="hero-title">My Personal AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask me anything, I promise to be a gentleman about it 🌹</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=avatar):
        st.markdown("Hello, darling. What's on your mind today? Spill it out, I'm all yours. ✨")

for message in st.session_state.messages:
    render_message(message, show_reasoning)

if question := st.chat_input("Spill it out! What question do you have in your mind?"):
    history = to_langchain_history(st.session_state.messages)
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg, show_reasoning)

    with st.chat_message("assistant", avatar=avatar):
        try:
            with st.spinner("Thinking of something lovely..."):
                raw = get_chain(model_name, temperature).invoke(
                    {"question": question, "history": history}
                )
            answer, thinking = split_thinking(raw)
            st.markdown(answer)
            if show_reasoning and thinking:
                with st.expander("💭 See the model's reasoning"):
                    st.markdown(thinking)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "thinking": thinking}
            )
        except Exception as exc:
            st.error(
                "I couldn't reach the model. Make sure Ollama is running and the model "
                f"is pulled (`ollama pull {model_name}`).\n\nDetails: `{exc}`"
            )
            st.session_state.messages.pop()  # drop the unanswered question
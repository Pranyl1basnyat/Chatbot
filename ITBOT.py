import os
import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -----------------------
# Configuration
# -----------------------

st.set_page_config(
    page_title="Local AI Knowledge Bot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "distilgpt2"

# -----------------------
# Prompt Template
# -----------------------

def build_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. Don't make up any information.

Context:
{context}

Question: {question}

Answer:
""".strip(),
        input_variables=["context", "question"]
    )

# -----------------------
# Model & Vectorstore Loaders
# -----------------------

@st.cache_resource(show_spinner=True)
def load_local_llm() -> HuggingFacePipeline:
    try:
        local_pipe = pipeline(
            "text-generation",
            model=LOCAL_LLM_MODEL,
            tokenizer=LOCAL_LLM_MODEL,
            max_new_tokens=200,
            temperature=0.7,
            pad_token_id=50256  # Required for distilgpt2 to avoid warnings
        )
        return HuggingFacePipeline(pipeline=local_pipe)
    except Exception as e:
        st.error(f"❌ Failed to load local model `{LOCAL_LLM_MODEL}`: {e}")
        st.stop()

@st.cache_resource(show_spinner=True)
def load_vectorstore() -> FAISS:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore at `{DB_FAISS_PATH}`: {e}")
        st.stop()

# -----------------------
# RetrievalQA Chain
# -----------------------

def build_qa_chain(llm: HuggingFacePipeline, vectorstore: FAISS) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": build_prompt()}
    )

# -----------------------
# Sidebar
# -----------------------

st.sidebar.title("⚙️ Settings")
st.sidebar.info(f"Using local model: `{LOCAL_LLM_MODEL}`")
st.sidebar.success("FAISS vector DB loaded successfully.")

# -----------------------
# Main App UI
# -----------------------

st.title("🧠 Local AI Knowledge Bot")
st.markdown("Ask a question about your local documents.")

# Load resources
llm = load_local_llm()
vectorstore = load_vectorstore()
qa_chain = build_qa_chain(llm, vectorstore)

# Initialize chat session
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything based on the loaded documents."}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking..."):
            try:
                result = qa_chain.invoke({"query": user_input})
                answer = result.get("result", "No response.")

                st.markdown(answer)
                with st.expander("📚 Show Sources"):
                    for i, doc in enumerate(result.get("source_documents", []), 1):
                        st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'N/A')}`")
                        st.code(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

            except Exception as e:
                answer = f"⚠️ Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

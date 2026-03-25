# app.py

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# -----------------------------
# 🔹 Page Config
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 AI Knowledge Chatbot")
st.write("Ask anything about AI / ML!")

# -----------------------------
# 🔹 Safety Layer
# -----------------------------
danger_keywords = [
    "overdose", "kill myself", "suicide", "stop my heart",
    "high dose", "mg dosage"
]

def is_dangerous(text):
    return any(word in text.lower() for word in danger_keywords)

# -----------------------------
# 🔹 Load Vector Store
# -----------------------------
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("knowledge_base/ai_ml_notes.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# 🔹 Load LLM
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

# -----------------------------
# 🔹 Init
# -----------------------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
generator = load_model()

# -----------------------------
# 🔹 RAG Function
# -----------------------------
def rag_chatbot(query):

    if is_dangerous(query):
        return "⚠️ I can't help with harmful or medical-related queries."

    # ✅ CORRECT METHOD
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant info found."

    # ✅ MULTI-DOC CONTEXT (IMPORTANT)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ✅ STRONGER PROMPT
    prompt = f"""
You are a helpful AI assistant.

Rules:
- Answer ONLY from the context
- If answer not in context, say "I don't know"
- Keep answer short and clear

Context:
{context}

Question:
{query}

Answer:
"""

    output = generator(
        prompt,
        max_length=250,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    # ✅ CLEAN OUTPUT
    answer = output[0]["generated_text"].split("Answer:")[-1].strip()

    return answer


# -----------------------------
# 🔹 UI
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask something:")

if user_input:
    response = rag_chatbot(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

# Display chat
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
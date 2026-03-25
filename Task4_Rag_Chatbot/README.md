# 🤖 Task 4 – RAG (Retrieval-Augmented Generation) Chatbot

This project is part of my **AI/ML Internship at DevelopersHub Corporation**.  
It is a **context-aware chatbot** that answers questions using a **knowledge base** of AI/ML concepts.  

The chatbot uses **LangChain**, **FAISS**, and **HuggingFace Transformers** to retrieve relevant documents and generate answers in real-time.

---

## 🗂 Folder Structure
Task4_RAG_Chatbot/
│
├── app.py ← Streamlit app file
├── requirements.txt ← Project dependencies
└── knowledge_base/
└── ai_ml_notes.txt ← Text file containing AI/ML knowledge base

---

## ⚡ Features

- **Context-aware**: Retrieves relevant documents from knowledge base
- **RAG (Retrieval-Augmented Generation)**: Combines retrieval and text-generation
- **Fast response**: Uses caching for embeddings and model
- **Interactive UI**: Built with Streamlit
- **Portfolio-ready**: Suitable for internship/assessment demonstration

---

## 🛠 Technologies Used

- **Python 3.x**
- **Streamlit** – for building interactive web UI
- **LangChain** – document retrieval & vectorstore
- **FAISS** – vector similarity search
- **HuggingFace Transformers** – text generation (DistilGPT2)
- **Sentence-Transformers** – embeddings
- **Torch** – backend for transformers

---

## 🚀 How to Run Locally

1. Clone this repository:

```bash
git clone <your-repo-link>
cd Task4_RAG_Chatbot
2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Run the Streamlit app:
   streamlit run app.py
5. First run may take 1–2 minutes to download the model and build embeddings. After that, it will load faster.

🧩 How It Works

Load Knowledge Base:
Text file is loaded and split into smaller chunks.

Create Embeddings:
Each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2.

VectorStore:
FAISS vectorstore stores embeddings for similarity search.

Retrieve Relevant Docs:
For any user query, top-k similar documents are retrieved.

Generate Answer:
Prompt is sent to distilgpt2 to generate a coherent answer using retrieved context.

Display in UI:
Streamlit interface shows chat messages and history.

💡 Usage

Type a question in the input box (e.g., "What is a Decision Tree?")

Bot will return a response using knowledge base content

Example questions:

"Explain Logistic Regression"

"Difference between Random Forest and Decision Tree"

"What is a Transformer model?"

🏆 Key Skills Demonstrated

Retrieval-Augmented Generation (RAG) pipeline

Vector similarity search using FAISS

Embeddings creation using HuggingFace Sentence-Transformers

Transformer-based text generation (DistilGPT2)

Interactive web app development with Streamlit

Professional GitHub project structure

📂 Notes

Ensure knowledge_base/ai_ml_notes.txt exists and contains the relevant AI/ML content.

Streamlit caching (@st.cache_resource) is used for fast reloading of embeddings and model.

🔗 References

LangChain Documentation

FAISS

HuggingFace Transformers

Streamlit

# AI_ML_Internship-phase-2-
# End-to-End ML Pipeline for Customer Churn Prediction

## Objective
Build a **production-ready machine learning pipeline** to predict customer churn for a telecom company.  
Customer churn occurs when customers leave the service, and predicting it helps the company take proactive measures to retain them.

---

## Dataset
**Telco Customer Churn Dataset** contains customer information including:

- Gender  
- Tenure (months as a customer)  
- Contract type  
- Monthly charges  
- Internet service type  
- And other relevant features  

**Target Variable:** `Churn` (Yes / No)

Dataset Link: [Telco Customer Churn](https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv)

---

## Methodology

1. **Data Preprocessing**
   - Dropped irrelevant columns like `customerID`.  
   - Converted `Churn` to numeric (Yes → 1, No → 0).  
   - Identified categorical and numerical features.  
   - Applied **OneHotEncoder** to categorical features.  
   - Applied **StandardScaler** to numerical features.  
   - Combined preprocessing steps using **ColumnTransformer**.

2. **Machine Learning Pipeline**
   - Pipelines built for:
     - **Logistic Regression**
     - **Random Forest Classifier**  
   - Ensures preprocessing and model steps are applied together for reproducibility.

3. **Hyperparameter Tuning**
   - Used **GridSearchCV** to find best parameters:
     - Logistic Regression: tested multiple `C` values.  
     - Random Forest: tested `n_estimators` and `max_depth`.

4. **Model Evaluation**
   - Evaluated both models using:
     - Accuracy  
     - Precision, Recall, F1-Score  
     - Classification Report

5. **Model Comparison**
   - Compared models based on accuracy.  
   - Selected the **best-performing model** for deployment.

6. **Export**
   - Saved the final trained pipeline using **Joblib** (`churn_pipeline.pkl`) including preprocessing and model.  
   - Ready for reuse in production without repeating preprocessing or training.

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn (Pipeline, GridSearchCV, ColumnTransformer)  
- Matplotlib & Seaborn (visualization)  
- Joblib (model export)

---

## Results

| Model | Accuracy |
|-------|---------|
| Logistic Regression | 0.XX |
| Random Forest      | 0.XX |

> The model with higher accuracy was selected as the final model.

---

## Conclusion

This project demonstrates how to build a **scalable, reusable, and production-ready machine learning pipeline**.  
It includes **data preprocessing, model training, hyperparameter tuning, model comparison, and deployment**.  
The workflow can be adapted to other datasets and models, making it a flexible ML solution for real-world applications.

---

## How to Run

## Clone the repository:
bash
git clone <your-repo-link>
Install dependencies:

pip install -r requirements.txt

Open churn_pipeline.ipynb to view the full workflow.

The final model is saved as churn_pipeline.pkl for predictions on new data.


# 📰 News Topic Classifier Using BERT

## 📌 Objective
The objective of this project is to build a **News Topic Classification system** using a **Transformer-based model (BERT)**.  
The model classifies news headlines into predefined categories such as **World, Sports, Business, and Sci/Tech**.

This project demonstrates the use of **transfer learning with BERT**, fine-tuning a pre-trained transformer model on a news dataset and deploying it using **Streamlit for live interaction**.

---

## 📊 Dataset
The dataset used in this project is the **AG News Dataset**, which is publicly available on Hugging Face.

The dataset contains **news headlines and their corresponding topic labels**.

### Categories
- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Sci/Tech

Dataset Source:
https://huggingface.co/datasets/ag_news

---

## ⚙️ Methodology

### 1️⃣ Data Loading
The AG News dataset was loaded using the **Hugging Face Datasets library**.

### 2️⃣ Data Preprocessing
- Text tokenization using **BERT tokenizer**
- Padding and truncation of sequences
- Conversion of text into numerical tokens suitable for transformer models

### 3️⃣ Model Selection
The pre-trained **bert-base-uncased** model from Hugging Face Transformers was used.

### 4️⃣ Model Fine-Tuning
The model was fine-tuned on the AG News dataset for **text classification**.

Training included:
- Cross-entropy loss
- Optimization using AdamW
- Transformer-based feature extraction

### 5️⃣ Model Evaluation
The model performance was evaluated using:

- **Accuracy**
- **F1 Score**

These metrics help measure the effectiveness of the classifier.

### 6️⃣ Deployment
The trained model was deployed using **Streamlit**, allowing users to enter a news headline and receive a predicted category in real time.

---

## 🖥️ Streamlit Application

The application allows users to:

1. Enter a news headline
2. Click **Predict**
3. View the predicted news category

Example:

Input: Apple releases new AI chip for mobile devices

Output: Sci/Tech

---

## 📂 Project Structure
Task1_BERT_News_Classifier/

├── bert_news_classifier.ipynb

├── app.py

├── requirements.txt

└── README.md

---

## 📦 Model Files

Due to **GitHub file size limitations**, the trained BERT model files are not included in this repository.

To generate the trained model:

1. Run the notebook:
2. The trained model will automatically be saved in a folder named: bert_news_model/
3. The Streamlit application will then load the model from this folder.

---

## ▶️ Running the Project

### 1️⃣ Install Dependencies
pip install -r requirements.txt


### 2️⃣ Run the Streamlit App


### 3️⃣ Open in Browser
http://localhost:8501


---

## 🧠 Skills Demonstrated

- Natural Language Processing (NLP)
- Transformer Models (BERT)
- Transfer Learning
- Text Tokenization
- Model Fine-Tuning
- Model Evaluation (Accuracy, F1 Score)
- Streamlit Deployment
- Hugging Face Transformers

---

## 🚀 Conclusion

This project successfully demonstrates how **pre-trained transformer models like BERT can be fine-tuned for text classification tasks**.

The deployed Streamlit application provides an interactive interface for predicting news categories in real time, showcasing a practical application of modern NLP techniques.

---

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





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




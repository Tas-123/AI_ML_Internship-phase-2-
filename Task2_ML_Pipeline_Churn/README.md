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

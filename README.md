# DSA4263 Project: Detecting Fraudulent Transactions in Banking

This project builds a machine learning models to detect fraudulent transactions using real-world online payments data from [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset).

_This project was done by: Duocai, Edwards, Faldho, Vernen_

---

## 📚 Table of Contents

- [DSA4263 Project: Detecting Fraudulent Transactions in Banking](#dsa4263-project-detecting-fraudulent-transactions-in-banking)
  - [📚 Table of Contents](#-table-of-contents)
  - [1. Brief Summary](#1-brief-summary)
    - [1.1 Introducing Our Dataset](#11-introducing-our-dataset)
    - [1.2 EDA](#12-eda)
    - [1.3 Feature Selection](#13-feature-selection)
  - [2. Our Models](#2-our-models)
    - [2.1 Results from the Models](#21-results-from-the-models)
  - [| LightGBM         | 0.96    | 0.84      | 0.68           | 0.75     |](#-lightgbm----------096-----084-------068------------075-----)
  - [3. Discussion](#3-discussion)
  - [4. How to Run This Project](#4-how-to-run-this-project)
    - [4.1 Clone the Repo](#41-clone-the-repo)
    - [4.2 Download Dataset](#42-download-dataset)
    - [4.3 Run the Notebooks](#43-run-the-notebooks)
    - [4.4 Load Our Trained Models](#44-load-our-trained-models)

---

## 1. Brief Summary

In this project, we aimed to build a robust machine learning model to detect fraudulent transactions in online banking transactionss. We began by performing extensive data exploration on the dataset from Kaggle. Through exploratory data analysis, we examined the distribution of transaction types, identified imbalances between fraudulent and non-fraudulent transactions, and visualized key relationships using heatmaps and distribution plots.

To establish a performance baseline, we first trained a Logistic Regression model. This simple linear model helped us understand how well standard classification techniques perform under class imbalance. Following this, we developed two more sophisticated challenger models: XGBoost and LightGBM. These gradient boosting algorithms were chosen for their performance with imbalanced datasets and ability to capture non-linear relationships.

Evaluation was carried out using metrics like precision, recall, F1-score, and AUC-ROC — with a particular focus on recall due to the importance of correctly identifying fraudulent transactions. Finally, we fine-tuned our models using custom decision thresholds to balance false positives and false negatives. Our project is reproducible, and easily extendable, with support for saved models and a `main.ipynb` notebook that executes all the notebooks.

---

### 1.1 Introducing Our Dataset

 We utilised the [Online Payments Fraud Detection Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) from Kaggle. This dataset consists of **6,362,620 rows** and **11 columns**, offering detailed information about individual payment transactions.

The dataset includes the following features:

- **step**: Represents a unit of time, where 1 step = 1 hour.
- **type**: Type of transaction (e.g., PAYMENT, TRANSFER, CASH_OUT).
- **amount**: The amount involved in the transaction.
- **nameOrig**: Customer initiating the transaction.
- **oldbalanceOrg**: Balance of the sender before the transaction.
- **newbalanceOrig**: Balance of the sender after the transaction.
- **nameDest**: Recipient of the transaction.
- **oldbalanceDest**: Balance of the recipient before the transaction.
- **newbalanceDest**: Balance of the recipient after the transaction.
- **isFraud**: Flag indicating if the transaction was fraudulent (1) or not (0).
- **isFlaggedFraud**: Flag indicating if the transaction was flagged by the system as suspicious.

The dataset is notably clean — with **no missing or corrupted values** — which meant minimal data cleaning was required. This allowed us to focus more on exploratory data analysis and model development.

While the exact origin of the data is not fully detailed on the dataset’s Kaggle page, it appears to be **synthetically generated** to simulate real-world transactions, potentially originating from banking environments. This is supported by its close resemblance to another dataset on [Kaggle](https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction).

Overall, this dataset provides a realistic and challenging foundation for fraud detection modeling, especially due to the extreme class imbalance and detailed transaction-level data.

---

### 1.2 EDA

We began our analysis by conducting EDA to understand the structure, distribution, and key characteristics of the dataset. This step was crucial in uncovering patterns, spotting anomalies, and identifying relationships between features that may influence fraud prediction. Visualisations such as distribution plots, bar charts, and correlation heatmaps were used to support these observations.

➡️ For full details and visual insights, refer to the notebook:  
[`1.0-faldho-linitial-data-exploration.ipynb`](Notebooks/1.0-faldho-linitial-data-exploration.ipynb)

---

### 1.3 Feature Selection

Based on our insights from the EDA, we carefully selected features that contribute meaningful information for fraud prediction while excluding those that offer little to no predictive value.

The following features were excluded from our model:

- **nameOrig** and **nameDest**: These are anonymised identifiers for the sender and receiver of a transaction. While they may help in detecting specific account-level behaviours, they are high-cardinality categorical variables with no inherent meaning or patterns that generalise well across the dataset.

- **step**: This represents the time of the transaction in hourly intervals. Initially, our analysis showed no obvious relationship between `step` and fraud occurrences, suggesting limited time-dependency. However, during model experimentation, we observed that including `step` notably improved the performance of the LightGBM model. This suggests that `step` may contain subtle temporal patterns or positional cues that are non-trivial to detect during EDA but are leveraged effectively by gradient boosting algorithms. As a result, we retained `step` in our final feature set for LightGBM.

- **isFlaggedFraud**: Although intended to indicate whether a transaction was flagged as suspicious, all known fraud cases had this field set to 0. Thus, it did not help in detecting actual fraud and was dropped.

As a result, we retained the most relevant numerical and categorical features that are directly linked to transaction behaviour, such as amount, type, oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest.

This feature selection not only improved model interpretability and training efficiency but also reduced noise, leading to more robust performance in our final models.

---

## 2. Our Models

We trained and evaluated several models:
- Logistic Regression (Baseline)
- XGBoost
- LightGBM

---

### 2.1 Results from the Models

| Model            | AUC-ROC | Precision | Recall (Fraud) | F1 Score |
|------------------|---------|-----------|----------------|----------|
| Logistic         | 0.89    | 0.20      | 0.60           | 0.30     |
| XGBoost          | 0.97    | 0.85      | 0.72           | 0.78     |
| LightGBM         | 0.96    | 0.84      | 0.68           | 0.75     |
---

## 3. Discussion

...

---

## 4. How to Run This Project

### 4.1 Clone the Repo

```bash
git clone https://github.com/vernenlim/DSA4263.git
cd DSA4263
```
---

### 4.2 Download Dataset

Due to the dataset's large size, we were unable to upload it directly to this repository. Instead, we provide two ways for you to obtain the dataset:

1. **Run the download script**  
   We’ve prepared a Python script that uses the Kaggle API to automatically download and extract the dataset into the correct folder.

   To use it, make sure you have your Kaggle API credentials set up (see [Kaggle API setup guide](https://www.kaggle.com/docs/api)) and then run the following command:

   ```bash
   python src/data/download_dataset.py
   ```

This will download and unzip the dataset into the Dataset/ folder.

2. Manual download  

   Alternatively, you can manually download the CSV file from the [Kaggle dataset page](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) and place it in the Dataset/ directory.  

---

### 4.3 Run the Notebooks

Open `Notebooks/main.ipynb` in **Jupyter Notebook** and run all cells. This will run all the notebooks and show all the outputs in each notebook:

- EDA
- Base model
- Xgboost model
- LightGBM model

---

### 4.4 Load Our Trained Models

We provide saved model files in the `Models/` directory:

- `best_LGBM_model.txt`

To load the models, make sure you are in the root of the cloned repository (DSA4263/). Then, use the following Python code:
```python
# Load the saved Logistic Regression Model
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load model parameters from .npz file
data = np.load('Models/best_lr_model.npz')

loaded_lr_model = LogisticRegression()
loaded_lr_model.coef_ = data['coef']
loaded_lr_model.intercept_ = data['intercept']
loaded_lr_model.classes_ = data['classes']

# Load the saved XGBoost model
import xgboost as xgb

loaded_xgb_model = xgb.XGBClassifier()
loaded_xgb_model.load_model('Models/best_xgb_model.json')

# Load the saved lightgbm
import lightgbm as lgb

model = lgb.Booster(model_file='Models/best_LGBM_booster.txt')
```
Now you can use them without re-training.

---

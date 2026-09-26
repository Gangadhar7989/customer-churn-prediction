# customer-churn-prediction
This project involves building an Artificial Neural Network (ANN) for predicting customer churn. The dataset used contains various customer attributes, and the ANN is trained to predict whether a customer is likely to leave the bank.

A beginner-to-intermediate data science project that predicts whether a telecom
customer will churn (cancel their service), based on their account and usage
data. Built to practice — and demonstrate — the full data science workflow:
data cleaning, exploratory data analysis (EDA), feature engineering, model
training, and evaluation.

## Why this project

Customer churn prediction is one of the most common real-world use cases for
machine learning in industry (subscriptions, telecom, SaaS, banking). This
project mirrors that workflow end to end on a realistic synthetic dataset.

## What it does

1. **Generates a synthetic dataset** of 2,000 customers with realistic
   features (tenure, contract type, monthly charges, internet service, etc.)
2. **Cleans the data** — handles missing values, encodes categorical features
3. **Explores the data** — generates charts showing churn patterns by
   contract type, tenure, and monthly charges
4. **Trains a Random Forest classifier** to predict churn
5. **Evaluates the model** — accuracy, ROC AUC, confusion matrix, feature
   importance
6. **Predicts on new/sample customers**

## Results

On the held-out test set, the model achieves:

| Metric | Score |
|---|---|
| Accuracy | ~84% |
| ROC AUC | ~0.89 |

*(Exact numbers vary slightly by random seed but stay in this range.)*

### Sample charts

The `outputs/` folder (generated when you run the project) includes:
- `churn_by_contract.png` — customers on month-to-month contracts churn far
  more than those on annual contracts
- `tenure_vs_churn.png` — churned customers tend to have much shorter tenure
- `feature_importance.png` — which features the model relies on most
- `confusion_matrix.png` — model performance breakdown

## Project structure

```
customer-churn-prediction/
├── data/
│   └── generate_data.py      # Creates the synthetic dataset
├── src/
│   ├── data_preprocessing.py # Cleaning + encoding
│   ├── eda.py                 # Exploratory analysis + charts
│   ├── train_model.py         # Model training + evaluation
│   └── predict.py             # Predict on sample customers
├── tests/
│   └── test_preprocessing.py # Unit tests
├── outputs/                   # Generated charts, model, metrics (gitignored contents)
├── main.py                    # Runs the full pipeline
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data/generate_data.py
```

### 3. Run the full pipeline

```bash
python main.py
```

This will run EDA, train the model, and print sample predictions — charts
and the trained model are saved to `outputs/`.

### 4. Run the tests

```bash
pytest tests/ -v
```

## Tech stack

- **Python 3**
- **pandas / numpy** — data handling
- **scikit-learn** — machine learning (Random Forest)
- **matplotlib / seaborn** — visualization
- **pytest** — testing

## Possible extensions

Good next steps if you want to build on this further (great talking points
in an interview too):
- Try other models (Logistic Regression, XGBoost) and compare performance
- Add hyperparameter tuning with `GridSearchCV`
- Build a simple Streamlit or Flask app to serve predictions
- Add more features or use a real-world dataset (e.g., the Kaggle Telco
  Customer Churn dataset)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

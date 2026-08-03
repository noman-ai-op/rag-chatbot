# Customer Churn Prediction Project

## Overview
This project predicts whether a telecom customer is likely to churn (cancel their
subscription) based on their account and usage data. It was built as an end-to-end
portfolio project, covering the full machine learning lifecycle from raw data to a
deployed, interactive web application.

## Dataset
The project uses the Telco Customer Churn dataset from Kaggle. It contains customer
information such as tenure (months with the company), monthly charges, total charges,
contract type (month-to-month, one year, two year), internet service type, and whether
the customer has churned.

## Approach
1. **Exploratory Data Analysis (EDA)**: Analyzed churn patterns across different
   customer segments — for example, customers on month-to-month contracts churn far
   more often than customers on long-term contracts.
2. **Data Preprocessing**: Handled missing values, encoded categorical variables,
   and scaled numerical features.
3. **Model Training**: Trained and compared multiple models, including Logistic
   Regression, Random Forest, and others. Logistic Regression was selected as the
   final model, achieving approximately 80% accuracy, with a good balance of
   performance and interpretability.
4. **Deployment**: Built an interactive web app using Streamlit, allowing users to
   input customer details and get a real-time churn prediction.

## Bugs Fixed
During deployment, a column-order mismatch bug was discovered — the input data columns
were not in the same order the model was trained on, causing incorrect predictions.
This was fixed by saving the exact training column order into a `columns.pkl` file and
using it to reorder any new input data before prediction.

## Tech Stack
- Python, Pandas, Scikit-learn for data processing and modeling
- Streamlit for the web application interface
- Pickle for model and column-order serialization
- Hosted on GitHub for version control and portfolio visibility

## Key Result
The final model achieves approximately 80% accuracy in predicting customer churn,
and the deployed app allows non-technical users to get instant predictions by
entering basic customer account details.

## Repository
The full project code is available on GitHub at:
github.com/noman-ai-op/Artificial-intelligence

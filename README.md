# Customer Churn Prediction Using Machine Learning
This project aims to predict customer churn in a telecom dataset using machine learning. The goal is to identify patterns and factors influencing customer retention, helping businesses implement effective strategies to reduce churn rates.

## Features
Dataset: The project uses the "WA_Fn-UseC_-Telco-Customer-Churn.csv dataset", containing customer demographics, services subscribed, and payment details.

### Exploratory Data Analysis (EDA):
Distribution of numerical features  
Correlation heatmap  
Visualization of categorical features 

### Data Preprocessing:
Handling missing data and outliers  
Label encoding for categorical features  
Synthetic Minority Oversampling Technique (SMOTE) to address class imbalance  

### Model Training:
Models implemented: Decision Tree, Random Forest, and XGBoost  
Cross-validation for model comparison  

### Model Evaluation:
Confusion matrix  
Accuracy, precision, recall, and F1-score  

### Predictive System:
Deployment-ready system to predict churn for new data inputs  

## Requirements
Python 3.x  
Required Libraries:  
  

### Bash::
    pandas
    numpy  
    matplotlib  
    seaborn  
    scikit-learn  
    imbalanced-learn  
    xgboost  
    pickle  

## Install dependencies using:

### bash::
    pip install -r requirements.txt


## Workflow
#### 1. Data Loading and Exploration:
   Load the dataset and explore its structure.  
   Handle missing values and data inconsistencies.

#### 2. EDA:
Analyze numerical and categorical features.  
Generate insights to guide feature engineering.

#### 3. Data Preprocessing:
Encode categorical variables.  
Balance the dataset using SMOTE.

#### 4. Model Training and Evaluation:
Train models and evaluate their performance.  
Select the Random Forest model for deployment due to its superior accuracy.

#### 5. Deployment:
Save the trained model and label encoders as pickle files.  
Build a system to predict churn for new inputs.

## File Structure
    Customer-Churn-Prediction/
    │
    ├── data/
    │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
    ├── notebooks/
    │   └── exploratory_analysis.ipynb
    ├── scripts/
    │   ├── churn_prediction.py
    │   ├── predict.py
    ├── models/
    │   ├── customer_churn_model.pkl
    │   ├── encoders.pkl
    ├── README.md
    └── requirements.txt




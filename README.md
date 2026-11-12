# Employee Attrition Prediction

## Project Overview
An AI-powered Machine Learning application that predicts whether an employee is likely to leave a company.

The project uses the IBM HR Analytics Employee Attrition dataset and a Random Forest model trained with SMOTE to address class imbalance.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest
- SMOTE
- Streamlit
- Joblib

## Main Features
- Employee information input form
- Employee attrition prediction
- Estimated probability of leaving
- Quick testing with sample employee profiles
- Interactive Streamlit interface

## Project Structure
```text
Employee-Attrition-Prediction/
│
├── app.py
├── feature_names.txt
├── models/
│   ├── preprocessor.pkl
│   └── random_forest_smote.pkl
└── README.md
```

## Run Locally
Install the required packages:

```bash
pip install streamlit pandas scikit-learn imbalanced-learn joblib
```

Run the application:

```bash
streamlit run app.py
```

## Important Note
This project is for educational and decision-support purposes only. Predictions should not be used as the sole basis for employment decisions.

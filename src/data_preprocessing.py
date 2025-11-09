import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def preprocess_data(df, target_col = None):
    
    print("\n--- Starting Preprocessing ---")
    
    # Drop customerID as identifier
    df = df.drop('customerID', axis=1)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Handle target column
    if target_col and target_col in df.columns:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({'Yes': 1, 'No': 0}).fillna(df[target_col])
        y = df[target_col]
        X = df.drop(target_col, axis=1)
    else:
        print("Warning: Target column not provided or not found in dataframe.")
        X = df.copy()
        y = None

    # Identify feature data types

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    
    # Numeric pipeline: Impute (with median) THEN scale
    # Imputation is required for NaNs in 'TotalCharges'
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute (with 'missing') THEN OHE
    # This handles any potential missing categorical values
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # --- Define ColumnTransformer ---
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Pass through any unlisted columns
    )
    
    print("--- Preprocessor Defined ---")
    
    print("\n--- Train Test Split ---")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, preprocessor


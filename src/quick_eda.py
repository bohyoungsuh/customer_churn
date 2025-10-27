
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

def whats_in_my_data(df, target_col=None):
    """
    High-level EDA: shape, data types, missing values, column types,
    summary statistics, target distribution with counts & percentages,
    and numeric correlation matrix.
    """
    
    # 1. Dataset shape
    display("Dataset shape: " + str(df.shape))
    
    # 2. Column data types
    display("Column data types:")
    display(df.dtypes)
    
    # 3. Missing Value Count
    missing = df.isnull().sum()
    if missing.sum() == 0:
        display("No missing values found.")
    else:
        display("Missing values per column:")
        display(missing)
    
    # 4. Count by Column Data Types
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    
    display(f"Number of numeric columns: {len(numeric_cols)}")
    display(f"Number of categorical columns: {len(cat_cols)}")
    display(f"Number of datetime columns: {len(datetime_cols)}")
    
    # 5. Basic Statistics
    if numeric_cols:
        display("Numeric columns summary:")
        display(df[numeric_cols].describe().T)
    
    if cat_cols:
        display("Categorical columns summary:")
        display(df[cat_cols].describe())
    
    # 6. Target Metric distribution with counts and percentages
    if target_col and target_col in df.columns:
        plt.figure(figsize=(6,4))
        ax = sns.countplot(data=df, x=target_col)
        plt.title(f"{target_col} distribution")
        total = len(df)
        # Add count and percentage labels on bars
        for p in ax.patches:
            count = int(p.get_height())
            pct = count / total * 100
            ax.annotate(f'{count} ({pct:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom')
        plt.show()
    elif target_col:
        print(f"Warning: target column '{target_col}' not found in dataset.")
    
    # 7. Correlation matrix for numeric columns
    if numeric_cols:
        display("Correlation matrix:")
        display(df[numeric_cols].corr())

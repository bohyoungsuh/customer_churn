def whats_in_my_data(df, target_col=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    # 1. Preview first few rows
    print("First 5 rows of the dataset:")
    display(df.head())

    # 2. Shape and column types
    print(f"\nDataset shape: {df.shape}")
    print("\nColumn counts by type:")
    print(df.dtypes.value_counts())

    # 3. View Data Info
    print("\n--- Data Info ---")
    df.info()
    
    # 4. Numerical Summary
    print("\n--- Numerical Columns Summary ---")
    print(df.describe())


    # 5. Missing Values Check
    print(f"\n--- Data Quality Check ---")    
    # Check for standard NaN missing values
    missing_nan = df.isnull().sum()
    missing_nan = missing_nan[missing_nan > 0] # Filter to only columns with missing values
    if missing_nan.empty:
        print("No standard NaN (null) missing values found.")
    else:
        print("\nStandard NaN (null) missing values per column:")
        print(missing_nan)

    # Check for empty spaces ' ' in object columns
    print("\nChecking for empty spaces (' ') in object-type columns...")
    empty_spaces_found = False
    # Iterate only through object columns, as ' ' won't exist in numeric types
    for column in df.select_dtypes(include=['object']).columns:
        # Use .sum() on the boolean series
        empty_spaces_count = (df[column] == ' ').sum()
        if empty_spaces_count > 0:
            print(f"Column '{column}': Found {empty_spaces_count} rows with empty spaces.")
            empty_spaces_found = True
            
    if not empty_spaces_found:
        print("No empty spaces found in object columns.")

    # 6. Duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        print(f"\n⚠️ Found {num_duplicates} duplicate rows.")
    else:
        print("\nNo duplicate rows found.")

    # Target distribution
    if target_col and target_col in df.columns:
        plt.figure(figsize=(6,4))
        ax = sns.countplot(x=target_col, data=df)
        plt.title(f"Distribution of Target: {target_col}")  # <-- Added title
        total = len(df)
        for p in ax.patches:
            count = int(p.get_height())
            percentage = count / total * 100
            ax.annotate(f'{count} ({percentage:.1f}%)', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom')
        plt.show()

    # 7. Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        plt.figure(figsize=(10,8))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

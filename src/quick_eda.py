def whats_in_my_data(df, target_col=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display

    # Preview first few rows
    print("First 5 rows of the dataset:")
    display(df.head())

    # Shape and column types
    print(f"\nDataset shape: {df.shape}")
    print("\nColumn counts by type:")
    print(df.dtypes.value_counts())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\nNo missing values found.")
    else:
        print("\nMissing values per column:")
        print(missing)

    # Duplicates
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

    # Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include='number')
    if not numeric_cols.empty:
        plt.figure(figsize=(10,8))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.show()

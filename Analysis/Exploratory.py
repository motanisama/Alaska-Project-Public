# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_raw_data():
    """Load the raw Alaska data from Excel file"""
    data_path = Path("/Users/matthewotani/langchain/Analysis/RawData/AlaskaDataRaw.xlsx")
    
    try:
        # Load all sheets to see what's available
        excel_file = pd.ExcelFile(data_path)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # Load the main data sheet (assuming first sheet or looking for data sheet)
        if 'Data' in excel_file.sheet_names:
            df = pd.read_excel(data_path, sheet_name='Data')
        else:
            df = pd.read_excel(data_path, sheet_name=0)  # First sheet
            
        print(f"\nLoaded data from: {data_path}")
        print(f"Data shape: {df.shape}")
        return df, excel_file
    
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def basic_data_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*50)
    print("BASIC DATA INFORMATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMemory usage:")
    print(df.memory_usage(deep=True))
    
    print("\nBasic statistics:")
    print(df.describe(include='all'))

def missing_data_analysis(df):
    """Analyze missing data patterns"""
    print("\n" + "="*50)
    print("MISSING DATA ANALYSIS")
    print("="*50)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percent
    }).sort_values('Missing_Count', ascending=False)
    
    print("Missing data summary:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Visualize missing data
    if missing_data.sum() > 0:
        plt.figure(figsize=(12, 6))
        missing_data[missing_data > 0].plot(kind='bar')
        plt.title('Missing Data by Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def explore_categorical_data(df):
    """Explore categorical columns"""
    print("\n" + "="*50)
    print("CATEGORICAL DATA EXPLORATION")
    print("="*50)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Value counts:")
        print(df[col].value_counts().head(10))
        
        # Plot if reasonable number of categories
        if df[col].nunique() <= 20:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().head(15).plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def explore_numerical_data(df):
    """Explore numerical columns"""
    print("\n" + "="*50)
    print("NUMERICAL DATA EXPLORATION")
    print("="*50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        print("Numerical columns statistics:")
        print(df[numerical_cols].describe())
        
        # Distribution plots
        n_cols = min(len(numerical_cols), 4)
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        if len(numerical_cols) > 0:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        # Correlation matrix if multiple numerical columns
        if len(numerical_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Numerical Variables')
            plt.tight_layout()
            plt.show()

def explore_date_columns(df):
    """Explore date/datetime columns"""
    print("\n" + "="*50)
    print("DATE/TIME DATA EXPLORATION")
    print("="*50)
    
    # Look for date columns
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
    
    if date_cols:
        for col in date_cols:
            print(f"\n{col}:")
            print(f"  Date range: {df[col].min()} to {df[col].max()}")
            print(f"  Number of unique dates: {df[col].nunique()}")
    else:
        print("No obvious date columns found. Checking for columns that might contain dates...")
        for col in df.columns:
            sample_values = df[col].dropna().head().astype(str)
            if any(len(str(val)) > 8 and ('/' in str(val) or '-' in str(val)) for val in sample_values):
                print(f"  {col} might contain dates: {sample_values.tolist()}")

def sample_data_preview(df):
    """Show sample data"""
    print("\n" + "="*50)
    print("SAMPLE DATA PREVIEW")
    print("="*50)
    
    print("First 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    print("\nRandom sample (5 rows):")
    print(df.sample(min(5, len(df))))

def main():
    """Main exploration function"""
    print("ALASKA AIR DATA EXPLORATION")
    print("="*60)
    
    # Load data
    df, excel_file = load_raw_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Explore all sheets if multiple exist
    if excel_file and len(excel_file.sheet_names) > 1:
        print(f"\nExploring all sheets: {excel_file.sheet_names}")
        for sheet_name in excel_file.sheet_names:
            print(f"\n{'='*20} SHEET: {sheet_name} {'='*20}")
            sheet_df = pd.read_excel("Analysis/RawData/AlaskaDataRaw.xlsx", sheet_name=sheet_name)
            print(f"Shape: {sheet_df.shape}")
            print(f"Columns: {sheet_df.columns.tolist()}")
            if len(sheet_df) > 0:
                print(sheet_df.head(2))
    
    # Run all exploration functions
    sample_data_preview(df)
    basic_data_info(df)
    missing_data_analysis(df)
    explore_categorical_data(df)
    explore_numerical_data(df)
    explore_date_columns(df)
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE")
    print("="*60)
    
    return df

if __name__ == "__main__":
    df = main()

# %%

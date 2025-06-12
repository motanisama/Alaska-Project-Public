# %% [markdown]
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file into a DataFrame
file_path = '/Users/matthewotani/langchain/Analysis/RawData/AlaskaDataRaw.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows of the DataFrame
df.head()

# Display data types of the DataFrame columns
print("\nData Types:")
print("="*50)
print(df.dtypes)

# Check for missing values in the AIRCRAFT column
print("\nMissing Values in AIRCRAFT Column:")
print("="*50)
print(df['AIRCRAFT'].isnull().sum())

# Inspect unique values in the AIRCRAFT column
print("\nUnique Values in AIRCRAFT Column:")
print("="*50)
print(df['AIRCRAFT'].unique())

# Step 1: Calculate passengers per mile if not already done
#“How much fuel do I burn, on average, to fly one mile on one flight?”

df['PASSENGERS_PER_MILE'] = df['PASSENGERS'] / df['MILES_DISTANCE'] * df['FLIGHTS']

# Step 2: Prepare data and drop missing or invalid rows
df_scatter_fuel_vs_paxmile = df[['FUEL_COST', 'PASSENGERS_PER_MILE', 'AIRCRAFT']].dropna()

# Step 3: Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PASSENGERS_PER_MILE',
    y='FUEL_COST',
    hue='AIRCRAFT',
    data=df_scatter_fuel_vs_paxmile,
    alpha=0.7
)

# Step 4: Styling
plt.title('Fuel Cost per Mile per Flight (Color = Aircraft Type)', fontsize=14)
plt.xlabel('Passengers per Mile per Flight')
plt.ylabel('Fuel Cost ($)')
plt.grid(True)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Step 5: Show the plot
plt.show()

# %%

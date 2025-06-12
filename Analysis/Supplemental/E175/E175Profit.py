# %%
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
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

revenue_columns = [
    'FIRST_CLASS_REVENUE',
    'PREM_ECONOMY_REVENUE', 
    'ECONOMY_REVENUE',
    'BAGGAGE_REVENUE',
    'OTHER_ANCILLARY_REVENUE',
    'FREIGHT_REVENUE'
]

cost_columns = [
    'FUEL_COST',
    'AIRPORT_FEE_COST',
    'AIRCRAFT_OPERATION_COST',
    'CREW_AND_LABOR_COST'
]

# Calculate total revenue and total cost
df['TOTAL_REVENUE'] = df[revenue_columns].sum(axis=1)
df['TOTAL_COST'] = df[cost_columns].sum(axis=1)

# Calculate profit
df['PROFIT'] = df['TOTAL_REVENUE'] - df['TOTAL_COST']

# Convert AIRCRAFT column to string type
df['AIRCRAFT'] = df['AIRCRAFT'].astype(str)


# Create dummy variables for aircraft type
aircraft_dummies = pd.get_dummies(df['AIRCRAFT'], prefix='AIRCRAFT')

# Prepare X (independent variables) and y (dependent variable)
X = aircraft_dummies
y = df['PROFIT']

# Add constant to X for the intercept term
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y.astype(float), X.astype(float)).fit()

# Print the regression summary
print("\nRegression Results:")
print("="*50)
print(model.summary())

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot actual vs predicted values
plt.scatter(y, model.predict(X), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)

# Add labels and title
plt.xlabel('Actual Profit ($)')
plt.ylabel('Predicted Profit ($)')
plt.title('Actual vs Predicted Profit by Aircraft Type')

# Add grid
plt.grid(True, alpha=0.3)

# Format axis labels with comma separator for thousands
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()




# %%

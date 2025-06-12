# %% [markdown]
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
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

# Calculate load factor (percentage of seats filled)
df['LOAD_FACTOR'] = df['PASSENGERS'] / df['SEATS']

# Create scatter plot of Profit vs ASM
plt.figure(figsize=(10, 6))
plt.scatter(df['ASMS'], df['PROFIT'], alpha=0.5)

# Add trend line
z = np.polyfit(df['ASMS'], df['PROFIT'], 1)
p = np.poly1d(z)
plt.plot(df['ASMS'], p(df['ASMS']), "r--", alpha=0.8)

# Add labels and title
plt.xlabel('Available Seat Miles (ASMS)')
plt.ylabel('Profit ($)')
plt.title('Profit vs Available Seat Miles')

# Add grid
plt.grid(True, alpha=0.3)

# Format axes with comma separator for large numbers
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Run regression analysis
X = df['ASMS']
y = df['PROFIT']
X = sm.add_constant(X)  # Add constant term for intercept

# Fit the model
model = sm.OLS(y, X).fit()

# Print regression summary
print("\nRegression Analysis Results:")
print("="*50)
print(model.summary())  # Print coefficient table

# Calculate Mean Absolute Error (MAE)
predictions = model.predict(X)
mae = np.mean(np.abs(y - predictions))
print("\nMean Absolute Error:")
print("="*50)
print(f"${mae:,.2f}")

# Calculate Mean Squared Error (MSE)
mse = np.mean((y - predictions) ** 2)
print("\nMean Squared Error:")
print("="*50)
print(f"${mse:,.2f}")


# Add R-squared to plot
plt.text(0.02, 0.95, f'R² = {model.rsquared:.3f}', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

# Calculate correlation coefficient (r)
r = np.sqrt(model.rsquared)
print("\nCorrelation Coefficient (r):")
print("="*50)
print(f"{r:.3f}")

# Print R-squared value
print("\nR-squared Value:")
print("="*50)
print(f"{model.rsquared:.3f}")

# Conduct t-test analysis
t_stat = model.tvalues[1]  # t-statistic for ASMS coefficient
p_value = model.pvalues[1]  # p-value for ASMS coefficient

print("\nt-Test Results:")
print("="*50)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Add interpretation
if p_value < 0.05:
    print("\nInterpretation: The relationship between ASMS and Profit is statistically significant (p < 0.05)")
else:
    print("\nInterpretation: The relationship between ASMS and Profit is not statistically significant (p >= 0.05)")



# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()

#%%
"""
About 22.6 variance in profit is explained by ASMS. Which is pretty good for a single valariable
R tells us that there is a positive correlation between ASMS and profit.

If we used this model to predict profit:
our profit prediction would be off by $308,520,303,143.84 on average using MSE
- sensitive to outliers

our profit prediction would be off by $304,971.29 on average using MAE

"""






# %%

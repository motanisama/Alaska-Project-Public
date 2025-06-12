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

# Create origin-destination column by combining ORIGIN and DESTINATION
df['ROUTE'] = df['ORIGIN'] + '-' + df['DESTINATION']

df


# Filter dataframe to only include E175 aircraft
df_e175 = df[df['AIRCRAFT'] == 'E175']

# Get unique routes for E175
e175_routes = df_e175[['ROUTE']].drop_duplicates()

# Create a dataframe with only routes flown by E175
df_e175_routes = df[df['ROUTE'].isin(e175_routes['ROUTE'])]

# Calculate profit per passenger mile per flight
df_e175_routes['PROFIT_PER_PAX_MILE_FLIGHT'] = df_e175_routes['PROFIT'] / (df_e175_routes['PASSENGERS'] * df_e175_routes['MILES_DISTANCE'] * df_e175_routes['FLIGHTS'])

# Calculate average profit per pax mile flight by route and aircraft 
route_profit = df_e175_routes.groupby(['ROUTE', 'AIRCRAFT'])['PROFIT_PER_PAX_MILE_FLIGHT'].mean().reset_index()

#calculate average profit per pax mile flight by  
route_profit

# Create bar plot
plt.figure(figsize=(15, 8))
sns.barplot(
    data=route_profit,
    x='ROUTE',
    y='PROFIT_PER_PAX_MILE_FLIGHT',
    hue='AIRCRAFT'
)

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.title('Average Profit by Route and Aircraft Type')
plt.xlabel('Route')
plt.ylabel('Average Profit ($)')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
# Create bar plot comparing E175 vs other aircraft
plt.figure(figsize=(15, 8))

# Create a new column to group aircraft into 'E175' and 'Other'
route_profit['Aircraft_Group'] = route_profit['AIRCRAFT'].apply(lambda x: 'E175' if x == 'E175' else 'Other Aircraft')

sns.barplot(
    data=route_profit,
    x='ROUTE',
    y='PROFIT_PER_PAX_MILE_FLIGHT',
    hue='Aircraft_Group',
    palette={'E175': 'skyblue', 'Other Aircraft': 'lightgray'}
)

# Customize the plot
plt.xticks(rotation=45, ha='right')
plt.title('Average Profit by Route: E175 vs Other Aircraft')
plt.xlabel('Route')
plt.ylabel('Average Profit ($)')
plt.legend(title='Aircraft Type')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()




# %% [markdown]
""" 
### KEY FINDINGS
#### E175 Routes Analysis

#### On many routes (like BOI-SEA, SEA-BOI, SFO-GEG), E175 bars are negative or low, suggesting that:
- It's not generating enough revenue to cover its cost on those routes.
- It may be mismatched for those route profiles (too small, inefficient, or underloaded).

##### E175 should be reconsidered for several low-margin or negative-margin routes.
- 737-800 is a strong performer on most routes — optimize its deployment.
- consider:
    - Retiring or reassigning E175 from certain routes.
    - Replacing it with more efficient or larger aircraft.
    - Marketing/promotion efforts if E175 is needed but underutilized.
"""


# %%

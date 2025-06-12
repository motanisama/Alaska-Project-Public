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


# Create a route column by combining ORIGIN and DESTINATION
df['ROUTE'] = df['ORIGIN'] + '-' + df['DESTINATION']

# Convert DATE to datetime and extract year
df['YEAR'] = pd.to_datetime(df['MONTH']).dt.year

# Calculate fuel costs by year
fuel_by_year = df.groupby('YEAR')['FUEL_COST'].sum()

# Calculate year-over-year change
fuel_change = fuel_by_year[2023] - fuel_by_year[2022]
fuel_pct_change = (fuel_change / fuel_by_year[2022]) * 100

print("\nFuel Cost Analysis:")
print(f"2022 Total Fuel Cost: ${fuel_by_year[2022]:,.2f}")
print(f"2023 Total Fuel Cost: ${fuel_by_year[2023]:,.2f}")
print(f"Change in Fuel Cost: ${fuel_change:,.2f}")
print(f"Percent Change: {fuel_pct_change:.1f}%")
# Calculate total flights by year
flights_by_year = df.groupby('YEAR')['FLIGHTS'].sum()

# Calculate average fuel cost per flight
fuel_per_flight_2022 = fuel_by_year[2022] / flights_by_year[2022]
fuel_per_flight_2023 = fuel_by_year[2023] / flights_by_year[2023]

# Calculate change in fuel cost per flight
fuel_per_flight_change = fuel_per_flight_2023 - fuel_per_flight_2022
fuel_per_flight_pct_change = (fuel_per_flight_change / fuel_per_flight_2022) * 100

print("\nFuel Cost per Flight Analysis:")
print(f"2022 Fuel Cost per Flight: ${fuel_per_flight_2022:,.2f}")
print(f"2023 Fuel Cost per Flight: ${fuel_per_flight_2023:,.2f}") 
print(f"Change in Fuel Cost per Flight: ${fuel_per_flight_change:,.2f}")
print(f"Percent Change: {fuel_per_flight_pct_change:.1f}%")

# Calculate fuel cost per flight-mile for the entire DataFrame
df['FUEL_COST_PER_FLIGHT_MILE'] = df['FUEL_COST'] / (df['FLIGHTS'] * df['MILES_DISTANCE'])

# Calculate fuel cost per seat mile for the entire DataFrame
df['FUEL_COST_PER_SEAT_MILE'] = df['FUEL_COST'] / df['ASMS']

# Calculate average fuel cost per mile for each route
route_fuel_costs = df.groupby(['ROUTE', 'YEAR'])['FUEL_COST_PER_FLIGHT_MILE'].mean().unstack()

# Create a figure for the bar chart
plt.figure(figsize=(15, 8))

# Get routes sorted by 2023 values for better visualization
routes_sorted = route_fuel_costs[2023].sort_values(ascending=False).index

# Plot the bars
x = np.arange(len(routes_sorted))
width = 0.35

# Use distinct colors for years
plt.bar(x - width/2, route_fuel_costs.loc[routes_sorted, 2022], width, label='2022', color='#2E86C1')  # Dark blue for 2022
plt.bar(x + width/2, route_fuel_costs.loc[routes_sorted, 2023], width, label='2023', color='#E74C3C')  # Red for 2023

# Customize the plot
plt.xlabel('Routes')
plt.ylabel('Average Fuel Cost per Flight-Mile ($)')
plt.title('Average Fuel Cost per Flight-Mile by Route: 2022 vs 2023')
plt.xticks(x, routes_sorted, rotation=45, ha='right')
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Calculate average fuel cost per seat mile for each route
route_fuel_costs_per_seat = df.groupby(['AIRCRAFT', 'YEAR'])['FUEL_COST_PER_SEAT_MILE'].mean().unstack()

# Calculate percentage change for each route
route_fuel_costs_per_seat['PERCENT_CHANGE'] = ((route_fuel_costs_per_seat[2023] - route_fuel_costs_per_seat[2022]) / route_fuel_costs_per_seat[2022]) * 100

# Create a figure for the bar chart
plt.figure(figsize=(15, 8))

# Get routes sorted by 2023 values for better visualization
routes_sorted = route_fuel_costs_per_seat[2023].sort_values(ascending=False).index

# Plot the bars
x = np.arange(len(routes_sorted))
width = 0.35

# Use distinct colors for years
plt.bar(x - width/2, route_fuel_costs_per_seat.loc[routes_sorted, 2022], width, label='2022', color='#2E86C1')  # Dark blue for 2022
plt.bar(x + width/2, route_fuel_costs_per_seat.loc[routes_sorted, 2023], width, label='2023', color='#E74C3C')  # Red for 2023

# Annotate percentage change
for i, route in enumerate(routes_sorted):
    pct_change = route_fuel_costs_per_seat.loc[route, 'PERCENT_CHANGE']
    plt.text(i, route_fuel_costs_per_seat.loc[route, 2023] + 0.01, f'{pct_change:.1f}%', ha='center', va='bottom', fontsize=8, color='black')

# Customize the plot
plt.xlabel('Routes')
plt.ylabel('Average Fuel Cost per Available Seat Mile ($)')
plt.title('Average Fuel Cost per ASM by Route: 2022 vs 2023')
plt.xticks(x, routes_sorted, rotation=45, ha='right')
plt.legend()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Run regression analysis of fuel cost per mile vs profit
X = df['FUEL_COST_PER_FLIGHT_MILE']
y = df['TOTAL_REVENUE']

# Add constant to predictor 
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print("\nRegression Analysis: Fuel Cost per Flight-Mile vs Profit")
print("="*50)
print(model.summary())
# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['FUEL_COST_PER_FLIGHT_MILE'], df['PROFIT'], alpha=0.5)
plt.plot(df['FUEL_COST_PER_FLIGHT_MILE'], model.predict(X), color='red', linewidth=2)

plt.xlabel('Fuel Cost per Flight-Mile ($)')
plt.ylabel('Profit ($)') 
plt.title('Relationship between Fuel Cost per Flight-Mile and Profit')

# Add R-squared value to plot
plt.text(0.05, 0.95, f'R² = {model.rsquared:.3f}', 
         transform=plt.gca().transAxes,
         verticalalignment='top')

plt.tight_layout()
plt.show()


# Run regression analysis of total revenue vs fuel cost
X = df['FUEL_COST']
y = df['TOTAL_REVENUE']

# Add constant to predictor
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print regression summary
print("\nRegression Analysis: Fuel Cost vs Total Revenue")
print("="*50)
print(model.summary())

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(df['FUEL_COST'], df['TOTAL_REVENUE'], alpha=0.5)
plt.plot(df['FUEL_COST'], model.predict(X), color='red', linewidth=2)

plt.xlabel('Fuel Cost ($)')
plt.ylabel('Total Revenue ($)')
plt.title('Relationship between Fuel Cost and Total Revenue')

# Add R-squared value to plot
plt.text(0.05, 0.95, f'R² = {model.rsquared:.3f}',
         transform=plt.gca().transAxes,
         verticalalignment='top')

plt.tight_layout()
plt.show()








#%% [markdown]
"""
Use PER_FLIGHT_MILE if you're comparing routes or operations, regardless of aircraft size.
Use PER_SEAT_MILE to assess fuel efficiency in context of capacity—great for evaluating aircraft performance or load strategies.

30% increase in fuel cost normalized by dividing by flights
~2% has to do with routes (some routes are longer)
28% increase in fuel cost per flight-mile
this shows us that ther other 10% is due to incerased flights
"""







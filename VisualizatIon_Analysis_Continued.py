# %% [markdown]
"""
# Excel Data
Quick test to load and explore the CA_CANDIDATE_PROJECT_DATA.xlsx file.
"""

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load Excel data
def find_excel_file():
    """Find the Excel file in possible locations."""
    possible_locations = [
        "CA_CANDIDATE_PROJECT_DATA.xlsx",
        "../CA_CANDIDATE_PROJECT_DATA.xlsx", 
        "../../CA_CANDIDATE_PROJECT_DATA.xlsx",
        Path.home() / "langchain" / "CA_CANDIDATE_PROJECT_DATA.xlsx",
    ]
    
    for location in possible_locations:
        if Path(location).exists():
            return str(location)
    return None

excel_location = find_excel_file()

if excel_location:
    EXCEL_FILE = excel_location
else:
    EXCEL_FILE = "CA_CANDIDATE_PROJECT_DATA.xlsx"

try:
    df = pd.read_excel(EXCEL_FILE)
    print(df.shape)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()
    
    # Convert MONTH to datetime and add YEAR
    df['MONTH'] = pd.to_datetime(df['MONTH'])
    df['YEAR'] = df['MONTH'].dt.year    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    raise

# %%
# Key statistics
print("\n=== KEY STATISTICS ===")

# Date range
if 'MONTH' in df.columns:
    print(f"Date range: {df['MONTH'].min()} to {df['MONTH'].max()}")

# View Unique values in key columns
key_columns = ['ORIGIN', 'DESTINATION', 'AIRCRAFT', 'YEAR']
for col in key_columns:
    if col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= 20:  # Show values if not too many
            print(f"  Values: {sorted(df[col].unique())}")

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


# %% [markdown]
"""
## Cost Analysis: Double Bar Chart by Year
Create a double bar chart showing cost categories on x-axis and cost amounts on y-axis, separated by year.
"""

# %%
# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking charts
plt.style.use('default')
sns.set_palette("husl")

cost_columns = [
    'FUEL_COST',
    'AIRPORT_FEE_COST',
    'AIRCRAFT_OPERATION_COST',
    'CREW_AND_LABOR_COST'
]
#Labels for chart
cost_labels = {
    'FUEL_COST': 'Fuel',
    'AIRPORT_FEE_COST': 'Airport Fees',
    'AIRCRAFT_OPERATION_COST': 'Aircraft Operations',
    'CREW_AND_LABOR_COST': 'Crew & Labor'
}

# Calculate total costs by category and year grouping the data by year
cost_by_year = df.groupby('YEAR')[cost_columns].sum()

# Create double bar chart with percent change
fig, ax = plt.subplots(figsize=(14, 10))

# Get years and prepare data
years = cost_by_year.index.tolist()
x_labels = [cost_labels[col] for col in cost_columns]
x_pos = np.arange(len(cost_columns))

# Set bar width
bar_width = 0.35

# Create bars for each year
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
bars_by_year = []

for i, year in enumerate(years):
    year_costs = [cost_by_year.loc[year, col] / 1_000_000 for col in cost_columns]  # Convert to millions
    bars = ax.bar(x_pos + i * bar_width, year_costs, bar_width, 
                  label=f'{year}', color=colors[i % len(colors)], alpha=0.8)
    bars_by_year.append(bars)
    
    # Add value labels on bars
    for bar, cost in zip(bars, year_costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'${cost:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add percent change annotations between years
if len(years) > 1:
    for j, col in enumerate(cost_columns):
        # Calculate percent change from first to last year
        start_cost = cost_by_year.loc[years[0], col]
        end_cost = cost_by_year.loc[years[-1], col]
        change_pct = ((end_cost - start_cost) / start_cost) * 100
        
        # Position for percent change label (between the bars)
        x_center = x_pos[j] + bar_width * (len(years) - 1) / 2
        max_height = max([cost_by_year.loc[year, col] / 1_000_000 for year in years])
        y_position = max_height + 2
        
        # Color code the percent change
        change_color = '#00A86B' if change_pct > 0 else '#DC143C' if change_pct < 0 else '#666666'
        change_symbol = '▲' if change_pct > 0 else '▼' if change_pct < 0 else '●'
        
        # Add percent change annotation
        ax.text(x_center, y_position, f'{change_symbol} {change_pct:+.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                color=change_color, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor=change_color, alpha=0.8))

# Chart Settings
ax.set_xlabel('Cost Categories', fontweight='bold', fontsize=12)
ax.set_ylabel('Cost ($ Millions)', fontweight='bold', fontsize=12)
ax.set_title('Cost Analysis by Category and Year (with % Change)', fontweight='bold', fontsize=14)

# Set x-axis labels
ax.set_xticks(x_pos + bar_width * (len(years) - 1) / 2)
ax.set_xticklabels(x_labels)

# Legend Settings
ax.legend(title='Year', title_fontsize=12, fontsize=11)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust y-axis to accommodate percent change labels
y_max = ax.get_ylim()[1]
ax.set_ylim(0, y_max * 1.15)

# Add a text box explaining the symbols
legend_text = "▲ Increase  ▼ Decrease  ● No Change"
ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# %% [markdown]
"""
## Revenue Analysis: Double Bar Chart by Year
Create a double bar chart showing revenue categories on x-axis and revenue amounts on y-axis, separated by year.
"""
# Prepare revenue data by year
revenue_columns = [
    'FIRST_CLASS_REVENUE',
    'PREM_ECONOMY_REVENUE', 
    'ECONOMY_REVENUE',
    'BAGGAGE_REVENUE',
    'OTHER_ANCILLARY_REVENUE',
    'FREIGHT_REVENUE'
]

# Create readable labels for revenue categories
revenue_labels = {
    'FIRST_CLASS_REVENUE': 'First Class',
    'PREM_ECONOMY_REVENUE': 'Premium Economy',
    'ECONOMY_REVENUE': 'Economy',
    'BAGGAGE_REVENUE': 'Baggage',
    'OTHER_ANCILLARY_REVENUE': 'Other Ancillary',
    'FREIGHT_REVENUE': 'Freight'
}

# Calculate total revenue by category and year
revenue_by_year = df.groupby('YEAR')[revenue_columns].sum()

# Create double bar chart for revenue with percent change
fig, ax = plt.subplots(figsize=(16, 10))

# Get years and prepare data
years = revenue_by_year.index.tolist()
x_labels = [revenue_labels[col] for col in revenue_columns]
x_pos = np.arange(len(revenue_columns))

# Set bar width (narrower since we have more categories)
bar_width = 0.3

# Create bars for each year
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
bars_by_year = []

for i, year in enumerate(years):
    year_revenues = [revenue_by_year.loc[year, col] / 1_000_000 for col in revenue_columns]  # Convert to millions
    bars = ax.bar(x_pos + i * bar_width, year_revenues, bar_width, 
                  label=f'{year}', color=colors[i % len(colors)], alpha=0.8)
    bars_by_year.append(bars)
    
    # Add value labels on bars
    for bar, revenue in zip(bars, year_revenues):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'${revenue:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=8)

# Add percent change annotations between years
if len(years) > 1:
    for j, col in enumerate(revenue_columns):
        # Calculate percent change from first to last year
        start_revenue = revenue_by_year.loc[years[0], col]
        end_revenue = revenue_by_year.loc[years[-1], col]
        change_pct = ((end_revenue - start_revenue) / start_revenue) * 100
        
        # Position for percent change label (between the bars)
        x_center = x_pos[j] + bar_width * (len(years) - 1) / 2
        max_height = max([revenue_by_year.loc[year, col] / 1_000_000 for year in years])
        y_position = max_height + 2
        
        # Color code the percent change
        change_color = '#00A86B' if change_pct > 0 else '#DC143C' if change_pct < 0 else '#666666'
        change_symbol = '▲' if change_pct > 0 else '▼' if change_pct < 0 else '●'
        
        # Add percent change annotation
        ax.text(x_center, y_position, f'{change_symbol} {change_pct:+.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9,
                color=change_color, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor=change_color, alpha=0.8))

# Chart Settings
ax.set_xlabel('Revenue Categories', fontweight='bold', fontsize=12)
ax.set_ylabel('Revenue ($ Millions)', fontweight='bold', fontsize=12)
ax.set_title('Revenue Analysis by Category and Year (with % Change)', fontweight='bold', fontsize=14)

# Set x-axis labels with rotation for better readability
ax.set_xticks(x_pos + bar_width * (len(years) - 1) / 2)
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Legend Settings
ax.legend(title='Year', title_fontsize=12, fontsize=11)

# Grid Settings
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust y-axis to accommodate percent change labels
y_max = ax.get_ylim()[1]
ax.set_ylim(0, y_max * 1.15)

# Add a text box explaining the symbols
legend_text = "▲ Increase  ▼ Decrease  ● No Change"
ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Adjust layout
plt.tight_layout()
plt.show()

# %%
# Print detailed revenue breakdown by year

for year in years:
    year_total = revenue_by_year.loc[year, revenue_columns].sum()
    print(f"   Total Revenue: ${year_total:,.0f}")
    
    for col in revenue_columns:
        revenue_amount = revenue_by_year.loc[year, col]
        revenue_pct = (revenue_amount / year_total) * 100
        print(f"   • {revenue_labels[col]}: ${revenue_amount:,.0f} ({revenue_pct:.1f}%)")

# %%
# Year over year
if len(years) > 1:
    print(f"\n YEAR-OVER-YEAR REVENUE CHANGES ({years[0]} to {years[-1]})")
    
    biggest_increase = {'category': '', 'pct': -float('inf'), 'abs': 0}
    biggest_decrease = {'category': '', 'pct': float('inf'), 'abs': 0}
    
    for col in revenue_columns:
        start_revenue = revenue_by_year.loc[years[0], col]
        end_revenue = revenue_by_year.loc[years[-1], col]
        change_abs = end_revenue - start_revenue
        change_pct = (change_abs / start_revenue) * 100
        
        print(f"{revenue_labels[col]}: {change_pct:+.1f}% (${change_abs:+,.0f})")
        
        # Track biggest changes
        if change_pct > biggest_increase['pct']:
            biggest_increase = {'category': revenue_labels[col], 'pct': change_pct, 'abs': change_abs}
        if change_pct < biggest_decrease['pct']:
            biggest_decrease = {'category': revenue_labels[col], 'pct': change_pct, 'abs': change_abs}
    
    print(f"\n Biggest Winner: {biggest_increase['category']} (+{biggest_increase['pct']:.1f}%, ${biggest_increase['abs']:+,.0f})")
    if biggest_decrease['pct'] < 0:
        print(f"📉 Biggest Decline: {biggest_decrease['category']} ({biggest_decrease['pct']:.1f}%, ${biggest_decrease['abs']:+,.0f})")

# %%
# %% [markdown]
"""
## Aircraft Profit Distribution Analysis
Create charts showing profit distribution for each aircraft type to identify performance patterns and outliers.
"""

# %%
print("=== CREATING AIRCRAFT PROFIT DISTRIBUTION CHARTS ===")

# %%
# Calculate profit by aircraft type
aircraft_profit = df.groupby(['AIRCRAFT', 'MONTH']).agg({
    'PROFIT': 'sum',
    'TOTAL_REVENUE': 'sum',
    'TOTAL_COSTS': 'sum',
    'FLIGHTS': 'sum',
    'PASSENGERS': 'sum'
}).reset_index()

# Convert to millions for better readability
aircraft_profit['PROFIT_M'] = aircraft_profit['PROFIT'] / 1_000_000

# Get unique aircraft types
aircraft_types = sorted(df['AIRCRAFT'].unique())

# %%
# Create box plots for profit distribution by aircraft
fig, ax = plt.subplots(figsize=(16, 10))
fig.suptitle('Aircraft Profit Distribution Analysis', fontsize=16, fontweight='bold')

# Box plot showing distribution
aircraft_profit_data = [aircraft_profit[aircraft_profit['AIRCRAFT'] == aircraft]['PROFIT_M'].values 
                       for aircraft in aircraft_types]

box_plot = ax.boxplot(aircraft_profit_data, labels=aircraft_types, patch_artist=True)

# Color the boxes
colors = plt.cm.Set3(np.linspace(0, 1, len(aircraft_types)))
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Monthly Profit Distribution by Aircraft Type (Box Plot)', fontweight='bold', fontsize=14)
ax.set_ylabel('Monthly Profit ($ Millions)', fontweight='bold')
ax.set_xlabel('Aircraft Type', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# Add mean markers
means = [aircraft_profit[aircraft_profit['AIRCRAFT'] == aircraft]['PROFIT_M'].mean() 
         for aircraft in aircraft_types]
ax.scatter(range(1, len(aircraft_types) + 1), means, marker='D', s=50, color='red', 
           zorder=3, label='Mean')
ax.legend()

plt.tight_layout()
plt.show()

# Calculate overall metrics for ranking
aircraft_summary = aircraft_profit.groupby('AIRCRAFT').agg({
    'PROFIT_M': ['mean', 'median', 'sum', 'std', 'count'],
    'FLIGHTS': 'sum',
    'PASSENGERS': 'sum'
}).round(2)

# Flatten column names
aircraft_summary.columns = ['_'.join(col).strip() for col in aircraft_summary.columns]

# Calculate additional metrics
aircraft_summary['PROFIT_PER_FLIGHT'] = (aircraft_summary['PROFIT_M_sum'] * 1_000_000) / aircraft_summary['FLIGHTS_sum']
aircraft_summary['PROFIT_PER_PASSENGER'] = (aircraft_summary['PROFIT_M_sum'] * 1_000_000) / aircraft_summary['PASSENGERS_sum']

# Sort by average monthly profit
aircraft_ranking = aircraft_summary.sort_values('PROFIT_M_mean', ascending=False)

print(f"{'Rank':<5} {'Aircraft':<12} {'Avg Monthly':<12} {'Total Profit':<12} {'Profit/Flight':<12} {'Profit/Pax':<12}")
print(f"{'':5} {'Type':<12} {'Profit ($M)':<12} {'($M)':<12} {'($)':<12} {'($)':<12}")
print("-" * 80)

for rank, (aircraft, data) in enumerate(aircraft_ranking.iterrows(), 1):
    avg_monthly = data['PROFIT_M_mean']
    total_profit = data['PROFIT_M_sum']
    profit_per_flight = data['PROFIT_PER_FLIGHT']
    profit_per_pax = data['PROFIT_PER_PASSENGER']
    
    print(f"{rank:<5} {aircraft:<12} ${avg_monthly:<11.1f} ${total_profit:<11.1f} "
          f"${profit_per_flight:<11.0f} ${profit_per_pax:<11.0f}")

# Identify best and worst performers with insights
print(f"\n TOP PERFORMER: {aircraft_ranking.index[0]}")
top_aircraft = aircraft_ranking.index[0]
top_data = aircraft_ranking.iloc[0]
print(f"   • Average Monthly Profit: ${top_data['PROFIT_M_mean']:.1f}M")
print(f"   • Total Profit: ${top_data['PROFIT_M_sum']:.1f}M")
print(f"   • Consistency (Std Dev): ${top_data['PROFIT_M_std']:.1f}M")
print(f"   • Profit per Flight: ${top_data['PROFIT_PER_FLIGHT']:,.0f}")

print(f"\n LOWEST PERFORMER: {aircraft_ranking.index[-1]}")
bottom_aircraft = aircraft_ranking.index[-1]
bottom_data = aircraft_ranking.iloc[-1]
print(f"   • Average Monthly Profit: ${bottom_data['PROFIT_M_mean']:.1f}M")
print(f"   • Total Profit: ${bottom_data['PROFIT_M_sum']:.1f}M")
print(f"   • Consistency (Std Dev): ${bottom_data['PROFIT_M_std']:.1f}M")
print(f"   • Profit per Flight: ${bottom_data['PROFIT_PER_FLIGHT']:,.0f}")

# Most and least consistent aircraft
most_consistent = aircraft_ranking.loc[aircraft_ranking['PROFIT_M_std'].idxmin()]
least_consistent = aircraft_ranking.loc[aircraft_ranking['PROFIT_M_std'].idxmax()]

print(f"\n MOST CONSISTENT: {aircraft_ranking['PROFIT_M_std'].idxmin()}")
print(f"   • Standard Deviation: ${most_consistent['PROFIT_M_std']:.1f}M")
print(f"   • Average Profit: ${most_consistent['PROFIT_M_mean']:.1f}M")

print(f"\n MOST VOLATILE: {aircraft_ranking['PROFIT_M_std'].idxmax()}")
print(f"   • Standard Deviation: ${least_consistent['PROFIT_M_std']:.1f}M")
print(f"   • Average Profit: ${least_consistent['PROFIT_M_mean']:.1f}M")

print("\n" + "="*80)

# %% [markdown]
"""
## E175 Profitability Deep Dive Analysis
Investigate the factors contributing to E175's poor profitability performance.
"""


# %%
# Table to analyze E175 performance
if 'E175' in aircraft_ranking.index:
    e175_rank = list(aircraft_ranking.index).index('E175') + 1
    print(f" E175 Performance Ranking: #{e175_rank} out of {len(aircraft_ranking)} aircraft types")
    
    # Get E175 specific data
    e175_data = aircraft_ranking.loc['E175']
    print(f" E175 Average Monthly Profit: ${e175_data['PROFIT_M_mean']:.1f}M")
    print(f" E175 Total Profit: ${e175_data['PROFIT_M_sum']:.1f}M")
    print(f" E175 Profit per Flight: ${e175_data['PROFIT_PER_FLIGHT']:,.0f}")
    print(f" E175 Profit per Passenger: ${e175_data['PROFIT_PER_PASSENGER']:,.0f}")
else:
    print("E175 not found in aircraft ranking")

# Compare E175 to fleet average and best performer
# Fleet averages
fleet_avg_monthly = aircraft_ranking['PROFIT_M_mean'].mean()
fleet_avg_per_flight = aircraft_ranking['PROFIT_PER_FLIGHT'].mean()
fleet_avg_per_pax = aircraft_ranking['PROFIT_PER_PASSENGER'].mean()

# Best performer (top aircraft)
best_aircraft = aircraft_ranking.index[0]
best_data = aircraft_ranking.iloc[0]

if 'E175' in aircraft_ranking.index:
    e175_data = aircraft_ranking.loc['E175']
    
    print(f"{'Metric':<25} {'E175':<15} {'Fleet Avg':<15} {'Best ({})'.format(best_aircraft):<15} {'E175 vs Avg':<15}")
    
    # Monthly profit comparison
    e175_vs_avg_monthly = ((e175_data['PROFIT_M_mean'] - fleet_avg_monthly) / fleet_avg_monthly) * 100
    print(f"{'Monthly Profit ($M)':<25} ${e175_data['PROFIT_M_mean']:<14.1f} ${fleet_avg_monthly:<14.1f} ${best_data['PROFIT_M_mean']:<14.1f} {e175_vs_avg_monthly:<14.1f}%")
    
    # Per flight comparison
    e175_vs_avg_flight = ((e175_data['PROFIT_PER_FLIGHT'] - fleet_avg_per_flight) / fleet_avg_per_flight) * 100
    print(f"{'Profit per Flight ($)':<25} ${e175_data['PROFIT_PER_FLIGHT']:<14.0f} ${fleet_avg_per_flight:<14.0f} ${best_data['PROFIT_PER_FLIGHT']:<14.0f} {e175_vs_avg_flight:<14.1f}%")
    
    # Per passenger comparison
    e175_vs_avg_pax = ((e175_data['PROFIT_PER_PASSENGER'] - fleet_avg_per_pax) / fleet_avg_per_pax) * 100
    print(f"{'Profit per Passenger ($)':<25} ${e175_data['PROFIT_PER_PASSENGER']:<14.0f} ${fleet_avg_per_pax:<14.0f} ${best_data['PROFIT_PER_PASSENGER']:<14.0f} {e175_vs_avg_pax:<14.1f}%")

# %%
# Detailed revenue and cost breakdown for E175
print(f"\nE175 REVENUE AND COST BREAKDOWN")

# Get E175 specific flight data
e175_flights = df[df['AIRCRAFT'] == 'E175'].copy()

if len(e175_flights) > 0:
    # Calculate totals and averages
    e175_totals = e175_flights[revenue_columns + cost_columns + ['FLIGHTS', 'PASSENGERS']].sum()
    e175_per_flight = e175_totals / e175_totals['FLIGHTS']
    
    # Fleet averages for comparison
    fleet_totals = df[revenue_columns + cost_columns + ['FLIGHTS', 'PASSENGERS']].sum()
    fleet_per_flight = fleet_totals / fleet_totals['FLIGHTS']
    
    print("REVENUE BREAKDOWN (Per Flight):")
    print(f"{'Component':<25} {'E175 ($)':<15} {'Fleet Avg ($)':<15} {'Difference':<15}")
    print("-" * 70)
    
    for rev_col in revenue_columns:
        e175_rev = e175_per_flight[rev_col]
        fleet_rev = fleet_per_flight[rev_col]
        diff_pct = ((e175_rev - fleet_rev) / fleet_rev) * 100 if fleet_rev != 0 else 0
        
        rev_name = rev_col.replace('_REVENUE', '').replace('_', ' ').title()
        print(f"{rev_name:<25} ${e175_rev:<14.0f} ${fleet_rev:<14.0f} {diff_pct:<14.1f}%")
    
    # Total revenue
    e175_total_rev = e175_per_flight[revenue_columns].sum()
    fleet_total_rev = fleet_per_flight[revenue_columns].sum()
    rev_diff_pct = ((e175_total_rev - fleet_total_rev) / fleet_total_rev) * 100
    print("-" * 70)
    print(f"{'TOTAL REVENUE':<25} ${e175_total_rev:<14.0f} ${fleet_total_rev:<14.0f} {rev_diff_pct:<14.1f}%")
    
    print(f"\nCOST BREAKDOWN (Per Flight):")
    print(f"{'Component':<25} {'E175 ($)':<15} {'Fleet Avg ($)':<15} {'Difference':<15}")
    print("-" * 70)
    
    for cost_col in cost_columns:
        e175_cost = e175_per_flight[cost_col]
        fleet_cost = fleet_per_flight[cost_col]
        diff_pct = ((e175_cost - fleet_cost) / fleet_cost) * 100 if fleet_cost != 0 else 0
        
        cost_name = cost_col.replace('_COST', '').replace('_', ' ').title()
        print(f"{cost_name:<25} ${e175_cost:<14.0f} ${fleet_cost:<14.0f} {diff_pct:<14.1f}%")
    
    # Total costs
    e175_total_cost = e175_per_flight[cost_columns].sum()
    fleet_total_cost = fleet_per_flight[cost_columns].sum()
    cost_diff_pct = ((e175_total_cost - fleet_total_cost) / fleet_total_cost) * 100
    print("-" * 70)
    print(f"{'TOTAL COSTS':<25} ${e175_total_cost:<14.0f} ${fleet_total_cost:<14.0f} {cost_diff_pct:<14.1f}%")
    
    # Profit per flight
    e175_profit_flight = e175_total_rev - e175_total_cost
    fleet_profit_flight = fleet_total_rev - fleet_total_cost
    profit_diff_pct = ((e175_profit_flight - fleet_profit_flight) / fleet_profit_flight) * 100 if fleet_profit_flight != 0 else 0
    print("-" * 70)
    print(f"{'PROFIT PER FLIGHT':<25} ${e175_profit_flight:<14.0f} ${fleet_profit_flight:<14.0f} {profit_diff_pct:<14.1f}%")

# Operational efficiency analysis
print(f"\n E175 OPERATIONAL EFFICIENCY ANALYSIS")
print("="*60)

if len(e175_flights) > 0:
    # Calculate key operational metrics
    e175_avg_passengers = e175_flights['PASSENGERS'].mean()
    e175_load_factor = e175_flights['PASSENGERS'].sum() / (e175_flights['FLIGHTS'].sum() * 76)  # E175 typically seats ~76
    
    fleet_avg_passengers = df['PASSENGERS'].mean()
    
    print(f"Average Passengers per Flight:")
    print(f"  E175: {e175_avg_passengers:.1f}")
    print(f"  Fleet Average: {fleet_avg_passengers:.1f}")
    print(f"  Difference: {e175_avg_passengers - fleet_avg_passengers:+.1f} ({((e175_avg_passengers - fleet_avg_passengers) / fleet_avg_passengers) * 100:+.1f}%)")
    
    print(f"\nEstimated Load Factor:")
    print(f"  E175: {e175_load_factor:.1%}")
    
    # Revenue per passenger
    e175_rev_per_pax = e175_flights[revenue_columns].sum().sum() / e175_flights['PASSENGERS'].sum()
    fleet_rev_per_pax = df[revenue_columns].sum().sum() / df['PASSENGERS'].sum()
    
    print(f"\nRevenue per Passenger:")
    print(f"  E175: ${e175_rev_per_pax:.0f}")
    print(f"  Fleet Average: ${fleet_rev_per_pax:.0f}")
    print(f"  Difference: ${e175_rev_per_pax - fleet_rev_per_pax:+.0f} ({((e175_rev_per_pax - fleet_rev_per_pax) / fleet_rev_per_pax) * 100:+.1f}%)")

# Route analysis for E175
print(f"\n🗺️ E175 ROUTE ANALYSIS")
print("="*60)

if len(e175_flights) > 0:
    # E175 routes and their performance
    e175_routes = e175_flights.groupby(['ORIGIN', 'DESTINATION']).agg({
        'PROFIT': 'sum',
        'TOTAL_REVENUE': 'sum',
        'TOTAL_COSTS': 'sum',
        'FLIGHTS': 'sum',
        'PASSENGERS': 'sum'
    }).reset_index()
    
    e175_routes['ROUTE'] = e175_routes['ORIGIN'] + '-' + e175_routes['DESTINATION']
    e175_routes['PROFIT_PER_FLIGHT'] = e175_routes['PROFIT'] / e175_routes['FLIGHTS']
    e175_routes = e175_routes.sort_values('PROFIT', ascending=False)
    
    print(f"E175 operates on {len(e175_routes)} routes")
    print(f"\nTop 5 Most Profitable E175 Routes:")
    print(f"{'Route':<15} {'Total Profit ($M)':<18} {'Flights':<10} {'Profit/Flight ($)':<18}")
    print("-" * 70)
    
    for i, (_, route) in enumerate(e175_routes.head().iterrows()):
        if i < 5:
            print(f"{route['ROUTE']:<15} ${route['PROFIT']/1_000_000:<17.1f} {route['FLIGHTS']:<10.0f} ${route['PROFIT_PER_FLIGHT']:<17.0f}")
    
    print(f"\nBottom 5 Least Profitable E175 Routes:")
    print(f"{'Route':<15} {'Total Profit ($M)':<18} {'Flights':<10} {'Profit/Flight ($)':<18}")
    print("-" * 70)
    
    for i, (_, route) in enumerate(e175_routes.tail().iterrows()):
        if i < 5:
            print(f"{route['ROUTE']:<15} ${route['PROFIT']/1_000_000:<17.1f} {route['FLIGHTS']:<10.0f} ${route['PROFIT_PER_FLIGHT']:<17.0f}")


# %% [markdown]
"""
## E175 Route Competition Analysis
Analyze if E175 routes are also served by other aircraft types and compare their performance.
"""


# %%
# Check if E175 routes are served by other aircraft
if len(e175_flights) > 0:
    # Get all E175 routes
    e175_route_list = e175_flights[['ORIGIN', 'DESTINATION']].drop_duplicates()
    e175_route_list['ROUTE'] = e175_route_list['ORIGIN'] + '-' + e175_route_list['DESTINATION']
    e175_routes_set = set(e175_route_list['ROUTE'].tolist())
    
    print(f"E175 operates on {len(e175_routes_set)} unique routes")
    
    # Check each E175 route for other aircraft
    route_competition = []
    
    for route in e175_routes_set:
        origin, destination = route.split('-')
        
        # Find all aircraft serving this route
        route_aircraft = df[(df['ORIGIN'] == origin) & (df['DESTINATION'] == destination)]['AIRCRAFT'].unique()
        other_aircraft = [ac for ac in route_aircraft if ac != 'E175']
        
        # Get performance data for this route by aircraft
        route_data = df[(df['ORIGIN'] == origin) & (df['DESTINATION'] == destination)].groupby('AIRCRAFT').agg({
            'PROFIT': 'sum',
            'TOTAL_REVENUE': 'sum',
            'TOTAL_COSTS': 'sum',
            'FLIGHTS': 'sum',
            'PASSENGERS': 'sum'
        }).reset_index()
        
        route_data['PROFIT_PER_FLIGHT'] = route_data['PROFIT'] / route_data['FLIGHTS']
        route_data['REVENUE_PER_FLIGHT'] = route_data['TOTAL_REVENUE'] / route_data['FLIGHTS']
        route_data['COST_PER_FLIGHT'] = route_data['TOTAL_COSTS'] / route_data['FLIGHTS']
        
        competition_info = {
            'route': route,
            'other_aircraft': other_aircraft,
            'total_aircraft': len(route_aircraft),
            'has_competition': len(other_aircraft) > 0,
            'route_data': route_data
        }
        
        route_competition.append(competition_info)
    
    # Analyze competition patterns
    routes_with_competition = [r for r in route_competition if r['has_competition']]
    routes_exclusive = [r for r in route_competition if not r['has_competition']]
    

# %%
# Create bar chart comparing E175 vs other aircraft average profit per flight
print("=== CREATING E175 VS OTHER AIRCRAFT COMPARISON CHART ===")

# Prepare data for comparison chart
if 'E175' in aircraft_ranking.index:
    # Get profit per flight data for all aircraft
    aircraft_profit_comparison = aircraft_ranking[['PROFIT_PER_FLIGHT']].copy()
    aircraft_profit_comparison = aircraft_profit_comparison.sort_values('PROFIT_PER_FLIGHT', ascending=True)
    
    # Create color scheme - highlight E175 in red, others in blue
    colors = ['#DC143C' if aircraft == 'E175' else '#2E86AB' for aircraft in aircraft_profit_comparison.index]
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    bars = ax.barh(range(len(aircraft_profit_comparison)), 
                   aircraft_profit_comparison['PROFIT_PER_FLIGHT'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    ax.set_yticks(range(len(aircraft_profit_comparison)))
    ax.set_yticklabels(aircraft_profit_comparison.index, fontweight='bold')
    ax.set_xlabel('Average Profit per Flight ($)', fontweight='bold', fontsize=12)
    ax.set_title('Aircraft Profitability Comparison: Average Profit per Flight\n(E175 highlighted in red)', 
                 fontweight='bold', fontsize=14)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, aircraft_profit_comparison['PROFIT_PER_FLIGHT'])):
        # Position label at end of bar
        label_x = value + (max(aircraft_profit_comparison['PROFIT_PER_FLIGHT']) * 0.01)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'${value:,.0f}',
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    # Add vertical line for fleet average
    fleet_avg = aircraft_profit_comparison['PROFIT_PER_FLIGHT'].mean()
    ax.axvline(fleet_avg, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Fleet Average: ${fleet_avg:,.0f}')
    
    # Highlight E175 position
    e175_position = list(aircraft_profit_comparison.index).index('E175')
    e175_value = aircraft_profit_comparison.loc['E175', 'PROFIT_PER_FLIGHT']
    
    # Add annotation for E175
    ax.annotate(f'E175: ${e175_value:,.0f}\n(Rank: #{len(aircraft_profit_comparison) - e175_position} of {len(aircraft_profit_comparison)})',
                xy=(e175_value, e175_position), xytext=(e175_value + fleet_avg * 0.3, e175_position + 1),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontweight='bold', fontsize=11)
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right')
    
    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()
    

    
    # Sort by profit per flight (descending for ranking)
    ranking_sorted = aircraft_profit_comparison.sort_values('PROFIT_PER_FLIGHT', ascending=False)
    
    for rank, (aircraft, data) in enumerate(ranking_sorted.iterrows(), 1):
        profit_per_flight = data['PROFIT_PER_FLIGHT']
        vs_fleet_avg = profit_per_flight - fleet_avg
        vs_fleet_pct = (vs_fleet_avg / fleet_avg) * 100
        
        if aircraft == 'E175':
            vs_e175 = 0
            vs_e175_pct = 0
        else:
            vs_e175 = profit_per_flight - e175_value
            vs_e175_pct = (vs_e175 / e175_value) * 100 if e175_value != 0 else 0
        

        
        print(f"{aircraft:<12} ${profit_per_flight:<17,.0f} "
              f"{vs_fleet_pct:+6.1f}% ({vs_fleet_avg:+8,.0f}) "
              f"{vs_e175_pct:+6.1f}% ({vs_e175:+8,.0f})")
    


# %% [markdown]
"""
## Aircraft Fuel Efficiency and Load Factor Analysis
Compare aircraft performance by fuel efficiency and load factor to understand operational efficiency drivers.
"""

# Calculate fuel efficiency and load factor metrics by aircraft
aircraft_efficiency = df.groupby('AIRCRAFT').agg({
    'FUEL_COST': 'sum',
    'FLIGHTS': 'sum',
    'PASSENGERS': 'sum',
    'TOTAL_REVENUE': 'sum',
    'TOTAL_COSTS': 'sum',
    'PROFIT': 'sum'
}).reset_index()

# Calculate efficiency metrics
aircraft_efficiency['FUEL_COST_PER_FLIGHT'] = aircraft_efficiency['FUEL_COST'] / aircraft_efficiency['FLIGHTS']
aircraft_efficiency['PASSENGERS_PER_FLIGHT'] = aircraft_efficiency['PASSENGERS'] / aircraft_efficiency['FLIGHTS']

# Estimate aircraft capacity (approximate based on typical configurations)
aircraft_capacity = {
    '737-700': 126, '737-800': 162, '737-900': 178, '737-Max 9': 178,
    'E175': 76, '757-300': 243, '777-200': 314, 
    'A321': 185, 'A330-200': 247, 'A320': 150
}

# Add capacity and calculate load factor
aircraft_efficiency['ESTIMATED_CAPACITY'] = aircraft_efficiency['AIRCRAFT'].map(aircraft_capacity)
aircraft_efficiency['LOAD_FACTOR'] = aircraft_efficiency['PASSENGERS_PER_FLIGHT'] / aircraft_efficiency['ESTIMATED_CAPACITY']

# Calculate fuel efficiency (passengers per dollar of fuel)
aircraft_efficiency['FUEL_EFFICIENCY'] = aircraft_efficiency['PASSENGERS'] / aircraft_efficiency['FUEL_COST']

print(f" Aircraft efficiency metrics calculated for {len(aircraft_efficiency)} aircraft types")

# %%
# Create dual chart: Fuel Efficiency vs Load Factor
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Aircraft Operational Efficiency Comparison', fontsize=16, fontweight='bold')

# Chart 1: Fuel Efficiency (Passengers per $ of Fuel Cost)
aircraft_efficiency_sorted = aircraft_efficiency.sort_values('FUEL_EFFICIENCY', ascending=True)

# Color scheme - highlight E175 in red
colors1 = ['#DC143C' if aircraft == 'E175' else '#2E86AB' for aircraft in aircraft_efficiency_sorted['AIRCRAFT']]

bars1 = ax1.barh(range(len(aircraft_efficiency_sorted)), 
                 aircraft_efficiency_sorted['FUEL_EFFICIENCY'],
                 color=colors1, alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_yticks(range(len(aircraft_efficiency_sorted)))
ax1.set_yticklabels(aircraft_efficiency_sorted['AIRCRAFT'], fontweight='bold')
ax1.set_xlabel('Fuel Efficiency (Passengers per $ Fuel Cost)', fontweight='bold', fontsize=12)
ax1.set_title('Fuel Efficiency by Aircraft Type\n(Higher = More Efficient)', fontweight='bold', fontsize=14)

# Add value labels
for i, (bar, value) in enumerate(zip(bars1, aircraft_efficiency_sorted['FUEL_EFFICIENCY'])):
    label_x = value + (max(aircraft_efficiency_sorted['FUEL_EFFICIENCY']) * 0.01)
    ax1.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
             va='center', ha='left', fontweight='bold', fontsize=10)

# Add fleet average line
fuel_eff_avg = aircraft_efficiency_sorted['FUEL_EFFICIENCY'].mean()
ax1.axvline(fuel_eff_avg, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Fleet Avg: {fuel_eff_avg:.3f}')
ax1.legend()
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Chart 2: Load Factor
aircraft_lf_sorted = aircraft_efficiency.sort_values('LOAD_FACTOR', ascending=True)

# Color scheme - highlight E175 in red
colors2 = ['#DC143C' if aircraft == 'E175' else '#2E86AB' for aircraft in aircraft_lf_sorted['AIRCRAFT']]

bars2 = ax2.barh(range(len(aircraft_lf_sorted)), 
                 aircraft_lf_sorted['LOAD_FACTOR'] * 100,  # Convert to percentage
                 color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_yticks(range(len(aircraft_lf_sorted)))
ax2.set_yticklabels(aircraft_lf_sorted['AIRCRAFT'], fontweight='bold')
ax2.set_xlabel('Average Load Factor (%)', fontweight='bold', fontsize=12)
ax2.set_title('Load Factor by Aircraft Type\n(Higher = Better Utilization)', fontweight='bold', fontsize=14)

# Add value labels
for i, (bar, value) in enumerate(zip(bars2, aircraft_lf_sorted['LOAD_FACTOR'] * 100)):
    label_x = value + (max(aircraft_lf_sorted['LOAD_FACTOR'] * 100) * 0.01)
    ax2.text(label_x, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
             va='center', ha='left', fontweight='bold', fontsize=10)

# Add fleet average line
lf_avg = aircraft_lf_sorted['LOAD_FACTOR'].mean() * 100
ax2.axvline(lf_avg, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Fleet Avg: {lf_avg:.1f}%')
ax2.legend()
ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Create scatter plot showing relationship between fuel efficiency and load factor
fig, ax = plt.subplots(figsize=(14, 10))

# Create scatter plot
for i, row in aircraft_efficiency.iterrows():
    color = '#DC143C' if row['AIRCRAFT'] == 'E175' else '#2E86AB'
    size = 100 if row['AIRCRAFT'] == 'E175' else 80
    ax.scatter(row['LOAD_FACTOR'] * 100, row['FUEL_EFFICIENCY'], 
               color=color, s=size, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add aircraft labels
    ax.annotate(row['AIRCRAFT'], 
                (row['LOAD_FACTOR'] * 100, row['FUEL_EFFICIENCY']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Load Factor (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Fuel Efficiency (Passengers per $ Fuel Cost)', fontweight='bold', fontsize=12)
ax.set_title('Aircraft Efficiency Matrix: Load Factor vs Fuel Efficiency\n(E175 highlighted in red)', 
             fontweight='bold', fontsize=14)

# Add quadrant lines
ax.axhline(fuel_eff_avg, color='green', linestyle='--', alpha=0.5, label=f'Avg Fuel Efficiency: {fuel_eff_avg:.3f}')
ax.axvline(lf_avg, color='orange', linestyle='--', alpha=0.5, label=f'Avg Load Factor: {lf_avg:.1f}%')

# Add quadrant labels
ax.text(0.95, 0.95, 'High LF\nHigh Fuel Eff', transform=ax.transAxes, 
        ha='right', va='top', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax.text(0.05, 0.95, 'Low LF\nHigh Fuel Eff', transform=ax.transAxes, 
        ha='left', va='top', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.text(0.95, 0.05, 'High LF\nLow Fuel Eff', transform=ax.transAxes, 
        ha='right', va='bottom', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.text(0.05, 0.05, 'Low LF\nLow Fuel Eff', transform=ax.transAxes, 
        ha='left', va='bottom', fontweight='bold', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Create detailed comparison table
print(f"\n AIRCRAFT EFFICIENCY COMPARISON TABLE")
print(f"{'Aircraft':<12} {'Load Factor':<12} {'Fuel Eff':<12} {'Fuel $/Flight':<15} {'Pax/Flight':<12} {'Capacity':<10} {'Efficiency Rank':<15}")

# Create combined ranking (normalize both metrics and combine)
aircraft_efficiency['LF_RANK'] = aircraft_efficiency['LOAD_FACTOR'].rank(ascending=False)
aircraft_efficiency['FUEL_RANK'] = aircraft_efficiency['FUEL_EFFICIENCY'].rank(ascending=False)
aircraft_efficiency['COMBINED_RANK'] = (aircraft_efficiency['LF_RANK'] + aircraft_efficiency['FUEL_RANK']) / 2
aircraft_efficiency_ranked = aircraft_efficiency.sort_values('COMBINED_RANK')

for i, (_, row) in enumerate(aircraft_efficiency_ranked.iterrows()):
    aircraft = row['AIRCRAFT']
    load_factor = row['LOAD_FACTOR'] * 100
    fuel_eff = row['FUEL_EFFICIENCY']
    fuel_cost_flight = row['FUEL_COST_PER_FLIGHT']
    pax_flight = row['PASSENGERS_PER_FLIGHT']
    capacity = row['ESTIMATED_CAPACITY']
    
    
    print(f" {aircraft:<10} {load_factor:<11.1f}% {fuel_eff:<11.3f} ${fuel_cost_flight:<14,.0f} "
          f"{pax_flight:<11.1f} {capacity:<9} {row['COMBINED_RANK']:<14.1f}")


  

# %% [markdown]
"""
## E175 Exclusive Routes Analysis
Identify and analyze routes that are only served by E175 aircraft.
"""

# Get all routes in the dataset
all_routes = df.groupby(['ORIGIN', 'DESTINATION']).agg({
    'AIRCRAFT': lambda x: list(x.unique()),
    'FLIGHTS': 'sum',
    'PASSENGERS': 'sum',
    'TOTAL_REVENUE': 'sum',
    'TOTAL_COSTS': 'sum',
    'PROFIT': 'sum'
}).reset_index()

all_routes['ROUTE'] = all_routes['ORIGIN'] + '-' + all_routes['DESTINATION']
all_routes['NUM_AIRCRAFT_TYPES'] = all_routes['AIRCRAFT'].apply(len)
all_routes['AIRCRAFT_LIST'] = all_routes['AIRCRAFT'].apply(lambda x: ', '.join(sorted(x)))

# Identify E175-exclusive routes
e175_exclusive = all_routes[
    (all_routes['NUM_AIRCRAFT_TYPES'] == 1) & 
    (all_routes['AIRCRAFT'].apply(lambda x: 'E175' in x))
].copy()

# Identify routes with E175 + other aircraft
e175_shared = all_routes[
    (all_routes['NUM_AIRCRAFT_TYPES'] > 1) & 
    (all_routes['AIRCRAFT'].apply(lambda x: 'E175' in x))
].copy()

# Routes without E175
non_e175_routes = all_routes[
    ~all_routes['AIRCRAFT'].apply(lambda x: 'E175' in x)
].copy()
# Detailed analysis of E175-exclusive routes
if len(e175_exclusive) > 0:
    # Calculate performance metrics for exclusive routes
    e175_exclusive['PROFIT_PER_FLIGHT'] = e175_exclusive['PROFIT'] / e175_exclusive['FLIGHTS']
    e175_exclusive['REVENUE_PER_FLIGHT'] = e175_exclusive['TOTAL_REVENUE'] / e175_exclusive['FLIGHTS']
    e175_exclusive['COST_PER_FLIGHT'] = e175_exclusive['TOTAL_COSTS'] / e175_exclusive['FLIGHTS']
    e175_exclusive['PASSENGERS_PER_FLIGHT'] = e175_exclusive['PASSENGERS'] / e175_exclusive['FLIGHTS']
    
    # Sort by total profit
    e175_exclusive_sorted = e175_exclusive.sort_values('PROFIT', ascending=False)
    
# Compare E175 performance on exclusive vs shared routes
if len(e175_exclusive) > 0 and len(e175_shared) > 0:
    # Get E175 performance on shared routes
    e175_shared_performance = []
    
    for _, shared_route in e175_shared.iterrows():
        origin, destination = shared_route['ORIGIN'], shared_route['DESTINATION']
        
        # Get E175 specific data for this shared route
        e175_route_data = df[
            (df['AIRCRAFT'] == 'E175') & 
            (df['ORIGIN'] == origin) & 
            (df['DESTINATION'] == destination)
        ].agg({
            'FLIGHTS': 'sum',
            'PASSENGERS': 'sum',
            'TOTAL_REVENUE': 'sum',
            'TOTAL_COSTS': 'sum',
            'PROFIT': 'sum'
        })
        
        if e175_route_data['FLIGHTS'] > 0:
            e175_shared_performance.append({
                'route': f"{origin}-{destination}",
                'flights': e175_route_data['FLIGHTS'],
                'passengers': e175_route_data['PASSENGERS'],
                'revenue': e175_route_data['TOTAL_REVENUE'],
                'costs': e175_route_data['TOTAL_COSTS'],
                'profit': e175_route_data['PROFIT'],
                'profit_per_flight': e175_route_data['PROFIT'] / e175_route_data['FLIGHTS'],
                'passengers_per_flight': e175_route_data['PASSENGERS'] / e175_route_data['FLIGHTS'],
                'revenue_per_flight': e175_route_data['TOTAL_REVENUE'] / e175_route_data['FLIGHTS']
            })
    
    if e175_shared_performance:
        shared_df = pd.DataFrame(e175_shared_performance)
        
        # Calculate averages
        exclusive_avg_profit_flight = e175_exclusive['PROFIT_PER_FLIGHT'].mean()
        exclusive_avg_pax_flight = e175_exclusive['PASSENGERS_PER_FLIGHT'].mean()
        exclusive_avg_rev_flight = e175_exclusive['REVENUE_PER_FLIGHT'].mean()
        
        shared_avg_profit_flight = shared_df['profit_per_flight'].mean()
        shared_avg_pax_flight = shared_df['passengers_per_flight'].mean()
        shared_avg_rev_flight = shared_df['revenue_per_flight'].mean()
        
        profit_diff = exclusive_avg_profit_flight - shared_avg_profit_flight
        profit_diff_pct = (profit_diff / shared_avg_profit_flight) * 100 if shared_avg_profit_flight != 0 else 0
  
        
        pax_diff = exclusive_avg_pax_flight - shared_avg_pax_flight
        pax_diff_pct = (pax_diff / shared_avg_pax_flight) * 100 if shared_avg_pax_flight != 0 else 0

        
        rev_diff = exclusive_avg_rev_flight - shared_avg_rev_flight
        rev_diff_pct = (rev_diff / shared_avg_rev_flight) * 100 if shared_avg_rev_flight != 0 else 0

        # Market share analysis
        exclusive_flights = e175_exclusive['FLIGHTS'].sum()
        shared_flights = shared_df['flights'].sum()
        total_e175_flights = exclusive_flights + shared_flights
        
        # Revenue contribution
        exclusive_revenue = e175_exclusive['TOTAL_REVENUE'].sum()
        shared_revenue = shared_df['revenue'].sum()
        total_e175_revenue = exclusive_revenue + shared_revenue

        # Profit contribution
        exclusive_profit = e175_exclusive['PROFIT'].sum()
        shared_profit = shared_df['profit'].sum()
        total_e175_profit = exclusive_profit + shared_profit
      

# Get all routes that E175 operates on

e175_routes = df[df['AIRCRAFT'] == 'E175'].groupby(['ORIGIN', 'DESTINATION']).agg({
    'FLIGHTS': 'sum',
    'PASSENGERS': 'sum', 
    'TOTAL_REVENUE': 'sum',
    'TOTAL_COSTS': 'sum',
    'PROFIT': 'sum'
}).reset_index()

e175_routes['ROUTE'] = e175_routes['ORIGIN'] + '-' + e175_routes['DESTINATION']
e175_routes['E175_PROFIT_PER_FLIGHT'] = e175_routes['PROFIT'] / e175_routes['FLIGHTS']


route_comparison = []

for _, e175_route in e175_routes.iterrows():
    origin = e175_route['ORIGIN']
    destination = e175_route['DESTINATION']
    route_name = e175_route['ROUTE']
    
    # Get all aircraft on this route
    route_data = df[(df['ORIGIN'] == origin) & (df['DESTINATION'] == destination)]
    
    # Calculate performance by aircraft on this route
    route_by_aircraft = route_data.groupby('AIRCRAFT').agg({
        'FLIGHTS': 'sum',
        'PASSENGERS': 'sum',
        'TOTAL_REVENUE': 'sum', 
        'TOTAL_COSTS': 'sum',
        'PROFIT': 'sum'
    }).reset_index()
    
    route_by_aircraft['PROFIT_PER_FLIGHT'] = route_by_aircraft['PROFIT'] / route_by_aircraft['FLIGHTS']
    
    # Get E175 performance on this route
    e175_perf = route_by_aircraft[route_by_aircraft['AIRCRAFT'] == 'E175']
    if len(e175_perf) > 0:
        e175_profit_per_flight = e175_perf['PROFIT_PER_FLIGHT'].iloc[0]
        
        # Get other aircraft performance on same route
        other_aircraft = route_by_aircraft[route_by_aircraft['AIRCRAFT'] != 'E175']
        
        route_info = {
            'ROUTE': route_name,
            'E175_PROFIT_PER_FLIGHT': e175_profit_per_flight,
            'E175_FLIGHTS': e175_perf['FLIGHTS'].iloc[0],
            'OTHER_AIRCRAFT_COUNT': len(other_aircraft),
            'ROUTE_TYPE': 'SHARED' if len(other_aircraft) > 0 else 'EXCLUSIVE'
        }
        
        if len(other_aircraft) > 0:
            route_info['OTHER_AVG_PROFIT_PER_FLIGHT'] = other_aircraft['PROFIT_PER_FLIGHT'].mean()
            route_info['OTHER_BEST_PROFIT_PER_FLIGHT'] = other_aircraft['PROFIT_PER_FLIGHT'].max()
            route_info['OTHER_AIRCRAFT_LIST'] = ', '.join(other_aircraft['AIRCRAFT'].tolist())
            route_info['PERFORMANCE_GAP'] = e175_profit_per_flight - other_aircraft['PROFIT_PER_FLIGHT'].mean()
        else:
            route_info['OTHER_AVG_PROFIT_PER_FLIGHT'] = None
            route_info['OTHER_BEST_PROFIT_PER_FLIGHT'] = None
            route_info['OTHER_AIRCRAFT_LIST'] = 'None'
            route_info['PERFORMANCE_GAP'] = None
            
        route_comparison.append(route_info)

route_comparison_df = pd.DataFrame(route_comparison)

# Analyze shared vs exclusive routes for E175
shared_e175_routes = route_comparison_df[route_comparison_df['ROUTE_TYPE'] == 'SHARED']
exclusive_e175_routes = route_comparison_df[route_comparison_df['ROUTE_TYPE'] == 'EXCLUSIVE']


# Create visualization comparing E175 vs other aircraft on shared routes
if len(shared_e175_routes) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Chart 1: Route-by-route comparison
    shared_routes_viz = shared_e175_routes.sort_values('PERFORMANCE_GAP')
    
    x_pos = range(len(shared_routes_viz))
    ax1.barh(x_pos, shared_routes_viz['E175_PROFIT_PER_FLIGHT'], 
             alpha=0.7, color='red', label='E175')
    ax1.barh(x_pos, shared_routes_viz['OTHER_AVG_PROFIT_PER_FLIGHT'], 
             alpha=0.7, color='blue', label='Other Aircraft Avg')
    
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(shared_routes_viz['ROUTE'], fontsize=8)
    ax1.set_xlabel('Profit per Flight ($)')
    ax1.set_title('E175 vs Other Aircraft: Profit per Flight by Shared Route')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add performance gap annotations
    for i, gap in enumerate(shared_routes_viz['PERFORMANCE_GAP']):
        color = 'red' if gap < 0 else 'green'
        ax1.annotate(f'${gap:,.0f}', 
                    xy=(max(shared_routes_viz['E175_PROFIT_PER_FLIGHT'].iloc[i], 
                           shared_routes_viz['OTHER_AVG_PROFIT_PER_FLIGHT'].iloc[i]), i),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=8, color=color, weight='bold')
    
    # Chart 2: Performance gap distribution
    ax2.hist(shared_routes_viz['PERFORMANCE_GAP'], bins=10, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.set_xlabel('Performance Gap (E175 - Other Aircraft) ($)')
    ax2.set_ylabel('Number of Routes')
    ax2.set_title('Distribution of E175 Performance Gap on Shared Routes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %%



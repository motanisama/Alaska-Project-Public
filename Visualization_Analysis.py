# %% [markdown]
"""
# Operating Cost Per Flight Analysis
This notebook analyzes operating costs per flight by aircraft type and creates visualizations
comparing different aircraft metrics using the CA_CANDIDATE_PROJECT_DATA.xlsx dataset.

## Objectives:
- Calculate operating cost per flight for each aircraft type
- Create bar charts comparing aircraft performance metrics
- Analyze cost efficiency across different aircraft
- Provide actionable insights for fleet optimization
- Analyze E175  Operation metrics

"""

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

print("Libraries imported successfully!")

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# %%
# Function to find Excel file
def find_excel_file():
    """Find the Excel file in current directory or parent directories"""
    current_dir = Path.cwd()
    
    # Check current directory first
    excel_file = current_dir / "CA_CANDIDATE_PROJECT_DATA.xlsx"
    if excel_file.exists():
        return str(excel_file)
    
    # Check parent directory
    parent_excel = current_dir.parent / "CA_CANDIDATE_PROJECT_DATA.xlsx"
    if parent_excel.exists():
        return str(parent_excel)
    
    # Check if we're in a subdirectory, go up one more level
    grandparent_excel = current_dir.parent.parent / "CA_CANDIDATE_PROJECT_DATA.xlsx"
    if grandparent_excel.exists():
        return str(grandparent_excel)
    
    return None

# %%
# Load the Excel file
excel_location = find_excel_file()
if excel_location:
    EXCEL_FILE = excel_location
    print(f"✅ Found Excel file at: {EXCEL_FILE}")
else:
    EXCEL_FILE = "CA_CANDIDATE_PROJECT_DATA.xlsx"
    print(f"⚠️  Using default path: {EXCEL_FILE}")

try:
    df = pd.read_excel(EXCEL_FILE)
    print(f"✅ Successfully loaded Excel file")
    print(f"📊 Dataset shape: {df.shape}")
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.upper()
    
    # Convert MONTH to datetime and add YEAR
    if 'MONTH' in df.columns:
        df['MONTH'] = pd.to_datetime(df['MONTH'])
        df['YEAR'] = df['MONTH'].dt.year
        print(f"📅 Available years: {sorted(df['YEAR'].unique())}")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    raise

# %%
# Explore the dataset structure
print("=== DATASET EXPLORATION ===")
print(f"\nDataset columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Check for aircraft identifier column
aircraft_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['aircraft', 'plane', 'type'])]
print(f"\nAircraft-related columns: {aircraft_columns}")

# Identify the aircraft column
if 'AIRCRAFT' in df.columns:
    aircraft_col = 'AIRCRAFT'
elif 'AIRCRAFT_TYPE' in df.columns:
    aircraft_col = 'AIRCRAFT_TYPE'
else:
    aircraft_col = aircraft_columns[0] if aircraft_columns else None

if aircraft_col:
    print(f"✅ Using aircraft column: {aircraft_col}")
    print(f"Unique aircraft types: {df[aircraft_col].nunique()}")
    print(f"Aircraft types: {sorted(df[aircraft_col].unique())}")
else:
    print("❌ No aircraft column found!")

# %%
# Define cost components for operating cost calculation
print("\n=== OPERATING COST COMPONENTS ===")

# All potential cost columns
cost_columns = [
    'FUEL_COST',
    'AIRPORT_FEE_COST', 
    'AIRCRAFT_OPERATION_COST',
    'CREW_AND_LABOR_COST'
]

# Check which cost columns exist
existing_cost_columns = [col for col in cost_columns if col in df.columns]
print("Operating cost components found:")
for col in existing_cost_columns:
    print(f"  ✅ {col}")

missing_cost_columns = [col for col in cost_columns if col not in df.columns]
if missing_cost_columns:
    print("Missing cost components:")
    for col in missing_cost_columns:
        print(f"  ❌ {col}")

# %%
# Calculate total operating costs
if existing_cost_columns:
    # Calculate total operating costs
    df['TOTAL_OPERATING_COST'] = df[existing_cost_columns].sum(axis=1)
    print(f"✅ Total operating cost calculated from {len(existing_cost_columns)} components")
    
    # Display cost breakdown summary
    print(f"\nOperating Cost Breakdown (Total Dataset):")
    print(f"{'Component':<25} {'Total Cost':<15} {'Percentage':<12}")
    print("-" * 55)
    
    total_all_costs = df['TOTAL_OPERATING_COST'].sum()
    for col in existing_cost_columns:
        component_total = df[col].sum()
        percentage = (component_total / total_all_costs) * 100
        print(f"{col:<25} ${component_total:<14,.0f} {percentage:<11.1f}%")
    
    print(f"{'TOTAL OPERATING COST':<25} ${total_all_costs:<14,.0f} {'100.0%':<12}")
    
else:
    print("❌ No cost columns found for operating cost calculation!")

# %%
# Calculate operating cost per flight by aircraft
if aircraft_col and 'TOTAL_OPERATING_COST' in df.columns:
    
    # Group by aircraft and calculate metrics
    aircraft_metrics = df.groupby(aircraft_col).agg({
        'TOTAL_OPERATING_COST': ['sum', 'mean', 'count'],
        'FUEL_COST': 'mean' if 'FUEL_COST' in df.columns else lambda x: 0,
        'AIRPORT_FEE_COST': 'mean' if 'AIRPORT_FEE_COST' in df.columns else lambda x: 0,
        'AIRCRAFT_OPERATION_COST': 'mean' if 'AIRCRAFT_OPERATION_COST' in df.columns else lambda x: 0,
        'CREW_AND_LABOR_COST': 'mean' if 'CREW_AND_LABOR_COST' in df.columns else lambda x: 0,
    }).round(0)
    
    # Flatten column names
    aircraft_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aircraft_metrics.columns]
    
    # Rename for clarity
    aircraft_metrics = aircraft_metrics.rename(columns={
        'TOTAL_OPERATING_COST_sum': 'TOTAL_COST',
        'TOTAL_OPERATING_COST_mean': 'COST_PER_FLIGHT',
        'TOTAL_OPERATING_COST_count': 'FLIGHTS'
    })
    
    # Sort by cost per flight (descending)
    aircraft_metrics = aircraft_metrics.sort_values('COST_PER_FLIGHT', ascending=False)

    
    for aircraft, row in aircraft_metrics.iterrows():
        flights = int(row['FLIGHTS'])
        cost_per_flight = row['COST_PER_FLIGHT']
        total_cost = row['TOTAL_COST']
        fuel_cost = row.get('FUEL_COST_mean', 0)
        airport_cost = row.get('AIRPORT_FEE_COST_mean', 0)
        operation_cost = row.get('AIRCRAFT_OPERATION_COST_mean', 0)
        crew_cost = row.get('CREW_AND_LABOR_COST_mean', 0)
        

  

# %%
# Create cost breakdown comparison chart
if aircraft_col and len(existing_cost_columns) > 1:
    
    # Select top 10 aircraft by cost per flight for detailed breakdown
    top_10_aircraft = aircraft_metrics.head(10)
    
    # Prepare data for cost per mile stacked bar chart
    cost_breakdown_data = []
    aircraft_names = []
    
    for aircraft in top_10_aircraft.index:
        aircraft_names.append(aircraft)
        breakdown = []
        
        # Get average miles per flight for this aircraft
        avg_miles = df[df[aircraft_col] == aircraft]['MILES_DISTANCE'].mean() if 'MILES_DISTANCE' in df.columns else 1
        
        for cost_col in existing_cost_columns:
            col_name = f"{cost_col}_mean"
            if col_name in aircraft_metrics.columns:
                # Convert cost per flight to cost per mile
                cost_per_flight = aircraft_metrics.loc[aircraft, col_name]
                cost_per_mile = cost_per_flight / avg_miles if avg_miles > 0 else 0
                breakdown.append(cost_per_mile)
            else:
                breakdown.append(0)
        cost_breakdown_data.append(breakdown)
    
    # Convert to numpy array for easier manipulation
    cost_breakdown_array = np.array(cost_breakdown_data)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for each cost component
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bottom = np.zeros(len(aircraft_names))
    
    # Create stacked bars
    for i, cost_col in enumerate(existing_cost_columns):
        cost_name = cost_col.replace('_COST', '').replace('_', ' ').title()
        ax.bar(aircraft_names, cost_breakdown_array[:, i], bottom=bottom, 
               label=cost_name, color=colors[i % len(colors)])
        bottom += cost_breakdown_array[:, i]
    
    # Customize the chart
    ax.set_title('Operating Cost Per Mile by Aircraft Type (Top 10)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Aircraft Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Operating Cost Per Mile ($)', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Format y-axis to show cost per mile with more appropriate precision
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('operating_cost_per_mile_breakdown_by_aircraft.png', dpi=300, bbox_inches='tight')
    print("✅ Operating cost per mile breakdown chart saved as 'operating_cost_per_mile_breakdown_by_aircraft.png'")
    
    plt.show()

# %%
# Additional metrics comparison

if aircraft_col:
    # Calculate additional metrics if available
    additional_metrics = df.groupby(aircraft_col).agg({
        'PASSENGERS': 'mean' if 'PASSENGERS' in df.columns else lambda x: 0,
        'MILES_DISTANCE': 'mean' if 'MILES_DISTANCE' in df.columns else lambda x: 0,
        'TOTAL_REVENUE': 'mean' if 'TOTAL_REVENUE' in df.columns else lambda x: 0,
    }).round(2)
    
    # Add revenue columns if they exist
    revenue_columns = ['FIRST_CLASS_REVENUE', 'PREM_ECONOMY_REVENUE', 'ECONOMY_REVENUE', 
                      'BAGGAGE_REVENUE', 'OTHER_ANCILLARY_REVENUE', 'FREIGHT_REVENUE']
    existing_revenue_columns = [col for col in revenue_columns if col in df.columns]
    
    if existing_revenue_columns and 'TOTAL_REVENUE' not in df.columns:
        df['TOTAL_REVENUE'] = df[existing_revenue_columns].sum(axis=1)
        additional_metrics = df.groupby(aircraft_col).agg({
            'PASSENGERS': 'mean' if 'PASSENGERS' in df.columns else lambda x: 0,
            'MILES_DISTANCE': 'mean' if 'MILES_DISTANCE' in df.columns else lambda x: 0,
            'TOTAL_REVENUE': 'mean',
        }).round(2)
    
    # Merge with cost metrics
    if 'COST_PER_FLIGHT' in aircraft_metrics.columns:
        combined_metrics = aircraft_metrics[['COST_PER_FLIGHT', 'FLIGHTS']].join(additional_metrics)
        
        # Calculate efficiency metrics
        if 'TOTAL_REVENUE' in combined_metrics.columns:
            combined_metrics['PROFIT_PER_FLIGHT'] = combined_metrics['TOTAL_REVENUE'] - combined_metrics['COST_PER_FLIGHT']
            combined_metrics['PROFIT_MARGIN'] = (combined_metrics['PROFIT_PER_FLIGHT'] / combined_metrics['TOTAL_REVENUE']) * 100
        
        if 'PASSENGERS' in combined_metrics.columns:
            combined_metrics['COST_PER_PASSENGER'] = combined_metrics['COST_PER_FLIGHT'] / combined_metrics['PASSENGERS']
        
        if 'MILES_DISTANCE' in combined_metrics.columns:
            combined_metrics['COST_PER_MILE'] = combined_metrics['COST_PER_FLIGHT'] / combined_metrics['MILES_DISTANCE']
            # Add profit margin per mile calculation
            if 'PROFIT_PER_FLIGHT' in combined_metrics.columns:
                combined_metrics['PROFIT_MARGIN_PER_MILE'] = combined_metrics['PROFIT_MARGIN'] / combined_metrics['MILES_DISTANCE']
        
      

# %%
# Create profit margin per mile chart

if 'combined_metrics' in locals() and 'PROFIT_MARGIN_PER_MILE' in combined_metrics.columns:
    
    # Filter out aircraft with valid profit margin per mile data
    valid_margin_per_mile = combined_metrics[combined_metrics['PROFIT_MARGIN_PER_MILE'].notna() & 
                                           (combined_metrics['PROFIT_MARGIN_PER_MILE'] != 0)]
    
    if len(valid_margin_per_mile) > 0:
        # Sort by profit margin per mile (descending)
        top_aircraft_margin_mile = valid_margin_per_mile.sort_values('PROFIT_MARGIN_PER_MILE', ascending=False).head(15)
        
        # Create the bar chart
        plt.figure(figsize=(14, 8))
        
        # Color bars based on positive/negative values
        colors = ['green' if x >= 0 else 'red' for x in top_aircraft_margin_mile['PROFIT_MARGIN_PER_MILE']]
        
        bars = plt.bar(range(len(top_aircraft_margin_mile)), top_aircraft_margin_mile['PROFIT_MARGIN_PER_MILE'], 
                       color=colors, alpha=0.7)
        
        # Customize the chart
        plt.title('Profit Margin Per Mile by Aircraft Type', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Aircraft Type', fontsize=12, fontweight='bold')
        plt.ylabel('Profit Margin Per Mile (%)', fontsize=12, fontweight='bold')
        
        # Set x-axis labels
        plt.xticks(range(len(top_aircraft_margin_mile)), top_aircraft_margin_mile.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (aircraft, margin) in enumerate(zip(top_aircraft_margin_mile.index, top_aircraft_margin_mile['PROFIT_MARGIN_PER_MILE'])):
            plt.text(i, margin + (abs(margin)*0.02 if margin >= 0 else -abs(margin)*0.02), 
                    f'{margin:.3f}%', ha='center', va='bottom' if margin >= 0 else 'top', fontweight='bold')
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()

        
        plt.show()
    else:
        print("No valid profit margin per mile data available for visualization")






# %%
# E175 Cargo and Seat Space Analysis

if aircraft_col and 'combined_metrics' in locals():
    
    # Check for relevant columns
    cargo_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['freight', 'cargo'])]
    seat_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['seat', 'capacity'])]
    passenger_class_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['first_class_pax', 'premium_economy_pax', 'economy_pax'])]
    
    print(f"Available cargo-related columns: {cargo_columns}")
    print(f"Available seat-related columns: {seat_columns}")
    print(f"Available passenger class columns: {passenger_class_columns}")
    
    # Find E175 variants
    e175_variants = [aircraft for aircraft in df[aircraft_col].unique() if 'E175' in str(aircraft).upper()]
    
    if e175_variants:
        e175_name = e175_variants[0]
        
        # Build aggregation dictionary with only existing columns
        agg_dict = {}
        
        # Check each column before adding to aggregation
        if 'FREIGHT_LBS' in df.columns:
            agg_dict['FREIGHT_LBS'] = ['mean', 'sum']
        if 'SEATS' in df.columns:
            agg_dict['SEATS'] = 'mean'
        if 'FIRST CLASS SEATS' in df.columns:  # Note the space in column name
            agg_dict['FIRST CLASS SEATS'] = 'mean'
        if 'PREM_ECONOMY_SEATS' in df.columns:
            agg_dict['PREM_ECONOMY_SEATS'] = 'mean'
        if 'ECONOMY_SEATS' in df.columns:
            agg_dict['ECONOMY_SEATS'] = 'mean'
        if 'PASSENGERS' in df.columns:
            agg_dict['PASSENGERS'] = 'mean'
        if 'FIRST_CLASS_PAX' in df.columns:
            agg_dict['FIRST_CLASS_PAX'] = 'mean'
        if 'PREMIUM_ECONOMY_PAX' in df.columns:
            agg_dict['PREMIUM_ECONOMY_PAX'] = 'mean'
        if 'ECONOMY_PAX' in df.columns:
            agg_dict['ECONOMY_PAX'] = 'mean'
        if 'FLIGHTS' in df.columns:
            agg_dict['FLIGHTS'] = 'sum'
        
        print(f"\nColumns to aggregate: {list(agg_dict.keys())}")
        
        # Calculate cargo and seat metrics by aircraft
        if agg_dict:  # Only proceed if we have columns to aggregate
            aircraft_space_metrics = df.groupby(aircraft_col).agg(agg_dict).round(2)
            
            # Flatten column names
            aircraft_space_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) and col[1] else col[0] if isinstance(col, tuple) else col for col in aircraft_space_metrics.columns]
            
            # Calculate utilization metrics
            if 'SEATS_mean' in aircraft_space_metrics.columns and 'PASSENGERS_mean' in aircraft_space_metrics.columns:
                aircraft_space_metrics['SEAT_UTILIZATION'] = (aircraft_space_metrics['PASSENGERS_mean'] / aircraft_space_metrics['SEATS_mean'] * 100).round(1)
            elif 'SEATS' in aircraft_space_metrics.columns and 'PASSENGERS' in aircraft_space_metrics.columns:
                aircraft_space_metrics['SEAT_UTILIZATION'] = (aircraft_space_metrics['PASSENGERS'] / aircraft_space_metrics['SEATS'] * 100).round(1)
            
            # Calculate freight per flight
            if 'FREIGHT_LBS_mean' in aircraft_space_metrics.columns:
                aircraft_space_metrics['FREIGHT_PER_FLIGHT'] = aircraft_space_metrics['FREIGHT_LBS_mean']
            elif 'FREIGHT_LBS' in aircraft_space_metrics.columns:
                aircraft_space_metrics['FREIGHT_PER_FLIGHT'] = aircraft_space_metrics['FREIGHT_LBS']
            
            # E175 specific analysis
            if e175_name in aircraft_space_metrics.index:
                e175_space_data = aircraft_space_metrics.loc[e175_name]
                
                print(f"\n E175 SPACE CONFIGURATION:")
                
                # Handle different possible column names for seats
                total_seats_col = None
                if 'SEATS_mean' in aircraft_space_metrics.columns:
                    total_seats_col = 'SEATS_mean'
                elif 'SEATS' in aircraft_space_metrics.columns:
                    total_seats_col = 'SEATS'
                
                if total_seats_col:
                    print(f"   • Total Seats: {e175_space_data[total_seats_col]:.0f}")
                
                # Check for class-specific seats
                first_class_col = None
                if 'FIRST CLASS SEATS_mean' in aircraft_space_metrics.columns:
                    first_class_col = 'FIRST CLASS SEATS_mean'
                elif 'FIRST CLASS SEATS' in aircraft_space_metrics.columns:
                    first_class_col = 'FIRST CLASS SEATS'
                
                if first_class_col and first_class_col in e175_space_data.index:
                    print(f"   • First Class Seats: {e175_space_data[first_class_col]:.0f}")
                
                prem_econ_col = None
                if 'PREM_ECONOMY_SEATS_mean' in aircraft_space_metrics.columns:
                    prem_econ_col = 'PREM_ECONOMY_SEATS_mean'
                elif 'PREM_ECONOMY_SEATS' in aircraft_space_metrics.columns:
                    prem_econ_col = 'PREM_ECONOMY_SEATS'
                
                if prem_econ_col and prem_econ_col in e175_space_data.index:
                    print(f"   • Premium Economy Seats: {e175_space_data[prem_econ_col]:.0f}")
                
                economy_col = None
                if 'ECONOMY_SEATS_mean' in aircraft_space_metrics.columns:
                    economy_col = 'ECONOMY_SEATS_mean'
                elif 'ECONOMY_SEATS' in aircraft_space_metrics.columns:
                    economy_col = 'ECONOMY_SEATS'
                
                if economy_col and economy_col in e175_space_data.index:
                    print(f"   • Economy Seats: {e175_space_data[economy_col]:.0f}")
                
                print(f"\n E175 UTILIZATION METRICS:")
                
                # Handle passengers column
                passengers_col = None
                if 'PASSENGERS_mean' in aircraft_space_metrics.columns:
                    passengers_col = 'PASSENGERS_mean'
                elif 'PASSENGERS' in aircraft_space_metrics.columns:
                    passengers_col = 'PASSENGERS'
                
                if passengers_col:
                    print(f"   • Average Passengers per Flight: {e175_space_data[passengers_col]:.0f}")
                
                if 'SEAT_UTILIZATION' in aircraft_space_metrics.columns:
                    print(f"   • Seat Utilization Rate: {e175_space_data['SEAT_UTILIZATION']:.1f}%")
                
                if 'FREIGHT_PER_FLIGHT' in aircraft_space_metrics.columns:
                    print(f"   • Average Freight per Flight: {e175_space_data['FREIGHT_PER_FLIGHT']:.0f} lbs")
                
                # Rankings
                print(f"\n E175 SPACE RANKINGS:")
                
                if total_seats_col and total_seats_col in aircraft_space_metrics.columns:
                    seat_ranking = aircraft_space_metrics[total_seats_col].rank(ascending=False)
                    e175_seat_rank = int(seat_ranking[e175_name])
                    print(f"   • Total Seats Rank: #{e175_seat_rank} out of {len(aircraft_space_metrics)}")
                
                if 'SEAT_UTILIZATION' in aircraft_space_metrics.columns:
                    util_ranking = aircraft_space_metrics['SEAT_UTILIZATION'].rank(ascending=False)
                    e175_util_rank = int(util_ranking[e175_name])
                    print(f"   • Seat Utilization Rank: #{e175_util_rank} out of {len(aircraft_space_metrics)}")
                
                if 'FREIGHT_PER_FLIGHT' in aircraft_space_metrics.columns:
                    freight_ranking = aircraft_space_metrics['FREIGHT_PER_FLIGHT'].rank(ascending=False)
                    e175_freight_rank = int(freight_ranking[e175_name])
                    print(f"   • Freight Capacity Rank: #{e175_freight_rank} out of {len(aircraft_space_metrics)}")
            
            # Create comprehensive space analysis charts
            # Determine which columns to use for charts
            seats_col = 'SEATS_mean' if 'SEATS_mean' in aircraft_space_metrics.columns else 'SEATS' if 'SEATS' in aircraft_space_metrics.columns else None
            passengers_col = 'PASSENGERS_mean' if 'PASSENGERS_mean' in aircraft_space_metrics.columns else 'PASSENGERS' if 'PASSENGERS' in aircraft_space_metrics.columns else None
            
            # Chart 1: Seat Configuration Comparison
            if seats_col:
                
                # Top 15 aircraft by total seats
                top_15_seats = aircraft_space_metrics.nlargest(15, seats_col)
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
                
                # 1. Total Seats Comparison
                colors = ['red' if aircraft == e175_name else 'lightblue' for aircraft in top_15_seats.index]
                bars1 = ax1.bar(range(len(top_15_seats)), top_15_seats[seats_col], color=colors, alpha=0.8)
                ax1.set_title('Total Seats by Aircraft Type', fontweight='bold', fontsize=14)
                ax1.set_ylabel('Number of Seats')
                ax1.set_xticks(range(len(top_15_seats)))
                ax1.set_xticklabels(top_15_seats.index, rotation=45, ha='right')
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, (aircraft, seats) in enumerate(zip(top_15_seats.index, top_15_seats[seats_col])):
                    label = f'{seats:.0f}'
                    if aircraft == e175_name:
                        label += ' (E175)'
                    ax1.text(i, seats + seats*0.01, label, ha='center', va='bottom', fontweight='bold')
                
                # 2. Seat Utilization Comparison
                if 'SEAT_UTILIZATION' in aircraft_space_metrics.columns:
                    util_data = aircraft_space_metrics.dropna(subset=['SEAT_UTILIZATION']).nlargest(15, 'SEAT_UTILIZATION')
                    colors2 = ['red' if aircraft == e175_name else 'lightgreen' for aircraft in util_data.index]
                    bars2 = ax2.bar(range(len(util_data)), util_data['SEAT_UTILIZATION'], color=colors2, alpha=0.8)
                    ax2.set_title('Seat Utilization Rate by Aircraft Type', fontweight='bold', fontsize=14)
                    ax2.set_ylabel('Utilization Rate (%)')
                    ax2.set_xticks(range(len(util_data)))
                    ax2.set_xticklabels(util_data.index, rotation=45, ha='right')
                    ax2.grid(axis='y', alpha=0.3)
                    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% Target')
                    ax2.legend()
                    
                    # Add value labels
                    for i, (aircraft, util) in enumerate(zip(util_data.index, util_data['SEAT_UTILIZATION'])):
                        label = f'{util:.1f}%'
                        if aircraft == e175_name:
                            label += ' (E175)'
                        ax2.text(i, util + util*0.01, label, ha='center', va='bottom', fontweight='bold')
                else:
                    ax2.text(0.5, 0.5, 'Seat Utilization Data\nNot Available', ha='center', va='center', 
                            transform=ax2.transAxes, fontsize=12, fontweight='bold')
                    ax2.set_title('Seat Utilization Rate by Aircraft Type', fontweight='bold', fontsize=14)
                
                # 3. Freight Capacity Comparison
                if 'FREIGHT_PER_FLIGHT' in aircraft_space_metrics.columns:
                    freight_data = aircraft_space_metrics.dropna(subset=['FREIGHT_PER_FLIGHT']).nlargest(15, 'FREIGHT_PER_FLIGHT')
                    colors3 = ['red' if aircraft == e175_name else 'gold' for aircraft in freight_data.index]
                    bars3 = ax3.bar(range(len(freight_data)), freight_data['FREIGHT_PER_FLIGHT'], color=colors3, alpha=0.8)
                    ax3.set_title('Average Freight per Flight by Aircraft Type', fontweight='bold', fontsize=14)
                    ax3.set_ylabel('Freight (lbs)')
                    ax3.set_xticks(range(len(freight_data)))
                    ax3.set_xticklabels(freight_data.index, rotation=45, ha='right')
                    ax3.grid(axis='y', alpha=0.3)
                    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                    
                    # Add value labels
                    for i, (aircraft, freight) in enumerate(zip(freight_data.index, freight_data['FREIGHT_PER_FLIGHT'])):
                        label = f'{freight:,.0f}'
                        if aircraft == e175_name:
                            label += ' (E175)'
                        ax3.text(i, freight + freight*0.01, label, ha='center', va='bottom', fontweight='bold', fontsize=8)
                else:
                    ax3.text(0.5, 0.5, 'Freight Data\nNot Available', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=12, fontweight='bold')
                    ax3.set_title('Average Freight per Flight by Aircraft Type', fontweight='bold', fontsize=14)
                
                # 4. Passenger vs Seat Capacity Scatter Plot
                if passengers_col and seats_col:
                    scatter_data = aircraft_space_metrics.dropna(subset=[passengers_col, seats_col])
                    
                    # Color E175 differently
                    colors4 = ['red' if aircraft == e175_name else 'purple' for aircraft in scatter_data.index]
                    sizes = [100 if aircraft == e175_name else 60 for aircraft in scatter_data.index]
                    
                    ax4.scatter(scatter_data[seats_col], scatter_data[passengers_col], 
                               c=colors4, s=sizes, alpha=0.7)
                    
                    # Add diagonal line for 100% utilization
                    max_seats = scatter_data[seats_col].max()
                    ax4.plot([0, max_seats], [0, max_seats], 'k--', alpha=0.5, label='100% Utilization')
                    
                    ax4.set_title('Average Passengers vs Seat Capacity', fontweight='bold', fontsize=14)
                    ax4.set_xlabel('Seat Capacity')
                    ax4.set_ylabel('Average Passengers')
                    ax4.grid(True, alpha=0.3)
                    ax4.legend()
                    
                    # Label E175 point
                    if e175_name in scatter_data.index:
                        e175_seats = scatter_data.loc[e175_name, seats_col]
                        e175_pax = scatter_data.loc[e175_name, passengers_col]
                        ax4.annotate('E175', (e175_seats, e175_pax), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontweight='bold', fontsize=10)
                else:
                    ax4.text(0.5, 0.5, 'Passenger vs Seat\nData Not Available', ha='center', va='center', 
                            transform=ax4.transAxes, fontsize=12, fontweight='bold')
                    ax4.set_title('Average Passengers vs Seat Capacity', fontweight='bold', fontsize=14)
                
                plt.suptitle('E175 Space & Capacity Analysis (E175 highlighted in red)', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.show()
            
        
    
    else:
        print(" No E175 aircraft found in dataset")

# %%





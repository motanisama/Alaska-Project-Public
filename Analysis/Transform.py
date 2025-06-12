# %%
import pandas as pd
import numpy as np
from datetime import datetime
import os

def transform_alaska_data(input_file_path, output_file_path=None):
    """
    Transform Alaska raw data by adding calculated columns for enhanced analysis.
    
    Parameters:
    input_file_path (str): Path to the raw Alaska data Excel file
    output_file_path (str): Path for the transformed data output (optional)
    
    Returns:
    pd.DataFrame: Transformed dataframe with additional columns
    """
    
    # Load the raw data
    print("Loading Alaska raw data...")
    df = pd.read_excel(input_file_path)
    print(f"Original data shape: {df.shape}")
    
    # Create a copy for transformation
    transformed_df = df.copy()
    
    # 1. Revenue Calculations
    print("Adding revenue calculations...")
    transformed_df['TOTAL_PASSENGER_REVENUE'] = (
        transformed_df['FIRST_CLASS_REVENUE'] + 
        transformed_df['PREM_ECONOMY_REVENUE'] + 
        transformed_df['ECONOMY_REVENUE']
    )
    
    transformed_df['TOTAL_REVENUE'] = (
        transformed_df['TOTAL_PASSENGER_REVENUE'] + 
        transformed_df['BAGGAGE_REVENUE'] + 
        transformed_df['OTHER_ANCILLARY_REVENUE'] + 
        transformed_df['FREIGHT_REVENUE']
    )
    
    # 2. Cost Calculations
    print("Adding cost calculations...")
    transformed_df['TOTAL_OPERATING_COST'] = (
        transformed_df['AIRCRAFT_OPERATION_COST'] + 
        transformed_df['AIRPORT_FEE_COST'] + 
        transformed_df['CREW_AND_LABOR_COST'] + 
        transformed_df['FUEL_COST']
    )
    
    # 3. Profitability Metrics
    print("Adding profitability metrics...")
    transformed_df['GROSS_PROFIT'] = transformed_df['TOTAL_REVENUE'] - transformed_df['TOTAL_OPERATING_COST']
    transformed_df['PROFIT_MARGIN'] = np.where(
        transformed_df['TOTAL_REVENUE'] > 0,
        transformed_df['GROSS_PROFIT'] / transformed_df['TOTAL_REVENUE'] * 100,
        0
    )
    
    # 4. Per-Unit Metrics
    print("Adding per-unit metrics...")
    # Revenue per passenger
    transformed_df['REVENUE_PER_PASSENGER'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['TOTAL_PASSENGER_REVENUE'] / transformed_df['PASSENGERS'],
        0
    )
    
    # Cost per passenger
    transformed_df['COST_PER_PASSENGER'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['TOTAL_OPERATING_COST'] / transformed_df['PASSENGERS'],
        0
    )
    
    # Profit per passenger
    transformed_df['PROFIT_PER_PASSENGER'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['GROSS_PROFIT'] / transformed_df['PASSENGERS'],
        0
    )
    
    # Revenue per mile
    transformed_df['REVENUE_PER_MILE'] = np.where(
        transformed_df['MILES_DISTANCE'] > 0,
        transformed_df['TOTAL_REVENUE'] / transformed_df['MILES_DISTANCE'],
        0
    )
    
    # Cost per mile
    transformed_df['COST_PER_MILE'] = np.where(
        transformed_df['MILES_DISTANCE'] > 0,
        transformed_df['TOTAL_OPERATING_COST'] / transformed_df['MILES_DISTANCE'],
        0
    )
    
    # 5. Efficiency Metrics
    print("Adding efficiency metrics...")
    # Load factor (passenger utilization)
    transformed_df['LOAD_FACTOR'] = np.where(
        transformed_df['SEATS'] > 0,
        transformed_df['PASSENGERS'] / transformed_df['SEATS'] * 100,
        0
    )
    
    # Revenue per Available Seat Mile (RASM)
    transformed_df['RASM'] = np.where(
        transformed_df['ASMS'] > 0,
        transformed_df['TOTAL_REVENUE'] / transformed_df['ASMS'] * 100,
        0
    )
    
    # Cost per Available Seat Mile (CASM)
    transformed_df['CASM'] = np.where(
        transformed_df['ASMS'] > 0,
        transformed_df['TOTAL_OPERATING_COST'] / transformed_df['ASMS'] * 100,
        0
    )
    
    # Revenue per passenger mile
    transformed_df['REVENUE_PER_PAX_MILE'] = np.where(
        (transformed_df['PASSENGERS'] > 0) & (transformed_df['MILES_DISTANCE'] > 0),
        transformed_df['TOTAL_PASSENGER_REVENUE'] / (transformed_df['PASSENGERS'] * transformed_df['MILES_DISTANCE']),
        0
    )
    
    # 6. Flight Performance Metrics
    print("Adding flight performance metrics...")
    # Average passengers per flight
    transformed_df['AVG_PASSENGERS_PER_FLIGHT'] = np.where(
        transformed_df['FLIGHTS'] > 0,
        transformed_df['PASSENGERS'] / transformed_df['FLIGHTS'],
        0
    )
    
    # Average flight hours per flight
    transformed_df['AVG_FLIGHT_HOURS_PER_FLIGHT'] = np.where(
        transformed_df['FLIGHTS'] > 0,
        transformed_df['FLIGHT_HOURS'] / transformed_df['FLIGHTS'],
        0
    )
    
    # Revenue per flight
    transformed_df['REVENUE_PER_FLIGHT'] = np.where(
        transformed_df['FLIGHTS'] > 0,
        transformed_df['TOTAL_REVENUE'] / transformed_df['FLIGHTS'],
        0
    )
    
    # Cost per flight
    transformed_df['COST_PER_FLIGHT'] = np.where(
        transformed_df['FLIGHTS'] > 0,
        transformed_df['TOTAL_OPERATING_COST'] / transformed_df['FLIGHTS'],
        0
    )
    
    # 7. Freight Metrics
    print("Adding freight metrics...")
    # Freight revenue per pound
    transformed_df['FREIGHT_REVENUE_PER_LB'] = np.where(
        transformed_df['FREIGHT_LBS'] > 0,
        transformed_df['FREIGHT_REVENUE'] / transformed_df['FREIGHT_LBS'],
        0
    )
    
    # 8. Route Characteristics
    print("Adding route characteristics...")
    # Create route identifier
    transformed_df['ROUTE'] = transformed_df['ORIGIN'] + '-' + transformed_df['DESTINATION']
    
    # Route type (domestic vs international)
    transformed_df['ROUTE_TYPE'] = np.where(
        transformed_df['ORIG_STATE'] == transformed_df['DEST_STATE'],
        'Intrastate',
        np.where(
            (transformed_df['ORIG_STATE'].isin(['Alaska', 'Hawaii', 'California', 'Washington', 'Oregon'])) &
            (transformed_df['DEST_STATE'].isin(['Alaska', 'Hawaii', 'California', 'Washington', 'Oregon'])),
            'Domestic',
            'International'
        )
    )
    
    # Distance categories
    transformed_df['DISTANCE_CATEGORY'] = pd.cut(
        transformed_df['MILES_DISTANCE'],
        bins=[0, 500, 1000, 2000, float('inf')],
        labels=['Short-haul (<500mi)', 'Medium-haul (500-1000mi)', 'Long-haul (1000-2000mi)', 'Ultra-long-haul (>2000mi)']
    )
    
    # 9. Time-based Features
    print("Adding time-based features...")
    # Extract year, month, quarter
    transformed_df['YEAR'] = transformed_df['MONTH'].dt.year
    transformed_df['MONTH_NUM'] = transformed_df['MONTH'].dt.month
    transformed_df['QUARTER'] = transformed_df['MONTH'].dt.quarter
    transformed_df['MONTH_NAME'] = transformed_df['MONTH'].dt.strftime('%B')
    
    # Season classification
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    transformed_df['SEASON'] = transformed_df['MONTH_NUM'].map(season_map)
    
    # 10. Performance Rankings (within each aircraft type)
    print("Adding performance rankings...")
    transformed_df['PROFIT_RANK_BY_AIRCRAFT'] = transformed_df.groupby('AIRCRAFT')['GROSS_PROFIT'].rank(ascending=False)
    transformed_df['LOAD_FACTOR_RANK_BY_AIRCRAFT'] = transformed_df.groupby('AIRCRAFT')['LOAD_FACTOR'].rank(ascending=False)
    transformed_df['REVENUE_RANK_BY_AIRCRAFT'] = transformed_df.groupby('AIRCRAFT')['TOTAL_REVENUE'].rank(ascending=False)
    
    # 11. Class Mix Analysis
    print("Adding class mix analysis...")
    # Percentage of passengers in each class
    transformed_df['FIRST_CLASS_PAX_PCT'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['FIRST_CLASS_PAX'] / transformed_df['PASSENGERS'] * 100,
        0
    )
    
    transformed_df['PREMIUM_ECONOMY_PAX_PCT'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['PREMIUM_ECONOMY_PAX'] / transformed_df['PASSENGERS'] * 100,
        0
    )
    
    transformed_df['ECONOMY_PAX_PCT'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['ECONOMY_PAX'] / transformed_df['PASSENGERS'] * 100,
        0
    )
    
    # Average revenue per passenger by class
    transformed_df['FIRST_CLASS_REV_PER_PAX'] = np.where(
        transformed_df['FIRST_CLASS_PAX'] > 0,
        transformed_df['FIRST_CLASS_REVENUE'] / transformed_df['FIRST_CLASS_PAX'],
        0
    )
    
    transformed_df['PREM_ECONOMY_REV_PER_PAX'] = np.where(
        transformed_df['PREMIUM_ECONOMY_PAX'] > 0,
        transformed_df['PREM_ECONOMY_REVENUE'] / transformed_df['PREMIUM_ECONOMY_PAX'],
        0
    )
    
    transformed_df['ECONOMY_REV_PER_PAX'] = np.where(
        transformed_df['ECONOMY_PAX'] > 0,
        transformed_df['ECONOMY_REVENUE'] / transformed_df['ECONOMY_PAX'],
        0
    )
    
    # 12. Ancillary Revenue Analysis
    print("Adding ancillary revenue analysis...")
    transformed_df['ANCILLARY_REVENUE_TOTAL'] = (
        transformed_df['BAGGAGE_REVENUE'] + 
        transformed_df['OTHER_ANCILLARY_REVENUE']
    )
    
    transformed_df['ANCILLARY_REV_PER_PAX'] = np.where(
        transformed_df['PASSENGERS'] > 0,
        transformed_df['ANCILLARY_REVENUE_TOTAL'] / transformed_df['PASSENGERS'],
        0
    )
    
    transformed_df['ANCILLARY_REV_PCT'] = np.where(
        transformed_df['TOTAL_REVENUE'] > 0,
        transformed_df['ANCILLARY_REVENUE_TOTAL'] / transformed_df['TOTAL_REVENUE'] * 100,
        0
    )
    
    print(f"Transformation complete! Added {len(transformed_df.columns) - len(df.columns)} new columns.")
    print(f"Final data shape: {transformed_df.shape}")
    
    # Save transformed data if output path is provided
    if output_file_path:
        print(f"Saving transformed data to {output_file_path}...")
        transformed_df.to_excel(output_file_path, index=False)
        print("Data saved successfully!")
    
    return transformed_df

def main():
    """Main function to run the transformation"""
    # Define file paths
    input_file = '/Users/matthewotani/langchain/Analysis/RawData/AlaskaDataRaw.xlsx'
    output_file = 'Analysis/AlaskaDataTransformed.xlsx'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Run transformation
    try:
        transformed_data = transform_alaska_data(input_file, output_file)
        
        # Display summary of new columns
        original_cols = pd.read_excel(input_file).columns.tolist()
        new_cols = [col for col in transformed_data.columns if col not in original_cols]
        
        print("\n" + "="*50)
        print("SUMMARY OF NEW COLUMNS ADDED:")
        print("="*50)
        
        categories = {
            "Revenue Metrics": [col for col in new_cols if 'REVENUE' in col],
            "Cost Metrics": [col for col in new_cols if 'COST' in col],
            "Profitability Metrics": [col for col in new_cols if any(x in col for x in ['PROFIT', 'MARGIN'])],
            "Efficiency Metrics": [col for col in new_cols if any(x in col for x in ['LOAD_FACTOR', 'RASM', 'CASM'])],
            "Per-Unit Metrics": [col for col in new_cols if '_PER_' in col],
            "Flight Performance": [col for col in new_cols if any(x in col for x in ['FLIGHT', 'AVG_'])],
            "Route Analysis": [col for col in new_cols if any(x in col for x in ['ROUTE', 'DISTANCE_CATEGORY'])],
            "Time Features": [col for col in new_cols if any(x in col for x in ['YEAR', 'MONTH', 'QUARTER', 'SEASON'])],
            "Class Analysis": [col for col in new_cols if any(x in col for x in ['_PAX_PCT', '_REV_PER_PAX'])],
            "Rankings": [col for col in new_cols if 'RANK' in col],
            "Other": [col for col in new_cols if col not in sum([v for v in categories.values() if isinstance(v, list)], [])]
        }
        
        for category, cols in categories.items():
            if cols:
                print(f"\n{category}:")
                for col in cols:
                    print(f"  - {col}")
        
        print(f"\nTotal new columns added: {len(new_cols)}")
        print(f"Original columns: {len(original_cols)}")
        print(f"Final columns: {len(transformed_data.columns)}")
        
    except Exception as e:
        print(f"Error during transformation: {str(e)}")

if __name__ == "__main__":
    main()

# %%

import pandas as pd
import numpy as np

def calculate_batch_averages(file_path, start_row=96482, batch_size=96, num_batches=15, columns=['Demand_Forecast', 'Actuals']):
    """
    Calculate average for every 95 rows in specified columns for 15 batches starting from row 96482
    
    Parameters:
    file_path (str): Path to your Excel/CSV file
    start_row (int): Starting row number (default: 9653)
    batch_size (int): Number of rows per batch (default: 95)
    num_batches (int): Number of batches to process (default: 15)
    columns (list): List of columns to calculate averages for (default: ['E', 'F'])
    """
    
    # Read the file (adjust this based on your file type)
    try:
        # For Excel files
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        # For CSV files
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv)")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if columns exist
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"Column(s) {missing_columns} not found in the dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Convert to 0-based indexing (pandas uses 0-based indexing)
    start_index = start_row - 1
    
    # Calculate averages for each batch and column
    batch_results = []
    all_averages = {col: [] for col in columns}
    
    for i in range(num_batches):
        # Calculate start and end indices for current batch
        batch_start = start_index + (i * batch_size)
        batch_end = batch_start + batch_size
        
        # Check if we have enough data for this batch
        if batch_end > len(df):
            print(f"Warning: Batch {i+1} extends beyond available data")
            batch_end = len(df)
        
        # Initialize batch result
        batch_result = {
            'Batch': i + 1,
            'Start Row': batch_start + 1,  # Convert back to 1-based
            'End Row': batch_end,          # Convert back to 1-based
        }
        
        # Calculate averages for each column
        for col in columns:
            # Extract data for current batch and column
            batch_data = df.iloc[batch_start:batch_end][col]
            
            # Remove NaN values and calculate mean
            clean_data = batch_data.dropna()
            
            if len(clean_data) > 0:
                avg = clean_data.mean()
                all_averages[col].append(avg)
                batch_result[f'{col}_Valid_Values'] = len(clean_data)
                batch_result[f'{col}_Average'] = avg
            else:
                print(f"Warning: No valid data in batch {i+1} for column {col}")
                all_averages[col].append(np.nan)
                batch_result[f'{col}_Valid_Values'] = 0
                batch_result[f'{col}_Average'] = np.nan
        
        batch_results.append(batch_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(batch_results)
    
    # Display results
    print(f"\nBatch Averages for Columns {columns}:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Calculate and display overall statistics for each column
    for col in columns:
        valid_averages = [avg for avg in all_averages[col] if not np.isnan(avg)]
        if valid_averages:
            print(f"\nOverall Statistics for Column {col}:")
            print(f"Number of valid batches: {len(valid_averages)}")
            print(f"Mean of batch averages: {np.mean(valid_averages):.4f}")
            print(f"Standard deviation: {np.std(valid_averages):.4f}")
            print(f"Min batch average: {min(valid_averages):.4f}")
            print(f"Max batch average: {max(valid_averages):.4f}")
        else:
            print(f"\nNo valid data found for Column {col}")
    
    return results_df, all_averages

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "merged_demand_forecast_vs_actuals_reshaped - Copy.csv"  # or "your_data_file.csv"
    
    # Calculate batch averages
    results, all_averages = calculate_batch_averages(file_path)
    
    # Optional: Save results to a new file
    if results is not None:
        results.to_excel("batch_averages_results.xlsx", index=False)
        print("\nResults saved to 'batch_averages_results.xlsx'")
        
        # Also save separate sheets for each column's averages
        with pd.ExcelWriter("detailed_batch_averages.xlsx") as writer:
            results.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Create separate sheets for each column
            for col in ['E', 'F']:  # Adjust column names as needed
                col_data = []
                for i, avg in enumerate(all_averages.get(col, [])):
                    col_data.append({
                        'Batch': i + 1,
                        f'Column_{col}_Average': avg
                    })
                col_df = pd.DataFrame(col_data)
                col_df.to_excel(writer, sheet_name=f'Column_{col}', index=False)
        
        print("Detailed results saved to 'detailed_batch_averages.xlsx'")

# Alternative: If you want to work with data already loaded in memory
def calculate_batch_averages_from_dataframe(df, start_row=96482, batch_size=96, num_batches=15, columns=['Demand_Forecast', 'Actuals']):
    """
    Same function but works with an existing DataFrame for multiple columns
    """
    start_index = start_row - 1
    all_averages = {col: [] for col in columns}
    
    print(f"Processing {num_batches} batches of {batch_size} rows each for columns {columns}")
    print("=" * 80)
    
    for i in range(num_batches):
        batch_start = start_index + (i * batch_size)
        batch_end = batch_start + batch_size
        
        if batch_end > len(df):
            batch_end = len(df)
        
        print(f"\nBatch {i+1} (rows {batch_start+1}-{batch_end}):")
        
        for col in columns:
            batch_data = df.iloc[batch_start:batch_end][col]
            clean_data = batch_data.dropna()
            
            if len(clean_data) > 0:
                avg = clean_data.mean()
                all_averages[col].append(avg)
                print(f"  Column {col}: {avg:.4f} (valid values: {len(clean_data)})")
            else:
                print(f"  Column {col}: No valid data")
                all_averages[col].append(np.nan)
    
    return all_averages
import pandas as pd
import os

# List your CSV file paths (in order)
csv_files = [
    r"weather_data\file1.csv",
    r"weather_data\file2.csv",
    r"weather_data\file3.csv"
]

# Output path for the merged CSV
output_file = r"weather_data\andhra_pradesh_weather_merged.csv"

# Read and concatenate
dataframes = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(dataframes, ignore_index=True)

# Save to output file
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into: {output_file}")

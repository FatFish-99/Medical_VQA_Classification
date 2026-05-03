import pandas as pd
import os

# 1. Define paths
INPUT_CSV_PATH = "Phuong_report.csv"
OUTPUT_TABLE_PATH = "outputs/tables/model_comparison.csv"

print(f"Loading real metrics from: {INPUT_CSV_PATH}")

if not os.path.exists(INPUT_CSV_PATH):
    raise FileNotFoundError(f"Could not find {INPUT_CSV_PATH}.")

# 2. Read Phuong's real data
df = pd.read_csv(INPUT_CSV_PATH)

# Clean up column names just in case there are trailing spaces
df.columns = df.columns.str.strip()

# 3. Save the exact metrics to your output folder
os.makedirs(os.path.dirname(OUTPUT_TABLE_PATH), exist_ok=True)
df.to_csv(OUTPUT_TABLE_PATH, index=False)

print("\n=== Final Team Metrics Loaded ===")
print(df.to_string(index=False))
print(f"\nSaved updated table to: {OUTPUT_TABLE_PATH}")
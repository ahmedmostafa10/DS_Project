import pandas as pd
import os

def merge_datasets():
    print("Starting the merge process...")
    
    # 1. Load PropertyFinder data
    pf_file = 'propertyfinder.csv'
    if os.path.exists(pf_file):
        df_pf = pd.read_csv(pf_file)
        df_pf['source'] = 'PropertyFinder'
        print(f"Loaded {len(df_pf)} rows from PropertyFinder.")
        
        # Rename columns to match the Bayut standard
        df_pf.rename(columns={
            'bathrooms': 'bathroom',
            'is_direct_from_dev': 'is_direct_from_developer',
            'payment_method': 'payment_plan',
            'agent_is_super': 'agent_is_verified',
            'has_view_360': 'has_video'
        }, inplace=True)
    else:
        print(f"Warning: {pf_file} not found.")
        df_pf = pd.DataFrame()

    # 2. Load Bayut data
    bayut_file = 'bayut_properties_all.csv'
    if os.path.exists(bayut_file):
        df_bayut = pd.read_csv(bayut_file)
        df_bayut['source'] = 'Bayut'
        print(f"Loaded {len(df_bayut)} rows from Bayut.")
    else:
        print(f"Warning: {bayut_file} not found. Make sure to run merge_bayut_batches.py first!")
        df_bayut = pd.DataFrame()

    # 3. Combine DataFrames
    if df_pf.empty and df_bayut.empty:
        print("No datasets available to merge. Exiting.")
        return

    df_combined = pd.concat([df_pf, df_bayut], ignore_index=True)
    initial_count = len(df_combined)
    print(f"Combined total rows before deduplication: {initial_count}")

    # 4. Remove duplicates
    # Since the same property might be listed on both platforms with the exact same title, price, and location, we'll use those to deduplicate.
    # If you only meant exact row duplicates, pandas drop_duplicates() will handle that.
    df_combined.drop_duplicates(subset=['title', 'price_egp', 'location_full', 'bedrooms', 'bathroom'], keep='first', inplace=True)
    final_count = len(df_combined)
    removed_count = initial_count - final_count

    print(f"Removed {removed_count} duplicate properties across both platforms.")
    
    # 5. Save the final merged dataset
    output_filename = "all_properties_merged.csv"
    df_combined.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"✅ Successfully saved the final merged dataset with {final_count} rows to '{output_filename}'")

if __name__ == "__main__":
    merge_datasets()
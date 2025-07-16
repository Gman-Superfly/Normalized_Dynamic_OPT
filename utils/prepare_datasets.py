import os
from .download_gaia_data import download_gaia_subset

def prepare_all_datasets():
    """
    Downloads and saves multiple Gaia dataset sizes for the performance demo.
    """
    sizes = [500, 2000, 5000, 10000]
    data_dir = "data"
    print("--- Starting Dataset Preparation ---")
    
    for size in sizes:
        filename = f"gaia_data_{size}.csv"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Dataset '{filepath}' already exists. Skipping download.")
        else:
            print(f"\\n--- Downloading dataset with {size} points ---")
            try:
                # The download function will handle placing it in the data_dir
                download_gaia_subset(limit=size, filename=filename)
                print(f"Successfully created '{filepath}'.")
            except Exception as e:
                print(f"Failed to download dataset for {size} points. Error: {e}")
                break
    
    print("\\n--- Dataset Preparation Complete ---")

if __name__ == "__main__":
    prepare_all_datasets() 
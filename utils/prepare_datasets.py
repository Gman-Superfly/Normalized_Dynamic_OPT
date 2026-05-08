from __future__ import annotations

import os

from utils.download_gaia_data import GaiaDownloadError, download_gaia_subset


def prepare_all_datasets() -> None:
    """Download and persist multiple Gaia dataset sizes.

    Raises:
        GaiaDownloadError: If any dataset download step fails.
    """
    sizes = [500, 2000, 5000, 10000]
    data_dir = "data"
    print("--- Starting Dataset Preparation ---")

    for size in sizes:
        filename = f"gaia_data_{size}.csv"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Dataset '{filepath}' already exists. Skipping download.")
            continue

        print(f"\n--- Downloading dataset with {size} points ---")
        try:
            download_gaia_subset(limit=size, filename=filename)
        except GaiaDownloadError as exc:
            raise GaiaDownloadError(f"Failed while preparing '{filepath}': {exc}") from exc
        print(f"Successfully created '{filepath}'.")

    print("\n--- Dataset Preparation Complete ---")


if __name__ == "__main__":
    prepare_all_datasets()